"""Image providers: turn one `script["images"][i]` entry into a PIL image.

Order in `auto` mode: local ComfyUI -> local AUTOMATIC1111 -> AI Horde (free, crowd-sourced, open source) -> Cloudflare
Workers AI -> Pollinations. A provider that cannot serve an image raises `ProviderError`; the caller moves on to the next one
and NEVER substitutes a placeholder unless ALLOW_PLACEHOLDER_ART=1 (that silent fallback was the v1 "colour-changing
background" bug).

Network protocols follow the public docs of each service (AI Horde v2 async API, ComfyUI /prompt+/history+/view, A1111
/sdapi/v1/txt2img, Cloudflare Workers AI /ai/run). They are exercised in tests against local mock servers; the real services
are not reachable from the development sandbox, so live behaviour is verified by `python -m mangaforge doctor --live`.
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
import time
import uuid

import requests
from PIL import Image

from ..config import cfg, log
from . import prompts

CLIENT_AGENT = "MangaForge:2.0:github.com/mangaforge-studio"        # AI Horde wants name:version:contact
UA = {"User-Agent": "MangaForge/2.0"}


class ProviderError(RuntimeError):
    def __init__(self, msg, *, rc="", status=0, fatal=False):
        super().__init__(msg)
        self.rc, self.status, self.fatal = rc, status, fatal      # fatal = retrying/other models cannot help


def _to_image(data):
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"provider returned data that is not an image ({e}); head={data[:80]!r}") from e


def _round64(x):
    return max(64, int(round(x / 64.0)) * 64)


class Provider:
    name = "base"

    def available(self):
        """-> (ok: bool, why: str). Cheap check, no generation."""
        return True, ""

    def generate(self, image, script, *, seed, progress=None):
        """-> (PIL.Image RGB, meta dict). Raises ProviderError."""
        raise NotImplementedError


# ================================================================== AI Horde
HORDE_ANIME_KEYS = ("illustrious", "noobai", "noob", "waiani", "wai-ani", "animagine", "kohaku", "hassaku", "anything",
                    "counterfeit", "meina", "aniverse", "mistoon", "toonyou", "anylora", "cetus", "anime", "pony")
HORDE_AVOID = ("inpaint", "refiner", "flux", "video", "lcm only", "turbo")        # flux/turbo have other ceilings or semantics


class HordeProvider(Provider):
    name = "horde"
    # (width, height, steps, post-processing) from best to most conservative; walked down on kudos / size refusals
    LADDER = [(None, None, None, True), (None, None, None, False), (1216, 704, 24, False), (1024, 576, 22, False),
              (896, 512, 20, False)]

    def __init__(self):
        self.base = cfg.horde_url.rstrip("/")
        self.h = {"apikey": cfg.horde_key, "Client-Agent": CLIENT_AGENT, "Content-Type": "application/json", **UA}
        self._models = None

    # ---- transport
    def _req(self, method, path, tries=5, **kw):
        last = None
        timeout = kw.pop("timeout", 60)
        for i in range(tries):
            try:
                r = requests.request(method, self.base + path, headers=self.h, timeout=timeout, **kw)
            except requests.RequestException as e:
                last = ProviderError(f"AI Horde unreachable: {e}")
                time.sleep(min(30, 3 * 2 ** i) + random.random())
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = ProviderError(f"AI Horde {r.status_code}: {r.text[:160]}", status=r.status_code)
                time.sleep(min(60, 5 * 2 ** i) + random.random())
                continue
            if r.status_code >= 400:
                try:
                    j = r.json()
                except ValueError:
                    j = {}
                rc, msg = str(j.get("rc", "")), str(j.get("message", r.text[:240]))
                fatal = r.status_code in (401, 403) and rc not in ("KudosUpfront",)
                hint = ""
                if r.status_code in (401, 403) or "ip" in msg.lower():
                    hint = " - create a free API key at https://aihorde.net/register and set HORDE_API_KEY"
                raise ProviderError(f"AI Horde {r.status_code} {rc}: {msg}{hint}", rc=rc, status=r.status_code, fatal=fatal)
            return r.json() if r.content else {}
        raise last or ProviderError("AI Horde request failed")

    def available(self):
        try:
            requests.get(self.base + "/status/heartbeat", headers=self.h, timeout=15).raise_for_status()
            return True, "reachable" + ("" if cfg.horde_key != "0000000000" else " (anonymous key: slow queue)")
        except Exception as e:  # noqa: BLE001
            return False, f"unreachable: {e}"

    # ---- model discovery (names change; never hard-code)
    def models(self):
        if self._models is not None:
            return self._models
        if cfg.horde_models:
            self._models = list(cfg.horde_models)
            return self._models
        try:
            rows = self._req("GET", "/status/models", params={"type": "image", "min_count": 1}, tries=3)
        except ProviderError as e:
            log.warning("Horde model discovery failed (%s) - using generic names", e)
            self._models = ["Anything Diffusion", "Hassaku", "Counterfeit", "AlbedoBase XL (SDXL)"]
            return self._models
        scored = []
        for m in rows:
            n = str(m.get("name", ""))
            low = n.lower()
            if not n or any(b in low for b in HORDE_AVOID) or int(m.get("count") or 0) < 1:
                continue
            idx = next((i for i, k in enumerate(HORDE_ANIME_KEYS) if k in low), None)
            if idx is None:
                continue
            fam = prompts.model_family(n)
            score = (30 - idx * 2) + (6 if fam == "xl_anime" else 0) + (4 if fam == "pony" else 0) \
                + min(int(m.get("count") or 0), 12) * 0.5 - min(float(m.get("eta") or 0) / 120.0, 8)
            scored.append((score, n))
        scored.sort(reverse=True)
        self._models = [n for _, n in scored[:4]] or ["Anything Diffusion"]
        log.info("AI Horde models chosen: %s", self._models)
        return self._models

    # ---- one generation
    def _payload(self, image, script, seed, model, rung):
        fam = prompts.model_family(model)
        pos, neg = prompts.build_prompt(image, script, fam)
        w0, h0 = (_round64(cfg.art_w), _round64(cfg.art_h))
        w, h, steps, pp = self.LADDER[rung]
        w, h, steps = w or w0, h or h0, steps or cfg.horde_steps
        params = {"sampler_name": "k_euler_a", "cfg_scale": cfg.horde_cfg, "seed": str(seed), "width": w, "height": h,
                  "steps": steps, "n": 1, "karras": True, "hires_fix": False, "clip_skip": 2 if fam != "flux" else 1}
        if pp and cfg.horde_upscale:
            params["post_processing"] = ["RealESRGAN_x4plus_anime_6B"]
        loras = prompts.character_loras(image.get("characters") or [])
        if loras:
            params["loras"] = loras
        return {"prompt": f"{pos} ### {neg}" if neg else pos, "params": params, "nsfw": False, "censor_nsfw": True,
                "trusted_workers": False, "slow_workers": True, "models": [model], "r2": True, "shared": False}, fam

    def _wait(self, job_id, progress):
        t0, last_log = time.time(), 0.0
        while True:
            c = self._req("GET", f"/generate/check/{job_id}", tries=6)
            if c.get("faulted"):
                raise ProviderError("AI Horde job faulted (worker error)")
            if c.get("is_possible") is False:
                raise ProviderError("AI Horde: no worker can serve this request (model/size)", rc="NoWorker")
            if c.get("done"):
                return
            el = time.time() - t0
            if el > cfg.horde_timeout:
                try:
                    self._req("DELETE", f"/generate/status/{job_id}", tries=1)
                except ProviderError:
                    pass
                raise ProviderError(f"AI Horde timeout after {int(el)}s (queue position {c.get('queue_position')})")
            if progress and el - last_log > 30:
                last_log = el
                progress(f"horde waiting {int(el)}s queue={c.get('queue_position')} eta={c.get('wait_time')}s")
            time.sleep(min(15.0, max(3.0, float(c.get("wait_time") or 6) / 4)))

    def generate(self, image, script, *, seed, progress=None):
        errors = []
        for model in self.models():
            rung = 0
            while rung < len(self.LADDER):
                body, fam = self._payload(image, script, seed, model, rung)
                try:
                    sub = self._req("POST", "/generate/async", json=body, tries=4)
                    jid = sub.get("id")
                    if not jid:
                        raise ProviderError(f"AI Horde gave no job id: {sub}")
                    self._wait(jid, progress)
                    st = self._req("GET", f"/generate/status/{jid}", tries=4)
                    gens = [g for g in st.get("generations", []) if g.get("img")]
                    if not gens or gens[0].get("censored"):
                        raise ProviderError("AI Horde returned no image (censored or empty)", rc="Censored")
                    img = gens[0]["img"]
                    if img.startswith("http"):
                        r = requests.get(img, timeout=120, headers=UA)
                        r.raise_for_status()
                        pil = _to_image(r.content)
                    else:
                        pil = _to_image(base64.b64decode(img))
                    return pil, {"provider": "horde", "model": gens[0].get("model") or model, "family": fam,
                                 "seed": seed, "worker": gens[0].get("worker_name", ""), "rung": rung}
                except ProviderError as e:
                    if e.fatal:
                        raise
                    errors.append(f"{model}@{rung}: {e}")
                    kudos = e.rc in ("KudosUpfront", "TooManySteps", "InvalidSize", "TooManyPixels") or \
                        re.search(r"kudos|steps|resolution|pixels|size", str(e), re.I)
                    if kudos and rung + 1 < len(self.LADDER):
                        log.warning("Horde refused (%s) - degrading request to rung %d", e, rung + 1)
                        rung += 1
                        continue
                    break                                                   # other model
        raise ProviderError("AI Horde failed for every model: " + " | ".join(errors[-4:]))


# ================================================================== ComfyUI (local or remote)
class ComfyProvider(Provider):
    name = "comfyui"

    def __init__(self):
        self.base = cfg.comfy_url.rstrip("/")

    def available(self):
        try:
            requests.get(self.base + "/system_stats", timeout=4).raise_for_status()
        except Exception as e:  # noqa: BLE001
            return False, f"no ComfyUI at {self.base} ({e})"
        if not cfg.comfy_checkpoint and not self._checkpoints():
            return False, "ComfyUI is up but has no checkpoint (set COMFY_CHECKPOINT)"
        return True, "reachable"

    def _checkpoints(self):
        try:
            j = requests.get(self.base + "/object_info/CheckpointLoaderSimple", timeout=10).json()
            return j["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        except Exception:  # noqa: BLE001
            return []

    def _checkpoint(self):
        if cfg.comfy_checkpoint:
            return cfg.comfy_checkpoint
        cks = self._checkpoints()
        pref = sorted(cks, key=lambda n: (not any(k in n.lower() for k in prompts_anime_keys()), n))
        return pref[0] if pref else ""

    @staticmethod
    def graph(ckpt, pos, neg, w, h, seed, steps, cfg_scale):
        return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
                "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                                                            "latent_image": ["4", 0], "seed": seed, "steps": steps,
                                                            "cfg": cfg_scale, "sampler_name": "euler_ancestral",
                                                            "scheduler": "karras", "denoise": 1.0}},
                "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "mangaforge"}}}

    def generate(self, image, script, *, seed, progress=None):
        ck = self._checkpoint()
        fam = prompts.model_family(ck)
        pos, neg = prompts.build_prompt(image, script, fam)
        g = self.graph(ck, pos, neg, _round64(cfg.art_w), _round64(cfg.art_h), seed, cfg.horde_steps, cfg.horde_cfg)
        try:
            r = requests.post(self.base + "/prompt", json={"prompt": g, "client_id": uuid.uuid4().hex}, timeout=30)
            r.raise_for_status()
            pid = r.json()["prompt_id"]
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"ComfyUI rejected the prompt: {e}") from e
        t0 = time.time()
        while time.time() - t0 < 1800:
            try:
                h = requests.get(f"{self.base}/history/{pid}", timeout=20).json()
            except Exception:  # noqa: BLE001
                h = {}
            if pid in h:
                for out in h[pid].get("outputs", {}).values():
                    for im in out.get("images", []):
                        rr = requests.get(self.base + "/view", params={"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                                      "type": im.get("type", "output")}, timeout=60)
                        rr.raise_for_status()
                        return _to_image(rr.content), {"provider": "comfyui", "model": ck, "family": fam, "seed": seed}
                raise ProviderError("ComfyUI finished but produced no image (check the checkpoint/graph)")
            time.sleep(1.5)
        raise ProviderError("ComfyUI timeout")


def prompts_anime_keys():
    return ("anime", "illustrious", "noob", "animagine", "anything", "counterfeit", "hassaku", "pony", "meina", "wai")


# ================================================================== AUTOMATIC1111 / Forge
class A1111Provider(Provider):
    name = "a1111"

    def __init__(self):
        self.base = cfg.a1111_url.rstrip("/")

    def available(self):
        try:
            requests.get(self.base + "/sdapi/v1/options", timeout=4).raise_for_status()
            return True, "reachable"
        except Exception as e:  # noqa: BLE001
            return False, f"no A1111/Forge at {self.base} ({e})"

    def generate(self, image, script, *, seed, progress=None):
        fam = "xl_anime"
        pos, neg = prompts.build_prompt(image, script, fam)
        body = {"prompt": pos, "negative_prompt": neg, "steps": cfg.horde_steps, "cfg_scale": cfg.horde_cfg,
                "width": _round64(cfg.art_w), "height": _round64(cfg.art_h), "seed": seed, "sampler_name": "Euler a",
                "batch_size": 1, "n_iter": 1}
        try:
            r = requests.post(self.base + "/sdapi/v1/txt2img", json=body, timeout=1800)
            r.raise_for_status()
            return _to_image(base64.b64decode(r.json()["images"][0])), {"provider": "a1111", "family": fam, "seed": seed}
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"A1111 failed: {e}") from e


# ================================================================== Cloudflare Workers AI (v1 behaviour)
class CloudflareProvider(Provider):
    name = "cloudflare"
    MODELS = ["@cf/bytedance/stable-diffusion-xl-lightning", "@cf/black-forest-labs/flux-1-schnell"]

    def available(self):
        return (bool(cfg.cf_account and cfg.cf_token), "set CF_ACCOUNT_ID and CF_API_TOKEN" if not cfg.cf_token else "keys set")

    def generate(self, image, script, *, seed, progress=None):
        errs = []
        for m in self.MODELS:
            flux = "flux" in m
            fam = "flux" if flux else "xl_anime"
            pos, neg = prompts.build_prompt(image, script, fam)
            body = {"prompt": pos}
            body.update({"steps": 6} if flux else {"negative_prompt": neg, "width": 1344, "height": 768, "num_steps": 6,
                                                   "seed": seed})
            try:
                r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{cfg.cf_account}/ai/run/{m}",
                                  headers={"Authorization": f"Bearer {cfg.cf_token}", **UA}, json=body, timeout=180)
                if r.status_code != 200:
                    errs.append(f"{m}: {r.status_code} {r.text[:120]}")
                    continue
                data = r.content if r.headers.get("content-type", "").startswith("image") \
                    else base64.b64decode(r.json()["result"]["image"])
                return _to_image(data), {"provider": "cloudflare", "model": m, "family": fam, "seed": seed}
            except Exception as e:  # noqa: BLE001
                errs.append(f"{m}: {e}")
        raise ProviderError("Cloudflare failed: " + " | ".join(errs))


# ================================================================== Pollinations (keyless; API surface changes often)
class PollinationsProvider(Provider):
    name = "pollinations"
    URL = "https://image.pollinations.ai/prompt/{}"

    def available(self):
        return True, "keyless public endpoint (rate limited, may require POLLINATIONS_KEY)"

    def generate(self, image, script, *, seed, progress=None):
        pos, _ = prompts.build_prompt(image, script, "flux")
        params = {"width": 1344, "height": 768, "seed": seed, "model": "flux", "nologo": "true", "private": "true",
                  "safe": "true", "enhance": "false"}
        h = dict(UA)
        if cfg.pollinations_key:
            h["Authorization"] = f"Bearer {cfg.pollinations_key}"
        try:
            r = requests.get(self.URL.format(requests.utils.quote(pos[:900], safe="")), params=params, headers=h, timeout=240)
            r.raise_for_status()
            return _to_image(r.content), {"provider": "pollinations", "family": "flux", "seed": seed}
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Pollinations failed: {e}") from e


# ================================================================== registry
CLASSES = {"horde": HordeProvider, "comfyui": ComfyProvider, "a1111": A1111Provider, "cloudflare": CloudflareProvider,
           "pollinations": PollinationsProvider}
AUTO_ORDER = ["comfyui", "a1111", "horde", "cloudflare", "pollinations"]


def get_provider(name):
    if name not in CLASSES:
        raise ValueError(f"unknown image provider {name!r}; choose one of {', '.join(CLASSES)} or auto")
    return CLASSES[name]()


def provider_chain(name=None):
    """Providers to try, in order. `auto` keeps only the ones that answer an availability check."""
    name = (name or cfg.image_provider or "auto").lower()
    if name in ("none", "placeholder"):                        # offline demo / tests (needs ALLOW_PLACEHOLDER_ART=1)
        return []
    if name != "auto":
        return [get_provider(name)]
    chain = []
    for n in AUTO_ORDER:
        p = get_provider(n)
        ok, why = p.available()
        log.info("image provider %-12s %s %s", n, "OK  " if ok else "skip", why)
        if ok:
            chain.append(p)
    return chain


def status():
    out = {}
    for n in CLASSES:
        try:
            ok, why = get_provider(n).available()
        except Exception as e:  # noqa: BLE001
            ok, why = False, str(e)
        out[n] = {"ok": bool(ok), "info": why}
    return out
