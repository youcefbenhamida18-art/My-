"""Key-art generation: provider chain + layer building. NO silent placeholder (v1 bug) - see `ArtGenerationError`.

    generate_art(script, work) -> {image_id: {"image","backdrop","depth","sway","rim","subject","palette","ground_y",
                                              "provider","seed"}}
Resumable: finished images (`<id>_layers.json` + `<id>.meta.json`) are reused, so a re-run only generates what is missing.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import cfg, log
from . import layers, prompts, providers
from .providers import ProviderError

__all__ = ["generate_art", "provider_status", "ArtGenerationError", "make_placeholder"]


class ArtGenerationError(RuntimeError):
    """No provider could produce one or more images. Message says what to configure."""


def provider_status():
    return providers.status()


# ------------------------------------------------------------------ labelled placeholder (only with ALLOW_PLACEHOLDER_ART=1)
def make_placeholder(image, seed, size=(1344, 768)):
    """Procedural dusk landscape, clearly watermarked. Lets you test the whole pipeline offline - never ships silently."""
    rng = np.random.default_rng(seed)
    w, h = size
    top = rng.integers(20, 90, 3) + np.array([20, 10, 60])
    bot = rng.integers(120, 235, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    img = (top[None, None, :] * (1 - t) + bot[None, None, :] * t) * np.ones((h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    sun = np.exp(-(((xx - w * rng.uniform(.25, .75)) ** 2 + (yy - h * .42) ** 2) / (2 * (h * .07) ** 2)))
    img += sun[..., None] * np.array([120, 90, 40])
    for k, base in enumerate((0.58, 0.68, 0.80)):
        x = np.arange(w)
        ridge = h * base + np.sin(x / (90 + 40 * k) + rng.uniform(0, 6)) * h * .04 + np.sin(x / 31 + k) * h * .01
        mask = yy > ridge[None, :]
        img[mask] = img[mask] * (0.35 + 0.1 * k) + np.array([8, 10, 22]) * (1 - 0.1 * k)
    if not image.get("no_humans"):
        cx = int(w * 0.5)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.ellipse(img, (cx, int(h * 0.74)), (int(w * .06), int(h * .22)), 0, 0, 360, (18, 18, 30), -1)
        cv2.circle(img, (cx, int(h * 0.47)), int(h * .06), (18, 18, 30), -1)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.text((14, h - 22), "PLACEHOLDER ART - no image provider worked (ALLOW_PLACEHOLDER_ART=1)", fill=(255, 255, 255))
    return pil


# ------------------------------------------------------------------ one image
def _sharpness(pil):
    g = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _generate_one(image, script, chain, seed0, progress):
    """-> (PIL, meta). Tries every provider; raises ProviderError with the full story if all fail."""
    errors = []
    variants = max(1, cfg.art_variants)
    for p in chain:
        got = []
        for v in range(variants):
            seed = seed0 + v * 7919
            for attempt in range(2):
                try:
                    if progress:
                        progress(f"{image['id']}: {p.name} seed={seed} try={attempt + 1}")
                    pil, meta = p.generate(image, script, seed=seed + attempt, progress=progress)
                    if min(pil.size) < 256:
                        raise ProviderError(f"image too small {pil.size}")
                    got.append((pil, meta))
                    break
                except ProviderError as e:
                    errors.append(f"{p.name}: {e}")
                    log.warning("%s on %s failed: %s", p.name, image["id"], e)
                    if e.fatal:
                        break
            else:
                continue
        if got:
            return max(got, key=lambda g: _sharpness(g[0]))
    raise ProviderError(" | ".join(errors[-6:]) or "no provider available")


def _paths(art_dir, iid):
    return art_dir / f"{iid}_layers.json", art_dir / f"{iid}.meta.json"


def generate_art(script, work, *, progress=None):
    art_dir = pathlib.Path(work) / "art"
    art_dir.mkdir(parents=True, exist_ok=True)
    result, todo = {}, []
    for im in script["images"]:
        lp, mp = _paths(art_dir, im["id"])
        if lp.exists() and mp.exists():
            lay, meta = json.loads(lp.read_text("utf-8")), json.loads(mp.read_text("utf-8"))
            if all(pathlib.Path(lay[k]).exists() for k in ("image", "depth", "backdrop", "sway", "rim")):
                result[im["id"]] = {**lay, "provider": meta.get("provider", "cached"), "seed": meta.get("seed", 0)}
                continue
        todo.append(im)
    if not todo:
        log.info("art: all %d images already present", len(result))
        return result

    src = [im for im in todo if im.get("source_path")]
    gen = [im for im in todo if not im.get("source_path")]
    failures = {}

    def finish(im, pil, meta, raw_name=None):
        raw = art_dir / f"{im['id']}_raw.png"
        pil.save(raw)
        lay = layers.build_layers(raw, art_dir, im["id"], no_humans=bool(im.get("no_humans")))
        lp, mp = _paths(art_dir, im["id"])
        mp.write_text(json.dumps(meta, indent=1), "utf-8")
        result[im["id"]] = {**lay, "provider": meta.get("provider"), "seed": meta.get("seed", 0)}

    for im in src:                                   # manga mode: the panel IS the art
        try:
            finish(im, Image.open(im["source_path"]).convert("RGB"), {"provider": "source", "seed": 0})
        except Exception as e:  # noqa: BLE001
            failures[im["id"]] = f"source image {im['source_path']}: {e}"

    if gen:
        chain = providers.provider_chain()
        if not chain and not cfg.allow_placeholder_art:
            raise ArtGenerationError(
                "no image provider is available. Set HORDE_API_KEY (free: https://aihorde.net/register) or run ComfyUI / "
                "AUTOMATIC1111 and set COMFY_URL / A1111_URL, or set CF_ACCOUNT_ID + CF_API_TOKEN. "
                "(ALLOW_PLACEHOLDER_ART=1 enables a labelled placeholder for offline tests.)")
        log.info("art: generating %d image(s) with %s", len(gen), [p.name for p in chain] or "placeholder")

        def work_item(im):
            seed = prompts.image_seed(im, script)
            if not chain:
                return im, make_placeholder(im, seed), {"provider": "placeholder", "seed": seed}
            try:
                pil, meta = _generate_one(im, script, chain, seed, progress)
                return im, pil, meta
            except ProviderError as e:
                if cfg.allow_placeholder_art:
                    log.error("ART: using PLACEHOLDER for %s because every provider failed (%s)", im["id"], e)
                    return im, make_placeholder(im, seed), {"provider": "placeholder", "seed": seed, "error": str(e)}
                return im, None, {"error": str(e)}

        with cf.ThreadPoolExecutor(max_workers=max(1, cfg.art_concurrency)) as ex:
            for fut in cf.as_completed([ex.submit(work_item, im) for im in gen]):
                im, pil, meta = fut.result()
                if pil is None:
                    failures[im["id"]] = meta.get("error", "unknown")
                    continue
                try:
                    finish(im, pil, meta)
                    if progress:
                        progress(f"{im['id']}: done via {meta.get('provider')}")
                except Exception as e:  # noqa: BLE001
                    failures[im["id"]] = f"layer building failed: {e}"
                    log.exception("layers failed for %s", im["id"])
    if failures:
        lines = "\n".join(f"  - {k}: {v[:300]}" for k, v in failures.items())
        raise ArtGenerationError(
            f"{len(failures)} of {len(script['images'])} images could not be produced (finished ones are saved; re-run to "
            f"resume):\n{lines}\nFix the provider (see `python -m mangaforge doctor`) - the pipeline refuses to substitute "
            "grey placeholder art.")
    return result
