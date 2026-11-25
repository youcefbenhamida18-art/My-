"""Optional AI motion for hero shots: one key picture -> a few seconds of REAL animation (image-to-video).

Why optional: it needs a GPU. The free GitHub runners have none, so by default (I2V_BACKEND=off) every shot is a Blender-animated
plate. Point the project at any ComfyUI that has a video model (LTX-Video, Wan 2.x, HunyuanVideo, AnimateDiff ... on your PC, a Colab / Kaggle /
RunPod / vast.ai box, a tunnel) and the hero shots (impacts, action, close-ups, the opening) are animated by the model; Blender still adds
the camera, 3D effects, eyes, grading and the editing on top.

The workflow is YOUR ComfyUI graph in API format (ComfyUI -> Save (API Format)); node names change between ComfyUI versions, so nothing is
hard-coded here. Put placeholders into any string input:
    {{image}}  uploaded file name      {{prompt}} / {{negative}}   {{seed}}   {{width}} {{height}}   {{frames}}   {{fps}}
A string that is exactly one placeholder is replaced by a typed value (int for seed / frames ...), otherwise it is substituted in the text.
See assets/workflows/README.md for a worked LTX-Video example.

animate() -> {"dir", "pattern", "frames", "fps"} (a numbered PNG sequence at the pipeline fps) or None.
"""
from __future__ import annotations

import copy
import json
import pathlib
import re
import shutil
import subprocess
import time
import uuid

import requests

from .config import cfg, log

NEGATIVE = ("worst quality, blurry, jpeg artifacts, deformed face, extra limbs, text, watermark, subtitles, static image, still frame, "
            "slideshow, morphing, flicker")
MOTION_HINT = ("anime, cel shaded, smooth fluid animation, dynamic camera, dramatic motion, wind in hair and clothes, sparks and energy, "
               "consistent character, clean line art")
_PH = re.compile(r"\{\{(\w+)\}\}")


class I2VError(RuntimeError):
    pass


def fill(workflow, values):
    """Deep copy of an API-format workflow with {{placeholders}} replaced (typed when the whole string is one placeholder)."""
    def sub(x):
        if isinstance(x, str):
            m = _PH.fullmatch(x.strip())
            if m and m.group(1) in values:
                return values[m.group(1)]
            return _PH.sub(lambda mm: str(values.get(mm.group(1), mm.group(0))), x)
        if isinstance(x, list):
            return [sub(i) for i in x]
        if isinstance(x, dict):
            return {k: sub(v) for k, v in x.items()}
        return x
    return sub(copy.deepcopy(workflow))


def _size(w, h, mult=32, max_side=832):
    """Video models want multiples of 32 (or 16) and modest sizes: keep the 16:9 shape, cap the long side."""
    k = min(1.0, max_side / max(w, h))
    return max(mult, int(round(w * k / mult)) * mult), max(mult, int(round(h * k / mult)) * mult)


def _frames_for(seconds, fps, step=8):
    """Many video models need 8n+1 frames."""
    n = int(round(seconds * fps))
    return max(step + 1, (n // step) * step + 1)


class ComfyI2V:
    def __init__(self):
        self.base = cfg.comfy_url.rstrip("/")
        p = pathlib.Path(cfg.i2v_workflow)
        if not p.exists():
            raise I2VError(f"I2V_WORKFLOW not found: {p} (export your graph with 'Save (API Format)' and see assets/workflows/README.md)")
        self.workflow = json.loads(p.read_text("utf-8"))

    def available(self):
        try:
            requests.get(self.base + "/system_stats", timeout=6).raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _upload(self, image):
        with open(image, "rb") as f:
            r = requests.post(self.base + "/upload/image", files={"image": (f"mf_{uuid.uuid4().hex[:8]}_{pathlib.Path(image).name}", f, "image/png")},
                              data={"overwrite": "true"}, timeout=120)
        r.raise_for_status()
        return r.json()["name"]

    def _run(self, graph, timeout):
        r = requests.post(self.base + "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=60)
        if r.status_code >= 400:
            raise I2VError(f"ComfyUI refused the workflow ({r.status_code}): {r.text[:300]}")
        pid = r.json()["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                h = requests.get(f"{self.base}/history/{pid}", timeout=30).json()
            except Exception:  # noqa: BLE001
                h = {}
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    raise I2VError("ComfyUI execution error: " + json.dumps(st.get("messages", ""))[:300])
                for out in h[pid].get("outputs", {}).values():
                    for key in ("videos", "gifs", "images"):
                        for it in out.get(key, []) or []:
                            return it
                raise I2VError("ComfyUI finished without an output (does the graph end in a Save node?)")
            time.sleep(3.0)
        raise I2VError(f"ComfyUI timeout after {timeout}s")

    def animate(self, image, prompt, out_dir, *, seconds, fps, seed):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = out_dir / "raw"
        raw.mkdir(exist_ok=True)
        from PIL import Image
        with Image.open(image) as im:
            w, h = _size(*im.size)
        frames = _frames_for(seconds, cfg.i2v_native_fps)
        vals = {"image": self._upload(image), "prompt": f"{prompt}, {MOTION_HINT}"[:900], "negative": NEGATIVE, "seed": int(seed),
                "width": w, "height": h, "frames": frames, "fps": cfg.i2v_native_fps}
        it = self._run(fill(self.workflow, vals), timeout=cfg.i2v_timeout)
        r = requests.get(self.base + "/view", params={"filename": it["filename"], "subfolder": it.get("subfolder", ""),
                                                      "type": it.get("type", "output")}, timeout=300)
        r.raise_for_status()
        src = raw / it["filename"]
        src.write_bytes(r.content)
        return to_sequence(src, out_dir, fps)


def to_sequence(src, out_dir, fps):
    """video file (or single image) -> numbered PNGs at the pipeline fps (ffmpeg resamples; a still stays a 1-frame sequence)."""
    out_dir = pathlib.Path(out_dir)
    for old in out_dir.glob("f_*.png"):
        old.unlink()
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"fps={fps}", "-start_number", "1", str(out_dir / "f_%04d.png")]
    subprocess.run(cmd, check=True, timeout=600)
    n = len(list(out_dir.glob("f_*.png")))
    if n < 4:
        raise I2VError(f"decoded only {n} frame(s) from {src.name}")
    return {"dir": str(out_dir), "pattern": "f_%04d.png", "frames": n, "fps": fps}


def animate(image, prompt, out_dir, *, seconds=4.0, fps=24, seed=1):
    """-> sequence dict or None. Backends: off | comfyui. Re-runs reuse a finished clip."""
    out_dir = pathlib.Path(out_dir)
    done = out_dir / "sequence.json"
    if done.exists():
        d = json.loads(done.read_text("utf-8"))
        if len(list(pathlib.Path(d["dir"]).glob("f_*.png"))) >= d["frames"] > 0:
            return d
    be = cfg.i2v_backend
    if be == "off":
        return None
    if be != "comfyui":
        raise I2VError(f"unknown I2V_BACKEND {be!r} (off | comfyui)")
    c = ComfyI2V()
    if not c.available():
        log.warning("I2V: no ComfyUI at %s - hero shots stay Blender-animated", c.base)
        return None
    seq = c.animate(image, prompt, out_dir, seconds=seconds, fps=fps, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(seq), "utf-8")
    return seq
