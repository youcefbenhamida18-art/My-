"""Launch Blender headless, one process per shot (isolation: a crash or leak never loses more than one shot).

    blender -b --factory-startup -noaudio --python-exit-code 1 -P blender_scene.py -- --job <job.json>

`render_shot` returns the frames directory (contiguous f_0001.png ... one per DRAWING; "on twos" renders every `step`-th frame).
The scene script writes `done.json` with a per-feature report (ok | off | skipped:<why> | failed:<error>); a failed shot is
retried once in `safe_mode` (core 2.5D + camera only) before the dispatcher falls back to the cv engine.
"""
from __future__ import annotations

import functools
import glob
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from ..config import cfg, log

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "blender_scene.py"


class BlenderError(RuntimeError):
    pass


def _candidates():
    home = pathlib.Path.home()
    c = []
    if cfg.blender_bin:
        c.append(cfg.blender_bin)
    w = shutil.which("blender")
    if w:
        c.append(w)
    c += ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender", "/opt/blender/blender",
          "/Applications/Blender.app/Contents/MacOS/Blender"]
    for pat in (str(home / "blender*" / "blender"), str(home / "Blender*" / "blender"), "/opt/blender*/blender",
                "/usr/local/blender*/blender", str(home / "Applications" / "Blender*" / "blender"),
                "C:/Program Files/Blender Foundation/Blender */blender.exe", "D:/Program Files/Blender Foundation/Blender */blender.exe",
                "C:/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe"):
        c += sorted(glob.glob(pat), reverse=True)
    return c


@functools.lru_cache(1)
def find_blender():
    for c in _candidates():
        p = pathlib.Path(c)
        if p.is_file() and os.access(p, os.X_OK) or (p.is_file() and sys.platform == "win32"):
            return p
        w = shutil.which(str(c))
        if w:
            return pathlib.Path(w)
    return None


@functools.lru_cache(8)
def blender_version(binary):
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"Blender\s+(\d+)\.(\d+)(?:\.(\d+))?", out)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None
    except Exception as e:  # noqa: BLE001
        log.warning("could not run %s --version: %s", binary, e)
        return None


def _settings(extra=None):
    s = {"engine": cfg.render_engine, "samples": cfg.cycles_samples, "threads": cfg.render_threads, "denoise": cfg.render_denoise,
         "motion_blur": cfg.motion_blur, "view_transform": cfg.view_transform, "physics": cfg.physics,
         "volumetrics": cfg.volumetrics, "save_blend": cfg.save_blend, "export_usd": cfg.export_usd, "addons": cfg.blender_addons,
         "compositor": cfg.compositor, "passes": cfg.render_passes, "ink": cfg.ink,
         "scale": cfg.render_scale, "safe_mode": False, "root": str(HERE.parent.parent)}
    s.update(extra or {})
    return s


def _tail(path, n=40):
    try:
        lines = pathlib.Path(path).read_text("utf-8", "replace").splitlines()
    except OSError:
        return ""
    keep = [l for l in lines if re.search(r"error|traceback|exception|failed|File \"", l, re.I)]
    return "\n".join((keep[-25:] or lines[-n:]))


def _with_eye_patches(shot, fdir):
    """Blender has no OpenCV: the eye rig (blink / gaze / glints / glow / tears) is baked to RGBA patches out here."""
    if shot.get("eye_patches") is not None or not shot.get("eye_plan"):
        return shot
    try:
        from .eye_rig import bake_patches
        patches = bake_patches(shot, fdir / "eyes")
    except Exception as e:  # noqa: BLE001
        log.warning("eye patches failed for shot %s (the eyes stay static in Blender): %s", shot.get("idx"), e)
        patches = []
    return {**shot, "eye_patches": patches}


class BlenderTimeout(BlenderError):
    """Blender was stopped by the time limit (the finished drawings stay on disk and a later run continues from them)."""


def _run_once(shot, binary, settings, timeout):
    fdir = pathlib.Path(shot["paths"]["frames_dir"])
    fdir.mkdir(parents=True, exist_ok=True)
    if not settings.get("safe_mode"):
        shot = _with_eye_patches(shot, fdir)
    job = fdir / "job.json"
    job.write_text(json.dumps({"shot": shot, "settings": settings}, ensure_ascii=False), "utf-8")
    (fdir / "done.json").unlink(missing_ok=True)
    logf = fdir / "blender.log"
    cmd = [str(binary), "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "-P", str(SCENE), "--", "--job", str(job)]
    env = dict(os.environ, MANGAFORGE_ROOT=str(HERE.parent.parent), PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(logf, "wb") as lf:
        try:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, env=env, cwd=str(fdir))
        except subprocess.TimeoutExpired as e:
            raise BlenderTimeout(f"Blender timed out after {int(time.time() - t0)}s on shot {shot['idx']}") from e
    need = math.ceil(shot["frames"] / shot["step"])
    done = fdir / "done.json"
    have = len(list(fdir.glob("f_*.png")))
    if r.returncode != 0 or not done.exists() or have < need:
        raise BlenderError(f"Blender exit {r.returncode}, {have}/{need} drawings, done.json={'yes' if done.exists() else 'no'}:\n"
                           + _tail(logf))
    return json.loads(done.read_text("utf-8"))


def render_shot(shot, binary=None, *, extra_settings=None, timeout=None):
    binary = binary or find_blender()
    if not binary:
        raise BlenderError("Blender not found")
    timeout = int(timeout or os.environ.get("BLENDER_TIMEOUT", "14400"))
    try:
        rep = _run_once(shot, binary, _settings(extra_settings), timeout)
    except BlenderTimeout:
        raise
    except BlenderError as e:
        log.error("Blender failed on shot %s - retrying once in safe_mode (core 2.5D + camera only):\n%s", shot["idx"], str(e)[:1500])
        rep = _run_once(shot, binary, _settings({**(extra_settings or {}), "safe_mode": True}), timeout)
        rep.setdefault("warnings", []).append("rendered in safe_mode after a failure")
    bad = {k: v for k, v in (rep.get("features") or {}).items() if str(v).startswith("failed")}
    if bad:
        log.warning("shot %s: %d feature(s) failed (rendered without them): %s", shot["idx"], len(bad), bad)
    return pathlib.Path(shot["paths"]["frames_dir"])


# ------------------------------------------------------------------ selftest
def _selftest_shot(d):
    import cv2
    import numpy as np
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :] = (int(180 - 90 * y / h), int(140 - 30 * y / h), int(80 + 60 * y / h))
    cv2.circle(img, (470, 90), 40, (200, 235, 255), -1)
    cv2.rectangle(img, (0, 270), (w, h), (40, 70, 45), -1)
    cv2.ellipse(img, (320, 230), (60, 110), 0, 0, 360, (60, 40, 130), -1)
    cv2.circle(img, (320, 110), 34, (190, 210, 240), -1)
    cv2.imwrite(str(d / "plate.png"), img)
    cv2.imwrite(str(d / "backdrop.png"), cv2.GaussianBlur(img, (0, 0), 20))
    depth = np.zeros((h, w), np.uint8)
    for y in range(h):
        depth[y, :] = int(255 * (y / h) * 0.8)
    cv2.ellipse(depth, (320, 230), (70, 120), 0, 0, 360, 255, -1)
    depth = cv2.GaussianBlur(depth, (0, 0), 6)
    cv2.imwrite(str(d / "depth.png"), depth)
    sway = np.zeros((h, w), np.uint8)
    cv2.circle(sway, (320, 110), 50, 200, -1)
    cv2.imwrite(str(d / "sway.png"), cv2.GaussianBlur(sway, (0, 0), 8))
    rim = np.zeros((h, w), np.uint8)
    cv2.ellipse(rim, (320, 230), (62, 112), 0, 0, 360, 255, 6)
    cv2.imwrite(str(d / "rim.png"), cv2.GaussianBlur(rim, (0, 0), 2))
    fx = [{"kind": k, "amount": 0.7, "at": 0.1 if k in ("slash", "shockwave", "sparks", "debris", "lightning") else 0.0, "dur": None}
          for k in ("petals", "dust", "wind_sway", "hair_sway", "mist", "light_rays", "speed_lines", "slash", "shockwave", "embers")]
    return {"idx": 0, "id": 1, "scene": "s", "type": "action", "dur": 0.5, "frames": 12, "fps": 24, "step": 2, "size": [w, h],
            "plate": {"image": str(d / "plate.png"), "backdrop": str(d / "backdrop.png"), "depth": str(d / "depth.png"),
                      "sway": str(d / "sway.png"), "rim": str(d / "rim.png"),
                      "subject": {"x": 0.5, "y": 0.35, "depth": 0.9}, "palette": {"avg": [0.2, 0.25, 0.3], "top": [0.5, 0.6, 0.8],
                                                                                   "bottom": [0.1, 0.15, 0.1], "left": [0.2, 0.2, 0.3],
                                                                                   "right": [0.3, 0.3, 0.4], "accent": [1.0, 0.7, 0.4]},
                      "ground_y": 0.85, "parallax": 0.35, "backdrop_scale": 1.5, "sequence": None},
            "framing": {"zoom": 1.0, "cx": 0.5, "cy": 0.5},
            "camera": {"move": "orbit_r", "strength": 0.8, "dof": True, "shake": 0.2, "hit_at": 0.2, "focal": 35, "rack": None},
            "fx": fx, "post_fx": [], "grade": {"preset": "battle"}, "trans_in": "cut", "trans_out": "cut", "seed": 3, "wind": [0.5, 0.05],
            "physics": False, "auto_dress": ["pebbles", "grass", "leaves"], "title3d": _selftest_title(d),
            "paths": {"frames_dir": str(d / "frames"), "raw": "", "final": str(d / "final.mp4")}}


def _selftest_title(d):
    """Three transparent 1920x1080 layers (face / extrusion / glow) - real fonts when available, OpenCV lettering otherwise."""
    import cv2
    import numpy as np
    try:
        from .. import titles
        t = titles.render_title("MANGAFORGE", "اختبار", "series", pathlib.Path(d) / "title", "t")
        if t:
            return {"layers": {k: t[k] for k in ("face", "extrude", "glow")}, "at": 0.0, "dur": 0.45, "kind": "series"}
    except Exception as e:  # noqa: BLE001
        log.warning("selftest: title renderer unavailable (%s) - using OpenCV lettering", e)
    out = {}
    base = np.zeros((1080, 1920, 4), np.uint8)
    for k, off, col in (("glow", 0, (255, 200, 120)), ("extrude", 14, (40, 60, 120)), ("face", 0, (255, 255, 255))):
        im = base.copy()
        cv2.putText(im, "MANGAFORGE", (360 + off, 560 + off), cv2.FONT_HERSHEY_DUPLEX, 6.0, (*col, 255), 14, cv2.LINE_AA)
        if k == "glow":
            im = cv2.GaussianBlur(im, (0, 0), 18)
        p = pathlib.Path(d) / "title"
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / f"{k}.png"), im)
        out[k] = str(p / f"{k}.png")
    return {"layers": out, "at": 0.0, "dur": 0.45, "kind": "series"}


def selftest(binary=None):
    """Render a tiny shot through the real scene script (CPU, 640x360, 6 drawings) and report every feature."""
    import tempfile
    binary = binary or find_blender()
    if not binary:
        return {"ok": False, "summary": "Blender not found"}
    ver = blender_version(binary)
    d = pathlib.Path(tempfile.mkdtemp(prefix="mf_selftest_"))
    shot = _selftest_shot(d)
    t0 = time.time()
    try:
        rep = _run_once(shot, binary, _settings({"samples": 8, "scale": 1.0, "physics": False}), 1800)
    except BlenderError as e:
        return {"ok": False, "version": ver, "summary": f"render failed: {str(e)[:1500]}", "dir": str(d)}
    import cv2
    files = sorted((d / "frames").glob("f_*.png"))
    imgs = [cv2.imread(str(f)) for f in files]
    nonblack = all(i is not None and i.mean() > 4 for i in imgs)
    moving = len(imgs) > 1 and float(abs(imgs[0].astype(float) - imgs[-1].astype(float)).mean()) > 0.5
    feats = rep.get("features") or {}
    failed = {k: v for k, v in feats.items() if str(v).startswith("failed")}
    ok = bool(files) and nonblack and moving
    return {"ok": ok, "version": ver, "seconds": round(time.time() - t0, 1), "drawings": len(files), "non_black": nonblack,
            "moving": moving, "failed_features": failed, "features": feats, "dir": str(d),
            "summary": f"{len(files)} drawings in {time.time() - t0:.0f}s, {sum(1 for v in feats.values() if v == 'ok')} features ok, "
                       f"{len(failed)} failed"}
