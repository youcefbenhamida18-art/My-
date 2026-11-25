"""`python -m mangaforge doctor` - checks everything the pipeline needs and says exactly what to fix.

Statuses: ok | warn (works, but degraded) | FAIL (the pipeline cannot produce a good video). Exit code 1 if any FAIL.
`--live` also makes one real image request per configured provider; `--blender` runs the Blender self-test (renders a tiny
depth-mesh shot through the real scene script and checks every feature report).
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

from .config import ASSETS, cfg, gemini_keys, ROOT


def _mod(name):
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "")
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__


def run_checks(live=False, blender=False):
    rows = []

    def add(name, status, detail=""):
        rows.append((name, status, detail))

    add("python", "ok" if sys.version_info >= (3, 9) else "FAIL", sys.version.split()[0])
    for m, why, need in (("numpy", "everything", True), ("scipy", "procedural music", True), ("cv2", "cv engine, layers", True),
                         ("PIL", "images, subtitles", True), ("requests", "network", True), ("yaml", "-", False),
                         ("rembg", "accurate character matte (rim light, hair sway)", False),
                         ("transformers", "Depth-Anything-V2 depth maps", False), ("torch", "Depth-Anything-V2", False),
                         ("edge_tts", "fallback Japanese TTS", False), ("arabic_reshaper", "Pillow fallback without Raqm", False)):
        ok, v = _mod(m)
        add(f"py:{m}", "ok" if ok else ("FAIL" if need else "warn"), (v or "installed") if ok else f"missing ({why})")
    ff = shutil.which("ffmpeg")
    add("ffmpeg", "ok" if ff else "FAIL", ff or "not found - install ffmpeg")
    if ff:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        need = ["gblur", "rgbashift", "colorbalance", "vignette", "blend", "curves", "eq", "noise", "negate", "colorchannelmixer"]
        miss = [f for f in need if f" {f} " not in out]
        add("ffmpeg filters", "ok" if not miss else "FAIL", "all present" if not miss else f"missing {miss}")
        add("ffmpeg libass (Arabic subtitles)", "ok" if " ass " in out else "warn", "ASS burn-in" if " ass " in out else "no libass: slower PNG overlay fallback")
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        add("ffmpeg libx264", "ok" if "libx264" in enc else "FAIL", "")
    from . import subs
    sc = subs.self_check()
    add("arabic font", "ok" if sc["font"] else "FAIL", f"{sc['family']} ({sc['font']})" if sc["font"] else "put Cairo/Tajawal/Noto Sans Arabic .ttf into assets/fonts")
    add("arabic shaping", "ok" if sc["png_path"] != "UNAVAILABLE" else "FAIL", f"png:{sc['png_path']} ass:{sc['ass_path']}")
    from .render import blender_runner
    b = None
    try:
        b = blender_runner.find_blender()
    except Exception as e:  # noqa: BLE001
        add("blender", "warn", f"search failed: {e}")
    if b:
        v = blender_runner.blender_version(b)
        add("blender", "ok" if v and v >= (4, 2) else "warn", f"{b} version {'.'.join(map(str, v)) if v else '?'}")
    else:
        add("blender", "warn", "not found -> OpenCV 2.5D engine will be used. Install Blender 4.2 LTS or newer and set BLENDER_BIN for the cinematic render")
    if blender and b:
        try:
            r = blender_runner.selftest(b)
            add("blender selftest", "ok" if r.get("ok") else "FAIL", r.get("summary", ""))
        except Exception as e:  # noqa: BLE001
            add("blender selftest", "FAIL", str(e)[:300])
    keys = gemini_keys()
    add("gemini key (story)", "ok" if keys else ("warn" if cfg.source == "manga" else "FAIL"), f"{len(keys)} key(s)" if keys else "set GEMINI_API_KEY (free at aistudio.google.com/apikey)")
    from . import voice
    vp = voice.available_providers()
    add("voice", "ok" if (vp["voicevox"] or vp["human"]) else ("warn" if vp["edge"] else "FAIL"),
        ", ".join(k for k, v in vp.items() if v and k != "silent") or "no engine: lines will be silent! run VOICEVOX or pip install edge-tts")
    from .art import provider_status
    ps = provider_status()
    usable = [k for k, v in ps.items() if v["ok"] and k != "pollinations"]
    add("image providers", "ok" if usable else ("warn" if cfg.allow_placeholder_art or ps["pollinations"]["ok"] else "FAIL"),
        "; ".join(f"{k}:{'ok' if v['ok'] else 'no'}" for k, v in ps.items()) + ("" if usable else "  -> set HORDE_API_KEY (free) or start ComfyUI/A1111"))
    add("horde key", "ok" if cfg.horde_key != "0000000000" else "warn", "registered key" if cfg.horde_key != "0000000000" else
        "anonymous key: lowest priority + may be blocked on cloud IPs. Register free: https://aihorde.net/register")
    for sub in ("fonts", "music", "sfx", "voices", "overlays", "models", "manga"):
        d = ASSETS / sub
        n = len([p for p in d.glob("*") if p.name != ".gitkeep"]) if d.exists() else 0
        add(f"assets/{sub}", "ok", f"{n} file(s)" + (" (optional)" if n == 0 else ""))
    du = shutil.disk_usage(ROOT)
    add("disk", "ok" if du.free > 8e9 else "warn", f"{du.free / 1e9:.1f} GB free")
    if live:
        from .art import providers
        for p in providers.provider_chain():
            try:
                img = {"id": "doctor", "scene": "s", "prompt": "a lone swordsman on a hill at sunset, anime key visual", "characters": [],
                       "framing": "wide", "palette": "", "no_humans": False}
                pil, meta = p.generate(img, {"style": "anime", "characters": [], "images": [img]}, seed=1)
                add(f"live:{p.name}", "ok", f"{pil.size} model={meta.get('model', '')}")
            except Exception as e:  # noqa: BLE001
                add(f"live:{p.name}", "FAIL", str(e)[:300])
    return rows


def print_report(rows):
    w = max(len(r[0]) for r in rows)
    bad = 0
    for n, s, d in rows:
        bad += s == "FAIL"
        print(f"[{s:^4}] {n:<{w}}  {d}")
    print("\n" + ("All critical checks passed." if not bad else f"{bad} critical problem(s): fix the FAIL lines above."))
    return 1 if bad else 0
