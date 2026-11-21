"""MangaForge Studio - shared config and helpers."""
import functools
import json
import logging
import os
import pathlib
import random
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
STATE_FILE = ROOT / "state" / "series.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mangaforge")


def env(name, default):
    return os.getenv(name) or str(default)


FPS = int(env("FPS", 24))
W, H = int(env("VIDEO_W", 1920)), int(env("VIDEO_H", 1080))
VW, VH = 1080, 1920                       # vertical trailer (YouTube Shorts, up to 3 min)
EPISODE_MIN = float(env("EPISODE_MINUTES", 10))
TRAILER_SEC = float(env("TRAILER_SECONDS", 170))
WORKERS = int(env("WORKERS", os.cpu_count() or 2))
PRESET = env("X264_PRESET", "veryfast")
ACTS = 4


def retry(times=4, base=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            for i in range(times):
                try:
                    return fn(*a, **k)
                except Exception as e:  # noqa: BLE001
                    if i == times - 1:
                        raise
                    d = base * (2 ** i) + random.random()
                    log.warning("%s failed (%s) - retry in %.1fs", fn.__name__, e, d)
                    time.sleep(d)
        return wrap
    return deco


def gemini_keys():
    ks = [os.getenv(k) for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")]
    ks = [k for k in ks if k]
    random.shuffle(ks)
    return ks


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text("utf-8"))
    return {"bible": None, "episodes": [], "last_day": None}


def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), "utf-8")


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def media_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())
