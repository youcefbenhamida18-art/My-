"""MangaForge v2 - settings, paths, logging and small shared helpers (stdlib only, import-safe everywhere).

Everything is driven by environment variables (or a local `.env` file) so the same code runs on a laptop with a GPU
and on a free GitHub Actions runner. Read settings through the `cfg` object: `from .config import cfg`.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUT = ROOT / "out"
WORK_ROOT = ROOT / "work"
STATE_FILE = ROOT / "state" / "series.json"
STATE_DIR = ROOT / "state"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mangaforge")


# ------------------------------------------------------------------ env helpers
def _load_dotenv():
    """Tiny .env reader (no dependency). Real environment variables always win."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def env(name, default=""):
    v = os.getenv(name)
    return v if v not in (None, "") else str(default)


def env_int(name, default):
    try:
        return int(float(env(name, default)))
    except ValueError:
        return int(default)


def env_float(name, default):
    try:
        return float(env(name, default))
    except ValueError:
        return float(default)


def env_bool(name, default=False):
    return env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """Plain attribute bag, rebuilt by `cfg.reload()` (tests change the environment, then reload)."""

    def __init__(self):
        self.reload()

    def reload(self):
        # ---- output format
        self.fps = env_int("FPS", 24)
        self.anim_step = max(1, env_int("ANIM_STEP", 2))           # 2 = "on twos": 12 unique drawings/s, the anime look
        self.width = env_int("VIDEO_W", 1920)
        self.height = env_int("VIDEO_H", 1080)
        self.short_seconds = env_float("SHORT_SECONDS", 120)       # one daily short
        self.parts_per_episode = env_int("PARTS_PER_EPISODE", 7)   # 7 shorts = 1 weekly episode
        self.vertical = env_bool("VERTICAL", True)                 # also export a 9:16 cut (Shorts/Reels/TikTok) next to the landscape video
        self.x264_preset = env("X264_PRESET", "veryfast")
        self.crf = env_int("CRF", 16)
        # ---- rendering (Blender)
        self.blender_bin = env("BLENDER_BIN", "")
        self.render_engine = env("RENDER_ENGINE", "auto").lower()  # auto | cycles | eevee | cv   (cv = no-Blender fallback)
        self.cycles_samples = env_int("CYCLES_SAMPLES", 24)
        self.render_threads = env_int("RENDER_THREADS", os.cpu_count() or 2)
        self.render_workers = env_int("RENDER_WORKERS", 1)         # parallel Blender processes (1 is right for Cycles)
        self.render_strict = env_bool("RENDER_STRICT", False)       # True: a Blender failure aborts instead of falling back to the cv engine
        self.render_denoise = env_bool("RENDER_DENOISE", True)
        self.motion_blur = env_bool("MOTION_BLUR", True)
        self.render_scale = env_float("RENDER_SCALE", 1.0)         # <1 for quick drafts (0.5 = 960x540)
        self.physics = env_bool("PHYSICS_FX", False)               # tier 3: real simulations - Mantaflow smoke + fire, cloth banner (slow on CPU)
        self.save_blend = env_bool("SAVE_BLEND", False)            # keep one relative-path .blend per shot (open it in the Blender GUI to polish)
        self.view_transform = env("VIEW_TRANSFORM", "Standard")    # Standard | AgX | Filmic | ACES (ACES needs an OCIO config in $OCIO)
        self.volumetrics = env_bool("VOLUMETRICS", False)          # real scattering volume + spot light in epic shots (slow on CPU)
        self.export_usd = env_bool("EXPORT_USD", False)            # also write a .usd per shot
        self.blender_addons = [a.strip() for a in env("BLENDER_ADDONS", "").split(",") if a.strip()]
        self.compositor = env_bool("COMPOSITOR", True)             # Blender compositor: Glare bloom on the HDR emission (before the ffmpeg grade)
        self.render_passes = env_bool("RENDER_PASSES", False)      # also write Z + mist 16-bit EXR passes per drawing (work/<job>/frames/*/passes)
        self.ink = env_bool("INK", True)                           # Freestyle ink lines on the 3D dressing (pebbles, leaves)
        # ---- art
        self.image_provider = env("IMAGE_PROVIDER", "auto").lower()  # auto|horde|pollinations|comfyui|a1111|cloudflare
        self.art_w = env_int("ART_W", 1344)                        # native generation size (16:9-ish, multiples of 64)
        self.art_h = env_int("ART_H", 768)
        self.art_target_w = env_int("ART_TARGET_W", 3840)          # key art is upscaled to about this width
        self.art_variants = env_int("ART_VARIANTS", 1)             # candidates per key image (best-of-N picks by sharpness)
        self.allow_placeholder_art = env_bool("ALLOW_PLACEHOLDER_ART", False)
        self.art_concurrency = max(1, env_int("ART_CONCURRENCY", 3))   # images requested in parallel
        self.art_plate_format = env("ART_PLATE_FORMAT", "jpg").lower()  # jpg (small, default) | png (lossless, big)
        self.horde_url = env("HORDE_URL", "https://aihorde.net/api/v2")
        self.horde_key = env("HORDE_API_KEY", "0000000000")        # anonymous key works but queues last
        self.horde_models = [m.strip() for m in env("HORDE_MODELS", "").split(",") if m.strip()]
        self.horde_timeout = env_int("HORDE_TIMEOUT", 1500)        # seconds to wait for ONE image
        self.horde_upscale = env_bool("HORDE_UPSCALE", True)       # RealESRGAN anime 4x via the Horde post-processors
        self.horde_steps = env_int("HORDE_STEPS", 28)
        self.horde_cfg = env_float("HORDE_CFG", 6.5)
        self.comfy_url = env("COMFY_URL", "http://127.0.0.1:8188")
        self.comfy_checkpoint = env("COMFY_CHECKPOINT", "")        # file name of an anime SDXL checkpoint in ComfyUI
        self.a1111_url = env("A1111_URL", "http://127.0.0.1:7860")
        self.cf_account = env("CF_ACCOUNT_ID", "")
        self.cf_token = env("CF_API_TOKEN", "")
        self.pollinations_key = env("POLLINATIONS_KEY", "")
        self.depth_model = env("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
        # ---- image-to-video (optional GPU tier: hero shots get real motion)
        self.i2v_backend = env("I2V_BACKEND", "off").lower()       # off | comfyui
        self.i2v_workflow = env("I2V_WORKFLOW", str(ASSETS / "workflows" / "i2v_api.json"))
        self.i2v_budget = env_int("I2V_BUDGET", 6)                 # max hero shots per short that get AI motion
        self.i2v_seconds = env_float("I2V_SECONDS", 4.0)
        self.i2v_native_fps = env_int("I2V_NATIVE_FPS", 24)         # frame rate the video model was built for (LTX-Video 24, Wan 16)
        self.i2v_timeout = env_int("I2V_TIMEOUT", 3600)             # seconds to wait for one clip
        # ---- voices
        self.voice_provider = env("VOICE_PROVIDER", "auto").lower()  # auto | voicevox | edge | silent
        self.voicevox_url = env("VOICEVOX_URL", "http://127.0.0.1:50021")
        # ---- story
        self.story_lang_voice = env("VOICE_LANG", "ja")
        self.source = env("STORY_SOURCE", "manga").lower()         # manga (MangaDex, default) | original (invented story)
        self.manga_dir = env("MANGA_DIR", str(ASSETS / "manga"))   # extra local manga pages (used only if MangaDex is unreachable)
        self.manga_id = env("MANGA_ID", "")                        # force a MangaDex title (uuid) instead of the automatic pick
        self.manga_title = env("MANGA_TITLE", "")                  # or search a title by name
        self.manga_like = env("MANGA_LIKE", "Kimetsu no Yaiba")    # the series the automatic pick should resemble (its tags are the profile)
        self.manga_lang = env("MANGA_LANG", "en")                  # translation language read by the vision model
        self.manga_chapters_per_episode = max(1, env_int("MANGA_CHAPTERS_PER_EPISODE", 2))
        self.manga_min_chapters = env_int("MANGA_MIN_CHAPTERS", 8)  # a candidate needs at least this many readable chapters
        self.manga_exclude_tags = [t.strip() for t in env("MANGA_EXCLUDE_TAGS", "Sexual Violence,Loli,Shota,Incest,Boys' Love,Girls' Love,Harem,Gore").split(",") if t.strip()]
        self.manga_vision_pages = max(4, env_int("MANGA_VISION_PAGES", 14))   # pages per vision request
        self.pilot = env_bool("PILOT", True)                       # episode 1 / part 1 gets the cold-open + title-card + hook treatment
        # ---- eyes (the signature of the look): detection, enhancement, blinks, catch-lights, glow, extreme close-ups
        self.eyes = env_bool("EYES", True)
        self.eye_refine = env_bool("EYE_REFINE", True)             # second img2img pass over the face of close-ups (needs a provider with img2img)
        self.eye_refine_max = env_int("EYE_REFINE_MAX", 8)         # at most this many images per short get the extra pass
        self.art_refs = env_bool("ART_REFS", True)                 # manga/cover character crops guide hero portraits (img2img) when supported
        self.publish_tz = env("PUBLISH_TZ", "Asia/Riyadh")
        self.peak_slots = env("PEAK_SLOTS", "19:00,21:30")
        self.dry_run = env_bool("DRY_RUN", False)
        self.auto_weekly = env_bool("AUTO_WEEKLY", True)           # join the 7 shorts as soon as the 7th exists

    def __repr__(self):
        return "Settings(%s)" % ", ".join(f"{k}={v!r}" for k, v in sorted(vars(self).items())
                                          if "key" not in k and "token" not in k)


cfg = Settings()


def render_size():
    """Final video size, honouring RENDER_SCALE for drafts (always even numbers)."""
    w = int(cfg.width * cfg.render_scale) // 2 * 2
    h = int(cfg.height * cfg.render_scale) // 2 * 2
    return max(2, w), max(2, h)


# ------------------------------------------------------------------ generic helpers
def retry(times=4, base=2.0, exceptions=(Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            for i in range(times):
                try:
                    return fn(*a, **k)
                except exceptions as e:  # noqa: PERF203
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


def run(cmd, **kw):
    """subprocess.run with str() on every argument and check=True."""
    kw.setdefault("check", True)
    return subprocess.run([str(c) for c in cmd], **kw)


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def slug(s, default="x"):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    return s or default


def work_dir(job):
    d = WORK_ROOT / str(job)
    d.mkdir(parents=True, exist_ok=True)
    return d


def rebase_paths(obj, job):
    """A job's plan (script/voices/art/shots JSON) stores absolute paths. Written on a CI runner and read on another runner or on your PC
    they would point nowhere (or, for the render outputs, into a directory that does not exist here): every path inside `work/<job>/`
    is re-rooted to THIS checkout's work/<job>/."""
    base = WORK_ROOT / str(job)
    marker = f"/work/{job}/"

    def fix(x):
        if isinstance(x, str):
            y = x.replace("\\", "/")
            if marker in y:
                return str(base / y.split(marker, 1)[1])
            return x
        if isinstance(x, list):
            return [fix(i) for i in x]
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        return x
    return fix(obj)


# ------------------------------------------------------------------ series state (continuity between days)
def new_state():
    return {"version": 2, "bible": None, "manga": None, "episode": 1, "part": 1, "week_outline": None,
            "shorts": [], "episodes": [], "last_day": None}


def load_state():
    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text("utf-8"))
        base = new_state()
        base.update(st)                      # v1 files (bible/episodes/last_day) load fine
        return base
    return new_state()


def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), "utf-8")
