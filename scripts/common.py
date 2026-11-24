"""Small shared helpers (env, logging, hashing, state file)."""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
WORK = ROOT / "work"
OUT = ROOT / "out"
STATE_FILE = ROOT / "state" / "uploaded.json"

_T0 = time.time()


def env(name, default=None, cast=str):
    """Read an env var; empty string counts as 'not set'."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        if cast is bool:
            return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
        return cast(v)
    except Exception:
        return default


def log(*a):
    mins = (time.time() - _T0) / 60
    print(f"[{mins:6.1f}m]", *a, flush=True)


def group(title):
    print(f"::group::{title}", flush=True)


def endgroup():
    print("::endgroup::", flush=True)


def sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def slugify(s: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return (s or "lesson")[:n]


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def die(msg: str):
    print(f"::error::{msg}", flush=True)
    raise SystemExit(msg)
