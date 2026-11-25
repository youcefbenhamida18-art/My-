"""Shared vocabulary + defensive normalisation for `script.json` (story/manga -> art/voice/director).

LLM output is messy. `normalize_script()` is the single gate: after it, every later stage may assume enums are valid,
ids are unique/sequential and every shot points at an existing image. See SPEC.md for the full data contract.
"""
from __future__ import annotations

import re

SCHEMA_VERSION = 2

SHOT_TYPES = ("establishing", "wide", "medium", "closeup", "insert", "action", "impact", "reaction", "transition")
CAMERA_MOVES = ("static", "drift", "push_in", "pull_out", "pan_l", "pan_r", "tilt_up", "tilt_dn", "orbit_l", "orbit_r",
                "crane_up", "crane_dn", "whip_l", "whip_r", "handheld", "punch_in", "dolly_zoom")
# effects built inside Blender (geometry / shaders / camera) ...
FX_BLENDER = ("petals", "embers", "snow", "rain", "dust", "fireflies", "mist", "light_rays", "speed_lines", "slash",
              "shockwave", "lightning", "sparks", "wind_sway", "hair_sway", "heat_haze", "water_ripple", "glow_pulse",
              # tier 2: real 3D dressing / physics-style elements built in Blender
              "grass", "pebbles", "leaves", "debris", "glass_shards", "lens_drops", "lens_flare", "aura", "swirl",
              # tier 3: heavy simulations (only when cfg.physics is on; otherwise skipped with a log line)
              "smoke", "fire", "splash", "cloth_flag", "volume_fog")
# ... and effects applied on the rendered frames with ffmpeg (post.py)
FX_POST = ("flash", "impact_frame", "chromatic", "glitch", "anamorphic", "hit_stop", "speed_ramp", "overlay")
FX_ALL = FX_BLENDER + FX_POST
GRADES = ("calm", "warm", "tense", "dark", "battle", "sad", "epic", "mystic", "dawn", "night")
EMOTIONS = ("neutral", "calm", "angry", "sad", "happy", "shout", "whisper", "fear", "cold", "tender")
TRANSITIONS = ("cut", "fade", "flash", "whip", "dip_black", "dip_white")
MUSIC = ("calm", "tense", "dark", "battle", "sad", "epic", "mystic", "warm", "none")
SFX = ("boom", "hit", "slash", "whoosh", "footsteps", "thunder", "wind", "riser", "shimmer", "clash", "heartbeat",
       "rain", "fire", "glass")
VOICE_STYLES = ("hero", "villain", "elder", "young", "narrator", "girl", "boy")
FRAMINGS = ("wide", "medium", "closeup")

# which FX kinds suit which grade/mood by default (director uses it when the script leaves fx empty)
DEFAULT_FX = {
    "calm": ["dust", "wind_sway"], "warm": ["petals", "wind_sway"], "tense": ["mist", "dust"], "dark": ["mist", "embers"],
    "battle": ["embers", "speed_lines"], "sad": ["snow", "dust"], "epic": ["light_rays", "embers"],
    "mystic": ["fireflies", "mist"], "dawn": ["light_rays", "dust"], "night": ["fireflies", "mist"],
}


class ScriptError(ValueError):
    """The script has nothing usable (no shots / no images)."""


# ------------------------------------------------------------------ tiny helpers
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def as_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def as_int(x, default=0):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return int(default)


def pick(value, allowed, default=None):
    """Return value if it is one of `allowed` (case/space-insensitive), else default."""
    v = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in allowed else default


def pick_list(values, allowed, limit=6):
    out = []
    if isinstance(values, str):
        values = re.split(r"[,\s|/]+", values)
    for v in values or []:
        p = pick(v, allowed)
        if p and p not in out:
            out.append(p)
    return out[:limit]


def sid(x, default="x"):
    s = re.sub(r"[^a-z0-9_]+", "_", str(x or "").strip().lower()).strip("_")
    return s or default


def text(x, limit=1200):
    return re.sub(r"\s+", " ", str(x or "")).strip()[:limit]


# ------------------------------------------------------------------ normalisation
def normalize_characters(chars):
    out, seen = [], set()
    for c in chars or []:
        if not isinstance(c, dict):
            continue
        cid = sid(c.get("id") or c.get("name_ja") or c.get("name_ar") or f"char{len(out) + 1}")
        if cid in seen:
            continue
        seen.add(cid)
        out.append({
            "id": cid, "name_ja": text(c.get("name_ja"), 60), "name_ar": text(c.get("name_ar"), 60),
            "gender": pick(c.get("gender"), ("male", "female"), "male"),
            "voice": pick(c.get("voice"), VOICE_STYLES, "narrator"),
            "look": text(c.get("look"), 500),
            "eyes": text(c.get("eyes"), 240),                       # eye tags (colour/shape/pupil/lashes) - the signature detail
            "role": text(c.get("role"), 40), "personality": text(c.get("personality"), 200),
            "ref": c.get("ref") if isinstance(c.get("ref"), dict) else None,   # manga/cover crops {"body","face","colour"}
        })
    return out


def normalize_line(line, char_ids):
    if not isinstance(line, dict):
        return None
    ja, ar = text(line.get("ja"), 300), text(line.get("ar"), 300)
    if not (ja or ar):
        return None
    sp = sid(line.get("speaker"), "narrator")
    return {"speaker": sp if sp in char_ids else "narrator",
            "ja": ja or ar, "ar": ar or ja,
            "emotion": pick(line.get("emotion"), EMOTIONS, "neutral"),
            "intensity": clamp(as_int(line.get("intensity"), 2), 1, 5)}


def normalize_script(raw, characters=None):
    """-> clean script dict (see SPEC.md). Raises ScriptError if no usable shot remains."""
    if not isinstance(raw, dict):
        raise ScriptError("script is not an object")
    chars = normalize_characters(characters if characters is not None else raw.get("characters"))
    char_ids = {c["id"] for c in chars}

    meta_in = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    meta = {
        "title_ja": text(meta_in.get("title_ja"), 100), "title_ar": text(meta_in.get("title_ar"), 100),
        "description_ar": text(meta_in.get("description_ar"), 1500),
        "summary": text(meta_in.get("summary") or raw.get("summary"), 800),
        "tags": [text(t, 40) for t in (meta_in.get("tags") or [])][:15],
        "hashtags": [text(t, 40) for t in (meta_in.get("hashtags") or [])][:8],
    }
    for k in ("part", "of", "episode"):
        if k in meta_in:
            meta[k] = as_int(meta_in[k], 1)

    images, img_ids = [], set()
    for i, im in enumerate(raw.get("images") or []):
        if not isinstance(im, dict):
            continue
        iid = sid(im.get("id"), f"i{i + 1:02d}")
        while iid in img_ids:
            iid += "_"
        prompt = text(im.get("prompt"), 900)
        src = im.get("source_path")
        if not prompt and not src:
            continue
        img_ids.add(iid)
        images.append({
            "id": iid, "scene": sid(im.get("scene"), "s1"), "prompt": prompt,
            "characters": [c for c in (sid(x) for x in im.get("characters") or []) if c in char_ids],
            "framing": pick(im.get("framing"), FRAMINGS, "wide"),
            "palette": text(im.get("palette"), 120),
            "no_humans": bool(im.get("no_humans")),
            "source_path": str(src) if src else None,
            "hero": sid(im.get("hero"), "") if sid(im.get("hero"), "") in char_ids else "",   # portrait guided by this character's reference crop
        })
    if not images:
        raise ScriptError("script has no images")

    shots, last_img = [], images[0]["id"]
    for sh in raw.get("shots") or []:
        if not isinstance(sh, dict):
            continue
        iid = sid(sh.get("image"), "")
        iid = iid if iid in img_ids else last_img
        last_img = iid
        fr = sh.get("framing") if isinstance(sh.get("framing"), dict) else {}
        cam = sh.get("camera") if isinstance(sh.get("camera"), dict) else {}
        line = normalize_line(sh.get("line"), char_ids)
        stype = pick(sh.get("type"), SHOT_TYPES, "dialogue" if line else "medium")
        if stype == "dialogue":
            stype = "medium"
        shots.append({
            "id": len(shots) + 1, "image": iid, "type": stype,
            "dur": clamp(as_float(sh.get("dur"), 3.0), 0.8, 14.0),
            "framing": {"zoom": clamp(as_float(fr.get("zoom"), 1.0), 1.0, 3.2),
                        "cx": clamp(as_float(fr.get("cx"), 0.5), 0.0, 1.0),
                        "cy": clamp(as_float(fr.get("cy"), 0.5), 0.0, 1.0)},
            "camera": {"move": pick(cam.get("move"), CAMERA_MOVES, None),
                       "strength": clamp(as_float(cam.get("strength"), 0.5), 0.0, 1.0)},
            "fx": pick_list(sh.get("fx"), FX_ALL, 4),
            "grade": pick(sh.get("grade"), GRADES, None),
            "line": line,
            "sfx": pick_list(sh.get("sfx"), SFX, 3),
            "music": pick(sh.get("music"), MUSIC, None),
            "trans_in": pick(sh.get("trans_in"), TRANSITIONS, None),
            "title_card": pick(sh.get("title_card"), ("series", "episode"), None),   # 3D title sting on this shot
            "eyes": pick(sh.get("eyes"), ("closeup", "glow", "tears"), None),         # eye-direction hint
            "notes": text(sh.get("notes"), 200),
        })
    if not shots:
        raise ScriptError("script has no shots")
    return {"version": SCHEMA_VERSION, "meta": meta, "characters": chars, "style": text(raw.get("style"), 400),
            "images": images, "shots": shots}


def script_stats(script):
    lines = sum(1 for s in script["shots"] if s.get("line"))
    return {"images": len(script["images"]), "shots": len(script["shots"]), "lines": lines,
            "planned_seconds": round(sum(s["dur"] for s in script["shots"]), 1)}
