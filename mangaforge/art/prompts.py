"""Anime prompt builder: model-family aware tags, character look injection, stable seeds.

Rules baked in here (so no caller can forget them):
* never name a studio, a show or a character - style is described with generic production terms only;
* never ask for text/letters in the picture (subtitles are rendered later);
* character consistency = the same booru-style `look` tags in every prompt + stable per-character seeds
  (+ optional LoRAs from the CHARACTER_LORAS env var, honoured by providers that support them).
"""
from __future__ import annotations

import hashlib
import json
import re

from ..config import env, log

FAMILIES = ("xl_anime", "pony", "sd15", "flux", "generic")

# Horde caps the whole "prompt ### negative" string; stay well below it.
MAX_COMBINED = 950
MAX_NEGATIVE = 330

STYLE_CORE = ("anime screencap, cel shading, flat colors, clean sharp lineart, vibrant saturated palette, "
              "cinematic composition, dramatic lighting, volumetric light, depth of field, highly detailed background, "
              "dynamic angle")
STYLE_NATURAL = ("Anime key frame, cel shaded with clean sharp lineart and flat vibrant colors, cinematic composition, "
                 "dramatic lighting with volumetric light, depth of field, richly detailed painted background.")

QUALITY = {
    "xl_anime": "masterpiece, best quality, very aesthetic, absurdres, newest",
    "pony": "score_9, score_8_up, score_7_up, source_anime",
    "sd15": "masterpiece, best quality, ultra-detailed, illustration",
    "generic": "masterpiece, best quality, highly detailed",
    "flux": "",
}

_NEG_COMMON = ("text, letters, words, caption, subtitle, speech bubble, border, frame, logo, watermark, signature, "
               "username, nsfw, nude, gore, 3d, realistic, photo")
NEGATIVE = {
    "xl_anime": "lowres, bad anatomy, bad hands, missing finger, extra digits, fewer digits, cropped, worst quality, "
                "low quality, low score, jpeg artifacts, blurry, " + _NEG_COMMON,
    "pony": "score_6, score_5, score_4, lowres, bad anatomy, bad hands, blurry, " + _NEG_COMMON,
    "sd15": "(worst quality, low quality:1.4), lowres, bad anatomy, bad hands, extra fingers, missing fingers, "
            "cropped, jpeg artifacts, blurry, " + _NEG_COMMON,
    "generic": "lowres, bad anatomy, bad hands, worst quality, low quality, blurry, " + _NEG_COMMON,
    "flux": "",
}

FRAMING_TAGS = {
    "wide": "wide shot, full body, establishing shot, landscape composition",
    "medium": "medium shot, cowboy shot, upper body",
    "closeup": "close-up, portrait, face focus, detailed eyes",
}
FRAMING_NATURAL = {
    "wide": "wide establishing shot, full body visible",
    "medium": "medium shot from the waist up",
    "closeup": "close-up portrait with detailed eyes",
}

# eyes are the signature of the look: extra weight for any image where a face is large enough to matter
EYE_QUALITY = {
    "xl_anime": "detailed eyes, beautiful detailed iris, sharp eyelashes, eye highlights, expressive eyes, symmetrical eyes",
    "pony": "detailed eyes, beautiful detailed iris, eye highlights, expressive eyes",
    "sd15": "(detailed eyes:1.2), beautiful detailed iris, eyelashes, eye highlights, symmetrical eyes",
    "generic": "detailed eyes, beautiful iris, eye highlights, symmetrical eyes",
}
EYE_NEG = "bad eyes, asymmetrical eyes, cross-eyed, blurry eyes"
EYE_NATURAL = "Large, expressive, perfectly drawn anime eyes with a layered iris, crisp eyelashes and clear highlights."

_XL_HINTS = ("illustrious", "noob", "animagine", "kohaku", "waiani", "wai-", "hassaku xl", "xl")
_SD15_HINTS = ("anything", "counterfeit", "meina", "anylora", "hassaku", "aniverse", "mistoon", "toonyou", "waifu",
               "anime", "abyss", "orange", "pastel", "cetus", "revanimated", "dreamshaper", "deliberate", "sd1",
               "stable diffusion", "diffusion")


def model_family(name):
    """Prompt-tag family for a model/checkpoint name. Unknown -> 'xl_anime' (modern tags are harmless on most models)."""
    n = (name or "").lower()
    if not n:
        return "xl_anime"
    if "pony" in n or "autismmix" in n:
        return "pony"
    if "flux" in n or "chroma" in n:
        return "flux"
    if any(k in n for k in _XL_HINTS):
        return "xl_anime"
    if any(k in n for k in _SD15_HINTS):
        return "sd15"
    return "generic"


# ------------------------------------------------------------------ seeds
def stable_seed(*parts):
    """Deterministic 31-bit seed from arbitrary parts (same inputs -> same seed on every machine)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1) + 1


def character_seed(char_id):
    return stable_seed("character", char_id)


def image_seed(image, script, base=0):
    """Portrait of a single character -> that character's stable seed (keeps the face family similar);
    everything else -> a per-image deterministic seed."""
    chars = image.get("characters") or []
    if len(chars) == 1 and image.get("framing") == "closeup":
        return character_seed(chars[0])
    meta = (script or {}).get("meta") or {}
    return stable_seed("image", meta.get("episode", 0), meta.get("part", 0), image.get("id", ""), base)


# ------------------------------------------------------------------ LoRAs (Horde `loras`, optional)
def character_loras(char_ids):
    """CHARACTER_LORAS='{"kaito":[{"name":"123456","strength":0.8,"is_version":true}]}' -> Horde loras payload."""
    raw = env("CHARACTER_LORAS", "")
    if not raw:
        return []
    try:
        mapping = json.loads(raw)
    except ValueError:
        log.warning("CHARACTER_LORAS is not valid JSON - ignored")
        return []
    out, seen = [], set()
    for cid in char_ids:
        for ent in mapping.get(cid, []) if isinstance(mapping, dict) else []:
            name = str(ent.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append({"name": name, "model": float(ent.get("strength", 0.8)), "clip": float(ent.get("clip", 1.0)),
                        "inject_trigger": str(ent.get("trigger", "any")), "is_version": bool(ent.get("is_version", True))})
    return out[:5]


# ------------------------------------------------------------------ composition helpers
def _clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip(" ,.")


def _join_limited(parts, limit):
    """Join comma-separated parts in priority order, dropping whole trailing parts that would exceed `limit`."""
    out, size = [], 0
    for p in parts:
        p = _clean(p)
        if not p:
            continue
        add = len(p) + (2 if out else 0)
        if size + add > limit:
            # try to keep a truncated version of the first overflowing part only if nothing else fits yet
            if not out:
                out.append(p[:limit].rsplit(",", 1)[0])
            break
        out.append(p)
        size += add
    return ", ".join(out)


def _count_tags(chars, no_humans):
    if no_humans or not chars:
        return "no humans, scenery" if no_humans else ""
    m = sum(1 for c in chars if c.get("gender") != "female")
    f = len(chars) - m
    tags = []
    if m:
        tags.append(f"{m}boy" if m == 1 else f"{m}boys")
    if f:
        tags.append(f"{f}girl" if f == 1 else f"{f}girls")
    if len(chars) == 1:
        tags.append("solo")
    elif len(chars) > 1:
        tags.append("multiple characters")
    return ", ".join(tags)


def build_prompt(image, script, family="xl_anime", *, extra_style=""):
    """-> (positive, negative) for one `script["images"][i]` entry.

    Tag-style families put the subject and scene first and the generic style tags last, so truncation at the
    provider's length limit cuts style, never the subject.
    """
    script = script or {}
    by_id = {c["id"]: c for c in script.get("characters", [])}
    chars = [by_id[c] for c in image.get("characters", []) if c in by_id]
    no_humans = bool(image.get("no_humans"))
    scene = _clean(image.get("prompt"))
    palette = _clean(image.get("palette"))
    framing = image.get("framing") or "wide"
    style = _clean(script.get("style"))
    fam = family if family in FAMILIES else "generic"

    if fam == "flux":                                   # natural language, no negative prompt
        looks = " ".join(f"A {c.get('gender', 'male')} character: {_clean(c.get('look'))}"
                         + (f", eyes: {_clean(c.get('eyes'))}" if c.get("eyes") else "") + "." for c in chars)
        text = " ".join(x for x in (
            scene + ".", looks, FRAMING_NATURAL.get(framing, ""), EYE_NATURAL if chars and framing != "wide" else "", (palette + " color palette.") if palette else "",
            style + "." if style else "", extra_style, STYLE_NATURAL, "No text, no letters, no watermark.") if x)
        return text[:MAX_COMBINED], ""

    looks = [_clean(c.get("look")) for c in chars if c.get("look")]
    eyes = [_clean(c.get("eyes")) for c in chars if c.get("eyes")]
    eye_q = EYE_QUALITY.get(fam, "") if chars and framing != "wide" else ""
    parts = [QUALITY.get(fam, ""), _count_tags(chars, no_humans), *looks, *eyes, scene, FRAMING_TAGS.get(framing, ""), eye_q,
             (palette + " color palette") if palette else "", style, extra_style, STYLE_CORE]
    neg = NEGATIVE.get(fam, NEGATIVE["generic"])
    if chars and not no_humans:
        neg = EYE_NEG + ", " + neg
    if no_humans:
        neg += ", 1girl, 1boy, human, person, character"
    neg = _join_limited(neg.split(", "), MAX_NEGATIVE)
    pos = _join_limited(parts, MAX_COMBINED - len(neg) - 5)
    return pos, neg
