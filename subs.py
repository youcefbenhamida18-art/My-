"""Arabic subtitles drawn with Pillow (proper letter shaping + RTL), blended per frame by the engine."""
import functools
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _RESHAPER = arabic_reshaper.ArabicReshaper()
except Exception:  # noqa: BLE001  (selftest without the libs still runs)
    _RESHAPER = None

FONT_GLOBS = [
    "/usr/share/fonts/**/NotoNaskhArabic-Bold.ttf", "/usr/share/fonts/**/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/**/NotoNaskhArabic*.ttf", "/usr/share/fonts/**/NotoSansArabic*.ttf",
    "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
]


def shape(text):
    return get_display(_RESHAPER.reshape(text)) if _RESHAPER else text


@functools.lru_cache(1)
def font_path():
    for pat in FONT_GLOBS:
        hit = sorted(glob.glob(pat, recursive=True))
        if hit:
            return hit[0]
    return None


@functools.lru_cache(8)
def get_font(size):
    p = font_path()
    return ImageFont.truetype(p, size) if p else ImageFont.load_default()


def _width(font, s):
    try:
        return font.getlength(s)
    except Exception:  # noqa: BLE001
        return len(s) * 10


def _wrap(text, font, maxw):
    lines, cur = [], ""
    for w in text.split():
        cand = (cur + " " + w).strip()
        if _width(font, shape(cand)) <= maxw or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def render(text, ow, oh, color=(255, 255, 255), size=None):
    """-> (bgr float32 HxWx3, alpha float32 HxW, y0). Bottom-anchored; higher in vertical video (Shorts UI)."""
    vertical = oh > ow
    size = size or int(0.052 * ow if vertical else 0.048 * oh)
    font = get_font(size)
    lines = _wrap(text, font, int(ow * (0.86 if vertical else 0.78)))
    lh, pad, sw = int(size * 1.45), 12, max(3, size // 12)
    h = lh * len(lines) + 2 * pad
    st, fl = Image.new("L", (ow, h), 0), Image.new("L", (ow, h), 0)
    ds, df = ImageDraw.Draw(st), ImageDraw.Draw(fl)
    for j, line in enumerate(lines):
        s = shape(line)
        x, y = (ow - _width(font, s)) / 2, pad + j * lh
        ds.text((x, y), s, font=font, fill=255, stroke_width=sw, stroke_fill=255)
        df.text((x, y), s, font=font, fill=255)
    st, fl = np.asarray(st, np.float32) / 255, np.asarray(fl, np.float32) / 255
    alpha = np.clip(np.maximum(st, fl), 0, 1)
    col = fl[..., None] * np.array(color[::-1], np.float32)[None, None, :]   # white fill, black outline
    margin = int(oh * (0.20 if vertical else 0.06))
    return col, alpha, max(0, oh - h - margin)
