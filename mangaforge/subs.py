"""Arabic subtitles done right (the v1 "visual distortion" fix).

ROOT CAUSE (reproduced in the dev session): v1 pre-shaped the text with arabic_reshaper + python-bidi and then drew it with
Pillow. Pillow wheels ship with Raqm, which shapes and re-orders text by itself, so the text was processed TWICE ->
reversed words and wrong letter joins. Rules here make that impossible:
  * ASS output carries RAW LOGICAL Arabic. libass (HarfBuzz + FriBidi) shapes and re-orders it exactly once.
  * Pillow output: Raqm + raw text (language='ar') when available; otherwise reshaper + bidi with the BASIC engine.
"""
from __future__ import annotations

import functools
import pathlib
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont, features

from .config import ASSETS, log


class SubtitleError(RuntimeError):
    pass


AR_PREFS = ("cairo", "tajawal", "almarai", "notosansarabic", "notonaskharabic", "amiri", "vazirmatn", "lateef",
            "dejavusans-bold", "dejavusans", "arialbd", "arial", "tahoma", "segoeui", "freeserif")
CJK_PREFS = ("notosansjp", "notosanscjk", "notoserifcjk", "meiryo", "yugoth", "msgothic", "ipagothic", "takao",
             "hiragino", "droidsansfallback")
_EXTS = {".ttf", ".otf"}


def _font_dirs():
    import os
    dirs = [ASSETS / "fonts", pathlib.Path("/usr/share/fonts"), pathlib.Path("/usr/local/share/fonts"),
            pathlib.Path.home() / ".fonts", pathlib.Path.home() / ".local/share/fonts",
            pathlib.Path("/Library/Fonts"), pathlib.Path("/System/Library/Fonts")]
    if os.name == "nt":
        dirs.append(pathlib.Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    return [d for d in dirs if d.exists()]


@functools.lru_cache(1)
def _scan():
    files = []
    for d in _font_dirs():
        try:
            files += [p for p in d.rglob("*") if p.suffix.lower() in _EXTS]
        except OSError:
            pass
    return files


def _find(prefs, bold=True):
    files = _scan()
    for pref in prefs:
        cands = [f for f in files if pref in re.sub(r"[\s_-]", "", f.name.lower()).replace("-", "")
                 or pref in f.name.lower()]
        if cands:
            cands.sort(key=lambda f: (("bold" in f.name.lower()) != bold, "italic" in f.name.lower(),
                                      "oblique" in f.name.lower(), len(f.name)))
            return cands[0]
    return None


def _family(path):
    try:
        return ImageFont.truetype(str(path), 24).getname()[0]
    except Exception:  # noqa: BLE001
        return pathlib.Path(path).stem


@functools.lru_cache(8)
def find_arabic_font(bold=True):
    """-> (path, family) of the best Arabic-capable font, or (None, None). Drop a TTF into assets/fonts to override."""
    p = _find(AR_PREFS, bold)
    if not p:
        log.warning("no Arabic-capable font found: put Cairo/Tajawal/Noto Sans Arabic .ttf into assets/fonts")
        return None, None
    return str(p), _family(p)


def find_cjk_font():
    p = _find(CJK_PREFS, True)
    return (str(p), _family(p)) if p else (None, None)


@functools.lru_cache(1)
def _ffmpeg_has_libass():
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20).stdout
        return bool(re.search(r"\sass\s", out))
    except Exception:  # noqa: BLE001
        return False


def self_check():
    raqm = features.check("raqm")
    try:
        import arabic_reshaper  # noqa: F401
        import bidi  # noqa: F401
        reshaper = True
    except Exception:  # noqa: BLE001
        reshaper = False
    path, fam = find_arabic_font()
    return {"raqm": raqm, "fribidi": features.check("fribidi"), "harfbuzz": features.check("harfbuzz"),
            "reshaper": reshaper, "font": path, "family": fam, "ffmpeg_libass": _ffmpeg_has_libass(),
            "png_path": "raqm" if raqm else ("basic+reshaper" if reshaper else "UNAVAILABLE"),
            "ass_path": "libass" if _ffmpeg_has_libass() else "UNAVAILABLE (use png overlay)"}


# ------------------------------------------------------------------ Pillow renderer (thumbnails / fallback overlay)
def _shape_basic(s):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except Exception as e:  # noqa: BLE001
        raise SubtitleError("Pillow has no Raqm here: pip install arabic-reshaper python-bidi") from e
    return get_display(arabic_reshaper.reshape(s))


def _wrap(text, font, max_w, shape):
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and font.getlength(shape(cand)) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines or [""]


def render_line_png(text, size=(1920, 1080), *, bold=True, color=(255, 255, 255, 255), stroke=(8, 8, 14, 255),
                    px=None, bottom=None, font_path=None):
    """RGBA canvas of `size` with the (Arabic) line wrapped and centred near the bottom. Text is RAW logical order."""
    raqm = features.check("raqm")
    path = font_path or find_arabic_font(bold)[0]
    if not path:
        raise SubtitleError("no font available for subtitles")
    px = px or round(size[1] * 0.058)
    font = ImageFont.truetype(path, px, layout_engine=ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC)
    shape = (lambda s: s) if raqm else _shape_basic
    kw = {"language": "ar"} if raqm else {}
    lines = _wrap(text, font, size[0] * 0.84, shape)
    lh = int(px * 1.38)
    sw = max(2, px // 11)
    y = size[1] - (bottom if bottom is not None else int(size[1] * 0.075)) - lh * len(lines)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for ln in lines:
        s = shape(ln)
        x = (size[0] - font.getlength(s)) / 2
        d.text((x + 2, y + 3), s, font=font, fill=(0, 0, 0, 120), stroke_width=sw, stroke_fill=(0, 0, 0, 120), **kw)
        d.text((x, y), s, font=font, fill=color, stroke_width=sw, stroke_fill=stroke, **kw)
        y += lh
    return img


# ------------------------------------------------------------------ ASS (default path: libass shapes Arabic correctly)
def _ts(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _clean(s):
    s = re.sub(r"[{}]", "", str(s)).replace("\\", "")
    return re.sub(r"\s*\n\s*", r"\\N", s.strip())


def timeline(shots):
    """-> [(start, end, shot)] absolute times of every shot (cumulative `dur`)."""
    t, out = 0.0, []
    for s in shots:
        out.append((t, t + s["dur"], s))
        t += s["dur"]
    return out


def write_ass(shots, out_path, *, size=(1920, 1080), vertical=False, show_ja=False):
    w, h = (size[1], size[0]) if (vertical and size[0] > size[1]) else size
    _, fam = find_arabic_font(True)
    fam = fam or "DejaVu Sans"
    _, jfam = find_cjk_font()
    fs = round(h * (0.040 if vertical else 0.057))
    mv = round(h * (0.16 if vertical else 0.075))
    mx = round(w * (0.07 if vertical else 0.09))
    ol = max(2.0, fs / 11)
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {w}\nPlayResY: {h}\nYCbCr Matrix: TV.709\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
        "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: AR,{fam},{fs},&H00FFFFFF,&H000000FF,&H00181010,&H96000000,-1,0,0,0,100,100,0,0,1,{ol:.1f},1.5,2,{mx},{mx},{mv},1\n"
        f"Style: JA,{jfam or fam},{round(fs * 0.52)},&H00D8D8D8,&H000000FF,&H00181010,&H96000000,0,0,0,0,100,100,0,0,1,{ol * .6:.1f},1,8,{mx},{mx},{round(h * .05)},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev = []
    tl = timeline(shots)
    for i, (t0, t1, s) in enumerate(tl):
        tx = s.get("text")
        if not tx or not (tx.get("ar") or tx.get("ja")):
            continue
        a = s.get("audio") or {}
        st = t0 + (a.get("voice_start") or 0.15) if a.get("voice") else t0 + 0.2
        en = st + (a.get("voice_dur") or (t1 - t0 - 0.4)) + 0.35
        en = min(en, t1 - 0.05)
        if en - st < 0.5:
            st, en = t0 + 0.1, max(t0 + 0.6, t1 - 0.05)
        fx = r"{\fscx94\fscy94\t(0,140,\fscx100\fscy100)\fad(110,140)}"
        if tx.get("ar"):
            ev.append(f"Dialogue: 0,{_ts(st)},{_ts(en)},AR,,0,0,0,,{fx}{_clean(tx['ar'])}")
        if show_ja and tx.get("ja"):
            ev.append(f"Dialogue: 0,{_ts(st)},{_ts(en)},JA,,0,0,0,,{{\\fad(110,140)}}{_clean(tx['ja'])}")
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(head + "\n".join(ev) + "\n", "utf-8-sig")
    return p


def _esc(p):
    return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def burn_filter(ass_path):
    """ffmpeg filter string that burns the ASS file (fonts from assets/fonts and system dirs)."""
    fd = ASSETS / "fonts"
    f = f"ass='{_esc(ass_path)}'"
    if fd.exists() and any(fd.glob("*.[to]tf")):
        f += f":fontsdir='{_esc(fd)}'"
    return f
