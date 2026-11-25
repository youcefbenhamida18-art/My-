"""Title cards: the series / episode title as three transparent layers (face, extrusion, glow) that the render engines animate
(scale-in, floating extrusion for a 3D feel, light sweep, additive glow). Text is rendered with Pillow so Japanese AND Arabic
work (Blender's own text object cannot shape Arabic); fonts come from subs.py (assets/fonts first, then system fonts).
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from . import subs
from .config import log


def _font(path, px, arabic=False):
    raqm = features.check("raqm")
    return ImageFont.truetype(path, px, layout_engine=ImageFont.Layout.RAQM if (raqm and arabic) else ImageFont.Layout.BASIC)


def _mask(text, path, px, W, H, cy, *, arabic=False, spacing=0.0):
    """Float mask (H, W) of `text` centred horizontally at vertical centre cy (fraction of H); also its bbox."""
    if arabic and not features.check("raqm"):
        text = subs._shape_basic(text)
    font = _font(path, px, arabic)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    kw = {"language": "ar"} if arabic and features.check("raqm") else {}
    if spacing and not arabic:                                      # tracked-out capitals/kana look more cinematic
        widths = [d.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * px * (len(text) - 1)
        x = (W - total) / 2
        for ch, w in zip(text, widths):
            d.text((x, H * cy), ch, font=font, fill=255, anchor="lm")
            x += w + spacing * px
    else:
        d.text((W / 2, H * cy), text, font=font, fill=255, anchor="mm", **kw)
    a = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.1)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return a, bbox


def _shift(a, dx, dy):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)


def render_title(ja, ar, kind, out_dir, stem, size=(1920, 1080), label=None):
    """-> {"face","extrude","glow": png paths, "size": [W,H], "kind"} or None when no font can draw the text."""
    W, H = size
    cjk, _ = subs.find_cjk_font()
    arp, _ = subs.find_arabic_font(True)
    ja, ar, label = (ja or "").strip(), (ar or "").strip(), (label or "").strip()
    big = kind == "series"
    parts = []                                                       # (mask, gradient top colour, gradient bottom colour)
    have_ja = bool(ja and cjk)
    try:
        if have_ja:
            px = int(H * (0.19 if big else 0.11)) if len(ja) <= 8 else int(H * (0.14 if big else 0.09))
            m, _ = _mask(ja, cjk, px, W, H, 0.43 if big else 0.76, spacing=0.06)
            parts.append((m, (255, 253, 242), (255, 206, 105)))
        if ar and arp:
            if have_ja:
                px, cy = int(H * (0.085 if big else 0.062)), (0.60 if big else 0.86)
            else:                                                     # no Japanese available: the Arabic title is the main line
                px, cy = int(H * (0.17 if big else 0.10)), (0.45 if big else 0.80)
            m, _ = _mask(ar, arp, px, W, H, cy, arabic=True)
            parts.append((m, (255, 244, 214), (240, 190, 110)))
        if label and arp:
            m, _ = _mask(label, arp, int(H * 0.045), W, H, (0.69 if have_ja else 0.60) if big else 0.70, arabic=True)
            parts.append((m, (230, 230, 236), (190, 190, 205)))
    except OSError as e:
        log.warning("title fonts failed: %s", e)
        return None
    if not parts:
        return None
    sil = np.clip(sum(p[0] for p in parts), 0, 1)
    ys = np.nonzero(sil.max(1) > 0.1)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    face = np.zeros((H, W, 3), np.float32)
    for m, c0, c1 in parts:
        yy = np.nonzero(m.max(1) > 0.1)[0]
        if not len(yy):
            continue
        a, b = yy.min(), yy.max()
        t = np.clip((np.arange(H, dtype=np.float32) - a) / max(1, b - a), 0, 1)[:, None, None]
        col = np.array(c0, np.float32) * (1 - t) + np.array(c1, np.float32) * t
        face += col * m[..., None]
    face = face / np.maximum(sil[..., None], 1e-3)
    fat = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    edge = np.clip(sil - _shift(sil, 0, 4), 0, 1)                       # top edges catch light
    face = np.clip(face + 90 * edge[..., None], 0, 255)
    face_rgba = np.dstack([np.where(sil[..., None] > 0.02, face, 0), sil * 255])
    outline = np.clip(fat - sil, 0, 1)
    face_rgba = np.dstack([face_rgba[..., :3] * (1 - outline[..., None]) + np.array([40, 22, 14], np.float32) * outline[..., None],
                           np.maximum(sil, outline) * 255])
    layer = np.zeros((H, W, 4), np.float32)
    n = 18
    for i in range(n, 0, -1):                                           # extrusion, back to front
        s = _shift(fat, i * 0.9, i * 1.6)
        c = np.array([170, 112, 42], np.float32) * (1 - i / n) + np.array([52, 26, 10], np.float32) * (i / n)
        a = s[..., None]
        layer[..., :3] = layer[..., :3] * (1 - a) + c * a
        layer[..., 3] = np.maximum(layer[..., 3], s * 255)
    glow = cv2.GaussianBlur(fat, (0, 0), H * 0.03)
    glow_rgba = np.dstack([np.full((H, W, 3), (255, 190, 90), np.float32), np.clip(glow * 255 * 1.2, 0, 255)])
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, arr in (("face", face_rgba), ("extrude", layer), ("glow", glow_rgba)):
        p = out / f"{stem}_{name}.png"
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA").save(p, compress_level=3)
        paths[name] = str(p)
    paths.update({"size": [W, H], "kind": kind, "bbox": [0.0, y0 / H, 1.0, y1 / H]})
    return paths
