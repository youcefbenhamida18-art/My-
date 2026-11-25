"""The eye system - the signature of the Demon-Slayer look.

    detect(plate)                      Gemini vision: faces + eye boxes (no local face model works on anime art)
    analyse(rgb, box)                  snap the box to the real eye, find skin / lash / iris colours, iris ellipse, existing highlight
    enhance(rgb, an)                   iris saturation + anime gradient (dark top, light bottom) + crisp micro-contrast
    make_layers(...)                   per eye RGBA drawings: closed lids x3 (blink), catch-light (additive), glow (additive)
    refine(...)                        optional img2img pass over the face (provider with img2img) - only kept if it measurably helps
    process(plate, image, script, ...) the whole chain for one key image -> `<id>_eyes.json` (+ gaze mask) used by both render engines
    closeup_framing(eyes, ...)         zoom/centre for an extreme eye close-up

Animation (blink schedule, saccades, pupil, glint, glow, tears) lives in eye_anim.py so Blender can import it too.
"""
from __future__ import annotations

import io
import json
import pathlib

import cv2
import numpy as np
from PIL import Image

from . import llm
from .config import cfg, log

SS = 3                                                    # supersampling for drawn layers


# ------------------------------------------------------------------ basics
def _load(src):
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGB"))
    return np.asarray(Image.open(src).convert("RGB"))


def _ls(rgb_f):
    mx, mn = rgb_f.max(-1), rgb_f.min(-1)
    return rgb_f @ np.array([0.299, 0.587, 0.114], np.float32), (mx - mn) / (mx + 1e-5)


def _nb(b):
    """Gemini [ymin,xmin,ymax,xmax] 0..1000 -> (x0,y0,x1,y1) in 0..1 or None."""
    try:
        y0, x0, y1, x1 = [min(1000.0, max(0.0, float(v))) / 1000.0 for v in b]
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 - x0 >= 0.004 and y1 - y0 >= 0.004 else None


def _plausible(eb, fb):
    fw, fh = fb[2] - fb[0], fb[3] - fb[1]
    ew, eh = eb[2] - eb[0], eb[3] - eb[1]
    cx, cy = (eb[0] + eb[2]) / 2, (eb[1] + eb[3]) / 2
    inside = fb[0] - 0.1 * fw <= cx <= fb[2] + 0.1 * fw and fb[1] - 0.1 * fh <= cy <= fb[3] + 0.1 * fh
    return inside and 0.06 * fw <= ew <= 0.62 * fw and 0.8 <= ew / max(eh, 1e-6) <= 8.0 and ew >= 0.006


# ------------------------------------------------------------------ detection (vision LLM)
def detect(src, hint=None):
    rgb = _load(src)
    H, W = rgb.shape[:2]
    k = 1280.0 / max(W, H)
    small = cv2.resize(rgb, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else rgb
    b = io.BytesIO()
    Image.fromarray(small).save(b, "JPEG", quality=88)
    prompt = f"""This is an anime key image. Locate every visible character FACE and each visible EYE.
Return JSON {{"faces": [{{"face_box_2d": [ymin, xmin, ymax, xmax], "facing": "front|three_quarter_left|three_quarter_right|profile_left|profile_right|up|down|back",
 "character": "one of {json.dumps(list(hint or []))} or empty", "emotion": "neutral|angry|sad|happy|fear|shock|determined",
 "eyes": [{{"box_2d": [ymin, xmin, ymax, xmax], "open": true}}]}}]}}
All coordinates are integers 0-1000 relative to the image. An eye box is TIGHT around ONE eye: from the inner to the outer corner, from the
upper lash line to the lower lash line (wider than tall). Skip faces narrower than 3% of the image width, faces turned away, hidden or in deep
shadow, and eyes that are closed or covered. If there is no usable face return {{"faces": []}}. JSON only."""
    try:
        o = llm.ask_json(prompt, images=[b.getvalue()], temperature=0.1, max_tokens=4000)
    except llm.LLMError as e:
        return {"faces": [], "method": "unavailable", "error": str(e)[:200]}
    faces = []
    for f in (o.get("faces") if isinstance(o, dict) else None) or []:
        if not isinstance(f, dict):
            continue
        fb = _nb(f.get("face_box_2d"))
        if not fb:
            continue
        eyes = []
        for e in f.get("eyes") or []:
            eb = _nb(e.get("box_2d")) if isinstance(e, dict) else None
            if eb and e.get("open", True) and _plausible(eb, fb):
                eyes.append({"box": eb})
        faces.append({"box": fb, "facing": str(f.get("facing") or "front"), "char": str(f.get("character") or ""),
                      "emotion": str(f.get("emotion") or "neutral"), "eyes": eyes})
    return {"faces": faces, "method": "gemini"}


# ------------------------------------------------------------------ pixel analysis
def analyse(rgb, box):
    """box = normalised (x0,y0,x1,y1). -> dict (pixel coordinates) or None when the pixels do not look like an open eye."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
    bw, bh = x1 - x0, y1 - y0
    if bw < 14 or bh < 5:
        return None
    # skin: the ring around the eye
    sx0, sy0, sx1, sy1 = max(0, x0 - int(0.9 * bw)), max(0, y0 - int(1.1 * bh)), min(W, x1 + int(0.9 * bw)), min(H, y1 + int(1.1 * bh))
    ring = rgb[sy0:sy1, sx0:sx1].astype(np.float32) / 255
    L, S = _ls(ring)
    inner = np.zeros(L.shape, bool)
    inner[max(0, y0 - sy0 - int(0.3 * bh)): y1 - sy0 + int(0.3 * bh), max(0, x0 - sx0 - int(0.2 * bw)): x1 - sx0 + int(0.2 * bw)] = True
    ok = (~inner) & (L > 0.5) & (L < 0.98) & (S < 0.55)
    if ok.sum() < 40:
        return None
    lv = L[ok]
    lo, hi = np.percentile(lv, [25, 75])
    sel = ok & (L >= lo) & (L <= hi)
    skin = np.median(ring[sel], 0)
    # structure of the eye inside a slightly larger window
    px, py = int(bw * 0.2), int(bh * 0.3)
    wx0, wy0, wx1, wy1 = max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)
    roi = rgb[wy0:wy1, wx0:wx1].astype(np.float32) / 255
    L, S = _ls(roi)
    dist = np.linalg.norm(roi - skin[None, None, :], axis=2)
    m = ((L < 0.33) | ((L > 0.8) & (S < 0.25) & (dist > 0.04)) | ((S > 0.30) & (L > 0.08) & (L < 0.85) & (dist > 0.15))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    box_roi = (x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0)
    best, bs = 0, 0
    for i in range(1, n):
        ix0, iy0 = max(box_roi[0], st[i, 0]), max(box_roi[1], st[i, 1])
        ix1, iy1 = min(box_roi[2], st[i, 0] + st[i, 2]), min(box_roi[3], st[i, 1] + st[i, 3])
        ov = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if ov > bs:
            best, bs = i, ov
    snap = False
    if best:
        sx, sy, sw, sh = st[best, 0], st[best, 1], st[best, 2], st[best, 3]
        if 0.5 <= (sw * sh) / max(bw * bh, 1) <= 2.0 and 1.1 <= sw / max(sh, 1) <= 7.0:
            x0, y0, x1, y1 = wx0 + sx, wy0 + sy, wx0 + sx + sw, wy0 + sy + sh
            bw, bh = x1 - x0, y1 - y0
            snap = True
    eye = rgb[y0:y1, x0:x1].astype(np.float32) / 255
    L, S = _ls(eye)
    dist = np.linalg.norm(eye - skin[None, None, :], axis=2)
    area = float(bw * bh)
    sclera = float(((L > 0.78) & (S < 0.25)).sum()) / area
    guess = False
    k = max(3, int(0.17 * bh) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    iris_m = ((S > 0.24) & (L > 0.10) & (L < 0.82) & (dist > 0.12)).astype(np.uint8)
    iris_m = cv2.morphologyEx(iris_m, cv2.MORPH_OPEN, kern)                 # cuts thin lids/lashes that touch the iris
    if iris_m.sum() < 0.04 * area:                                       # dark / low-saturation eyes: the dark blob minus the thin lash line
        iris_m = cv2.morphologyEx((L < 0.33).astype(np.uint8), cv2.MORPH_OPEN, kern)
    n, lab, st, _ = cv2.connectedComponentsWithStats(iris_m, 8)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        ix, iy, iw, ih = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        sel = lab == i
    else:
        ix, iy, iw, ih, sel = int(bw * 0.35), int(bh * 0.1), int(bw * 0.3), int(bh * 0.8), None
        guess = True
    if sel is not None and sel.sum() < 0.03 * area:
        ix, iy, iw, ih, sel, guess = int(bw * 0.35), int(bh * 0.1), int(bw * 0.3), int(bh * 0.8), None, True
    iris = [x0 + ix + iw / 2.0, y0 + iy + ih / 2.0, max(3.0, iw / 2.0), max(3.0, ih / 2.0)]
    color = np.median(eye[sel], 0) if sel is not None else np.array([0.35, 0.25, 0.2])
    lv = L.ravel()
    lash = eye.reshape(-1, 3)[np.argsort(lv)[: max(3, int(0.06 * lv.size))]].mean(0)
    hl, hl_xy = False, None
    if not guess:
        sub = L[iy: iy + ih, ix: ix + iw]
        subs = S[iy: iy + ih, ix: ix + iw]
        bright = (sub > 0.9) & (subs < 0.25)
        hl = bool(bright.sum() > 0.008 * max(sub.size, 1))
        if hl:
            ys_, xs_ = np.nonzero(bright)
            hl_xy = [float((xs_.mean() - iw / 2) / max(iw / 2, 1)), float((ys_.mean() - ih / 2) / max(ih / 2, 1))]
    ratio = (iw * ih) / area
    if not (0.04 <= ratio <= 0.95):
        return None
    return {"box_px": [int(x0), int(y0), int(x1), int(y1)], "skin": [float(v) for v in skin * 255], "lash": [float(v) for v in lash * 255],
            "iris": [float(v) for v in iris], "color": [float(v) for v in color * 255], "has_highlight": hl, "hl_xy": hl_xy, "guess": guess, "snapped": snap,
            "sclera": round(sclera, 3)}


# ------------------------------------------------------------------ iris enhancement (applied to the plate itself)
def enhance(rgb, an, strength=1.0):
    if an["guess"]:
        return False
    H, W = rgb.shape[:2]
    cx, cy, rx, ry = an["iris"]
    x0, y0, x1, y1 = an["box_px"]
    pad = int(max(rx, ry) * 0.5) + 2
    X0, Y0, X1, Y1 = max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(W, int(x1 + pad)), min(H, int(y1 + pad))
    roi = np.ascontiguousarray(rgb[Y0:Y1, X0:X1])
    h, w = roi.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ex, ey = (xx + X0 - cx) / rx, (yy + Y0 - cy) / ry
    r = np.sqrt(ex ** 2 + ey ** 2)
    m = cv2.GaussianBlur(np.clip((1.12 - r) / 0.18, 0, 1), (0, 0), 0.8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    g = np.clip((ey + 1) / 2, 0, 1)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.20 * strength * m), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + strength * m * (-0.14 + 0.30 * g)), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    blur = cv2.GaussianBlur(out, (0, 0), max(0.8, rx * 0.08))
    sharp = cv2.addWeighted(out, 1.18, blur, -0.18, 0)
    box = np.zeros((h, w), np.float32)
    box[max(0, y0 - Y0): y1 - Y0, max(0, x0 - X0): x1 - X0] = 1.0
    box = cv2.GaussianBlur(box, (0, 0), max(1.0, 0.05 * (x1 - x0)))[..., None]
    rgb[Y0:Y1, X0:X1] = (roi * (1 - box) + sharp * box).astype(np.uint8)
    return True


# ------------------------------------------------------------------ drawn layers (RGBA PNGs on one shared canvas per eye)
def _down(a):
    return cv2.resize(a, (a.shape[1] // SS, a.shape[0] // SS), interpolation=cv2.INTER_AREA)


def _canvas_box(an, W, H):
    x0, y0, x1, y1 = an["box_px"]
    bw, bh = x1 - x0, y1 - y0
    ox, oy = max(0, int(x0 - 0.4 * bw)), max(0, int(y0 - 0.7 * bh))
    ex, ey = min(W, int(x1 + 0.4 * bw)), min(H, int(y1 + 0.7 * bh))
    return ox, oy, ex - ox, ey - oy


def closed_layer(an, c, side, ox, oy, wc, hc, base=None):
    """Lid drawn down to level c (1.0 = fully closed eye). Straight-alpha RGBA uint8."""
    x0, y0, x1, y1 = [(v - o) * SS for v, o in zip(an["box_px"], (ox, oy, ox, oy))]
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = wc * SS, hc * SS
    ell = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(ell, (int(cx), int(cy)), (max(2, int(bw * 0.58)), max(2, int(bh * 0.68))), 0, 0, 360, 255, -1)
    xs = np.arange(cw, dtype=np.float32)
    u = np.clip((xs - x0) / max(bw, 1), -0.1, 1.1)
    line_y = y0 + 0.02 * bh + (0.10 + 0.62 * c) * bh + 0.16 * bh * c * 4 * u * (1 - u)
    ys = np.arange(ch, dtype=np.float32)[:, None]
    fill = ell if c >= 0.99 else np.where(ys < line_y[None, :], ell, 0).astype(np.uint8)
    if base is not None:                                                   # never paint over dark hair/brows that lie outside the eye
        Lb = (base.astype(np.float32) / 255) @ np.array([0.299, 0.587, 0.114], np.float32)
        dark = (Lb < 0.30).astype(np.uint8)
        bx0, by0, bx1, by1 = [int(v / SS) for v in (x0 - 0.10 * bw, y0 - 0.12 * bh, x1 + 0.10 * bw, y1 + 0.12 * bh)]
        dark[max(0, by0): max(0, by1), max(0, bx0): max(0, bx1)] = 0
        dark = cv2.dilate(dark, np.ones((3, 3), np.uint8))
        fill = np.where(cv2.resize(dark, (cw, ch), interpolation=cv2.INTER_NEAREST) > 0, 0, fill).astype(np.uint8)
    line = np.zeros((ch, cw), np.uint8)
    th = max(2, int(0.085 * bh * (1.15 if c >= 0.99 else 1.0)))
    px = np.arange(int(x0 - 0.05 * bw), int(x1 + 0.05 * bw) + 1, max(1, int(bw / 40)))
    py = np.interp(px, xs, line_y)
    cv2.polylines(line, [np.stack([px, py], 1).astype(np.int32).reshape(-1, 1, 2)], False, 255, th, cv2.LINE_AA)
    sgn = -1.0 if side == "left" else 1.0                                   # outer corner: image-left for the left eye
    ex = x0 if sgn < 0 else x1
    ey = float(np.interp(ex, xs, line_y))
    wing = [(ex, ey), (ex + sgn * 0.08 * bw, ey - 0.04 * bh), (ex + sgn * 0.17 * bw, ey - 0.15 * bh)]     # tapered outer flick
    for (a0, a1), tk in zip(zip(wing[:-1], wing[1:]), (0.75, 0.4)):
        cv2.line(line, (int(a0[0]), int(a0[1])), (int(a1[0]), int(a1[1])), 255, max(1, int(th * tk)), cv2.LINE_AA)
    a = np.maximum(fill, line).astype(np.float32) / 255
    skin, lash = np.array(an["skin"], np.float32), np.array(an["lash"], np.float32)
    rgb = np.where((line > 127)[..., None], lash[None, None], skin[None, None]).astype(np.float32)
    pre = _down(rgb * a[..., None])
    ad = _down(a)
    ad = cv2.GaussianBlur(ad, (0, 0), 0.7)
    ad0 = _down(a)[..., None]
    col = np.where(ad0 > 0.02, pre / np.maximum(ad0, 1e-3), skin[None, None, :])       # no black fringe where alpha is tiny
    out = np.dstack([np.clip(col, 0, 255), np.clip(ad * 255, 0, 255)])
    return np.round(out).astype(np.uint8)


def shine_layer(an, ox, oy, wc, hc):
    """Additive catch-lights (white, alpha = intensity), clipped to the iris. Never stacks on an existing big highlight."""
    rx, ry = an["iris"][2] * SS, an["iris"][3] * SS
    cx, cy = (an["iris"][0] - ox) * SS, (an["iris"][1] - oy) * SS
    A = np.zeros((hc * SS, wc * SS), np.float32)
    if an.get("has_highlight") and an.get("hl_xy"):                          # existing highlight: add the small one on the opposite side
        sx, sy = -float(an["hl_xy"][0]), -float(an["hl_xy"][1])
        sx, sy = (sx if abs(sx) > 0.15 else 0.3), (sy if abs(sy) > 0.15 else 0.38)
        sx, sy = max(-0.5, min(0.5, sx)), max(-0.55, min(0.55, sy))
    else:
        sx, sy = 0.30, 0.38
        cv2.ellipse(A, (int(cx - 0.28 * rx), int(cy - 0.32 * ry)), (max(1, int(0.34 * rx)), max(1, int(0.36 * ry))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(A, (int(cx + sx * rx), int(cy + sy * ry)), (max(1, int(0.14 * rx)), max(1, int(0.15 * ry))), 0, 0, 360, 0.85, -1, cv2.LINE_AA)
    star = np.array([[0, -1], [0.16, -0.16], [1, 0], [0.16, 0.16], [0, 1], [-0.16, 0.16], [-1, 0], [-0.16, -0.16]], np.float32) * 0.24 * rx
    cv2.fillPoly(A, [(star + np.array([cx + 0.16 * rx, cy + 0.04 * ry], np.float32)).astype(np.int32)], 0.9, cv2.LINE_AA)
    clip = np.zeros_like(A)
    cv2.ellipse(clip, (int(cx), int(cy)), (int(rx * 1.06), int(ry * 1.06)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    A = _down(cv2.GaussianBlur(A * clip, (0, 0), 0.8 * SS))
    rgba = np.zeros(A.shape + (4,), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.clip(A * 255, 0, 255)
    return rgba


def glow_layer(an, ox, oy, wc, hc):
    """Additive coloured glow: iris fill + ring + almond halo. RGB is the (saturated) iris colour, alpha is intensity."""
    yy, xx = np.mgrid[0:hc, 0:wc].astype(np.float32)
    cx, cy, rx, ry = an["iris"][0] - ox, an["iris"][1] - oy, an["iris"][2], an["iris"][3]
    x0, y0, x1, y1 = an["box_px"]
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    ex, ey = (x0 + x1) / 2 - ox, (y0 + y1) / 2 - oy
    d2 = ((xx - ex) / (0.75 * (x1 - x0))) ** 2 + ((yy - ey) / (1.1 * (y1 - y0))) ** 2
    a = 0.38 * np.clip((1.05 - r) / 0.35, 0, 1) + 0.70 * np.exp(-(((r - 1.05) / 0.18) ** 2)) + 0.26 * np.exp(-d2 * 2.0)
    hsv = cv2.cvtColor(np.uint8([[an["color"]]]), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    hsv[1], hsv[2] = min(255, hsv[1] * 1.2 + 50), 255
    col = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
    rgba = np.zeros((hc, wc, 4), np.uint8)
    rgba[..., :3] = col
    rgba[..., 3] = np.clip(a * 255, 0, 255)
    return rgba


def make_layers(rgb, ans, out_dir, stem):
    """-> list of per-eye dicts (see process) and the path of the gaze mask."""
    H, W = rgb.shape[:2]
    out_dir = pathlib.Path(out_dir)
    items = []
    mask = np.zeros((H // 2, W // 2), np.float32)
    for k, (an, side, fi) in enumerate(ans):
        ox, oy, wc, hc = _canvas_box(an, W, H)
        files = {"closed": [], "origin": [ox, oy], "size": [wc, hc]}
        for j, c in enumerate((0.45, 0.8, 1.0), 1):
            p = out_dir / f"{stem}_e{k}_closed{j}.png"
            Image.fromarray(closed_layer(an, c, side, ox, oy, wc, hc, rgb[oy:oy + hc, ox:ox + wc])).save(p, compress_level=3)
            files["closed"].append(str(p))
        for name, fn in (("shine", shine_layer), ("glow", glow_layer)):
            p = out_dir / f"{stem}_e{k}_{name}.png"
            Image.fromarray(fn(an, ox, oy, wc, hc)).save(p, compress_level=3)
            files[name] = str(p)
        cx, cy, rx, ry = an["iris"]
        cv2.ellipse(mask, (int(cx / 2), int(cy / 2)), (max(1, int(rx * 0.62)), max(1, int(ry * 0.62))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        items.append({"face": fi, "side": side, "box": [an["box_px"][0] / W, an["box_px"][1] / H, an["box_px"][2] / W, an["box_px"][3] / H],
                      "box_px": an["box_px"], "iris": an["iris"], "color": an["color"], "has_highlight": an["has_highlight"], "layers": files})
    mp = out_dir / f"{stem}_eyemask.png"
    Image.fromarray((cv2.GaussianBlur(mask, (0, 0), 1.5) * 255).astype(np.uint8)).save(mp)
    return items, str(mp)


# ------------------------------------------------------------------ img2img refinement (optional, quality-gated)
def _lapvar(rgb, boxes):
    v = []
    for (x0, y0, x1, y1) in boxes:
        g = cv2.cvtColor(rgb[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_RGB2GRAY)
        if g.size > 16:
            v.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    return float(np.mean(v)) if v else 0.0


def refine(rgb, faces, image, script, chain, seed, progress=None):
    """Re-draw the face area at higher effective resolution with a low-denoise img2img pass. -> number of faces refined."""
    from .art import prompts
    prov = next((p for p in chain if hasattr(p, "img2img")), None)
    if prov is None:
        return 0
    H, W = rgb.shape[:2]
    n = 0
    for f in sorted(faces, key=lambda f: -(f["box"][2] - f["box"][0]))[:2]:
        fx0, fy0, fx1, fy1 = f["box"][0] * W, f["box"][1] * H, f["box"][2] * W, f["box"][3] * H
        side = int(max(fx1 - fx0, fy1 - fy0) * 1.7)
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        X0, Y0 = int(max(0, min(W - side, cx - side / 2))), int(max(0, min(H - side, cy - side / 2)))
        X1, Y1 = min(W, X0 + side), min(H, Y0 + side)
        if X1 - X0 < 160 or Y1 - Y0 < 160:
            continue
        crop = rgb[Y0:Y1, X0:X1]
        pos, neg = prompts.build_prompt(dict(image, framing="closeup"), script, "xl_anime")
        try:
            out = prov.img2img(Image.fromarray(crop).resize((768, 768), Image.LANCZOS),
                               pos + ", extremely detailed eyes, perfect iris, crisp lashes", neg, seed=seed, denoise=0.32, progress=progress)
        except Exception as e:  # noqa: BLE001
            log.warning("eye refine failed (%s) - keeping the original", e)
            return n
        if out is None:
            continue
        new = np.asarray(out.convert("RGB").resize((X1 - X0, Y1 - Y0), Image.LANCZOS))
        boxes = [(int(e["box"][0] * W) - X0, int(e["box"][1] * H) - Y0, int(e["box"][2] * W) - X0, int(e["box"][3] * H) - Y0) for e in f["eyes"]]
        before, after = _lapvar(crop, boxes), _lapvar(new, boxes)
        drift = float(np.abs(new.astype(np.float32) - crop.astype(np.float32)).mean())
        if boxes and (after < 0.9 * before or drift > 24):
            log.info("eye refine rejected (sharpness %.0f -> %.0f, colour drift %.1f)", before, after, drift)
            continue
        m = np.zeros(crop.shape[:2], np.float32)
        cv2.ellipse(m, ((X1 - X0) // 2, (Y1 - Y0) // 2), (int((X1 - X0) * 0.40), int((Y1 - Y0) * 0.44)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), (X1 - X0) * 0.06)[..., None]
        rgb[Y0:Y1, X0:X1] = (crop * (1 - m) + new * m).astype(np.uint8)
        n += 1
    return n


# ------------------------------------------------------------------ the whole chain for one key image
def eyes_path(art_dir, stem):
    return pathlib.Path(art_dir) / f"{stem}_eyes.json"


def process(plate_path, image, script, art_dir, stem, chain=None, refine_ok=False, seed=0):
    """-> info dict (also written to <stem>_eyes.json) or None when the feature is off / image has no humans."""
    if not cfg.eyes or image.get("no_humans"):
        return None
    ids = list(image.get("characters") or [])
    det = detect(plate_path, ids)
    rgb = _load(plate_path).copy()
    H, W = rgb.shape[:2]
    faces = [f for f in det["faces"] if f["eyes"]]
    n_ref = 0
    if faces and refine_ok and cfg.eye_refine and chain:
        n_ref = refine(rgb, faces, image, script, chain, seed)
    ans = []
    for fi, f in enumerate(faces):
        for e in f["eyes"]:
            an = analyse(rgb, e["box"])
            if an is None:
                continue
            cx = (an["box_px"][0] + an["box_px"][2]) / 2
            fcx = (f["box"][0] + f["box"][2]) / 2 * W
            ans.append((an, "left" if cx < fcx else "right", fi))
    enhanced = sum(1 for an, _, _ in ans if enhance(rgb, an))
    items, mask = make_layers(rgb, ans, art_dir, stem) if ans else ([], None)
    if (n_ref or enhanced) and pathlib.Path(plate_path).exists():
        im = Image.fromarray(rgb)
        if str(plate_path).lower().endswith(".png"):
            im.save(plate_path, compress_level=3)
        else:
            im.save(plate_path, quality=94, subsampling=0, optimize=True)
    info = {"version": 1, "size": [W, H], "method": det.get("method"), "refined_faces": n_ref, "enhanced": enhanced, "mask": mask,
            "faces": [{"box": f["box"], "facing": f["facing"], "char": f["char"], "emotion": f["emotion"]} for f in faces], "eyes": items}
    if det.get("error"):
        info["error"] = det["error"]
    eyes_path(art_dir, stem).write_text(json.dumps(info, indent=1), "utf-8")
    log.info("eyes %s: %d faces, %d eyes ok, enhanced=%d refined=%d (%s)", stem, len(faces), len(items), enhanced, n_ref, det.get("method"))
    return info


def load(path):
    p = pathlib.Path(path) if path else None
    return json.loads(p.read_text("utf-8")) if p and p.exists() else None


# ------------------------------------------------------------------ framing helper for extreme eye close-ups
def closeup_framing(info, *, which="both", fill=0.46, zmin=2.0, zmax=3.6, margin=0.9):
    """-> (zoom, cx, cy) centred on the eyes of the largest face, or None. Frame covers `margin/zoom` of the plate width."""
    if not info or not info.get("eyes"):
        return None
    by_face = {}
    for e in info["eyes"]:
        by_face.setdefault(e["face"], []).append(e)
    fi = max(by_face, key=lambda k: sum((e["box"][2] - e["box"][0]) for e in by_face[k]))
    es = by_face[fi]
    if which in ("left", "right"):
        es = [e for e in es if e["side"] == which] or es
    x0, y0 = min(e["box"][0] for e in es), min(e["box"][1] for e in es)
    x1, y1 = max(e["box"][2] for e in es), max(e["box"][3] for e in es)
    zoom = max(zmin, min(zmax, margin * fill / max(x1 - x0, 1e-3)))
    return round(zoom, 3), round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4)
