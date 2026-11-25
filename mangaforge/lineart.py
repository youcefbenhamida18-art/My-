"""Manga ink: the author's own line work as the PRIMARY reference.

    prepare_panel(page, box, size)  crop a manga panel onto a 16:9 canvas -> (init picture for img2img, ink mask)
    ink_mask(gray)                  dark line work with screentone / grey washes removed (+ solid blacks kept as fills)
    alignment(ink, art)             how well the AI colour art follows those lines (0..1, chance level removed)
    overlay_ink(art, ink, k)        re-ink the colour art with the author's lines
    trace(ink)                      Zhang-Suen thinning -> centre-line polylines with widths (what Blender draws as 3D ink curves)

Pure numpy / OpenCV / Pillow, no heavy models: it runs the same in CI and on a laptop. Everything degrades visibly: the pipeline
only uses the ink when `alignment` says the colour art really follows it (a mis-registered second set of lines is worse than none).
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
from PIL import Image

PAPER = 252


# ------------------------------------------------------------------ cropping
def _box_px(box, W, H, pad=0.05):
    """box = [ymin, xmin, ymax, xmax] on a 0..1000 scale (the vision-model convention used everywhere in the project)."""
    y0, x0, y1, x1 = [float(v) for v in box]
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0
    bw, bh = (x1 - x0) / 1000.0 * W, (y1 - y0) / 1000.0 * H
    X0, Y0 = x0 / 1000.0 * W - bw * pad, y0 / 1000.0 * H - bh * pad
    X1, Y1 = x1 / 1000.0 * W + bw * pad, y1 / 1000.0 * H + bh * pad
    return (int(max(0, round(X0))), int(max(0, round(Y0))), int(min(W, round(X1))), int(min(H, round(Y1))))


def to_gray(im):
    a = np.asarray(im.convert("L") if isinstance(im, Image.Image) else im)
    return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def levels(gray, lo_p=1.0, hi_p=99.0):
    """Auto-levels: paper -> white, darkest ink -> black (scans are usually grey-ish)."""
    lo, hi = np.percentile(gray, [lo_p, hi_p])
    hi = max(hi, lo + 8)
    return np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ ink extraction
def ink_mask(gray, *, thr=None, min_area=None, max_stroke=None):
    """-> (stroke mask bool, fill mask bool). Lines are what is really dark after a light blur: screentone dots are small, grey after the
    page was downscaled, and form no long connected strokes, so a dark threshold + small-blob removal drops them while thin solid
    lines survive. Big solid blacks (hair, shadow shapes) are returned separately as fills - they are filled, not stroked."""
    g = levels(gray)
    H, W = g.shape
    b = cv2.GaussianBlur(g, (0, 0), 0.7)
    if thr is None:
        t_otsu, _ = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(np.clip(t_otsu * 0.6, 55, 105))
    ink = (b < thr).astype(np.uint8)
    min_area = int(min_area if min_area is not None else max(10, 0.00003 * H * W))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    ink = keep[lab]
    max_stroke = float(max_stroke if max_stroke is not None else max(6.0, 0.012 * W))       # widest thing still called a stroke (px)
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    big = dt > (max_stroke / 2.0)
    fill = cv2.dilate(big.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(max_stroke) | 1, int(max_stroke) | 1))) > 0
    fill &= ink
    stroke = ink & ~fill
    return stroke, fill


def strokes_only(ink, max_stroke=None):
    """Drop solid fills (blobs thicker than a stroke) from any ink mask: thinning a filled shape is slow and meaningless."""
    ink = ink.astype(bool)
    W = ink.shape[1]
    ms = float(max_stroke if max_stroke is not None else max(6.0, 0.012 * W))
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    fill = cv2.dilate((dt > ms / 2.0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ms) | 1, int(ms) | 1))) > 0
    return ink & ~fill


def prepare_panel(page_path, box, size=(1344, 768), pad=0.04):
    """Crop the panel and put it (contain-fit, paper-white margins) on a canvas of `size`.
    -> (init PIL RGB for img2img: tone softened, contrast-stretched; ink mask bool at canvas size; placement (x0, y0, x1, y1))"""
    page = Image.open(page_path).convert("L")
    W, H = page.size
    x0, y0, x1, y1 = _box_px(box, W, H, pad) if box else (0, 0, W, H)
    if x1 - x0 < 32 or y1 - y0 < 32:
        x0, y0, x1, y1 = 0, 0, W, H
    crop = np.asarray(page.crop((x0, y0, x1, y1)))
    cw, ch = size
    s = min(cw / crop.shape[1], ch / crop.shape[0])
    nw, nh = max(8, int(round(crop.shape[1] * s))), max(8, int(round(crop.shape[0] * s)))
    inter = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    fit = cv2.resize(crop, (nw, nh), interpolation=inter)
    canvas = np.full((ch, cw), PAPER, np.uint8)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = levels(fit)
    stroke, fill = ink_mask(canvas)
    placed = np.zeros((ch, cw), bool)
    placed[oy:oy + nh, ox:ox + nw] = True
    ink = (stroke | fill) & placed
    soft = cv2.GaussianBlur(canvas, (0, 0), 1.1)                          # screentone -> smooth grey: the model should shade, not paint dots
    init = np.where(ink, np.minimum(canvas, soft), soft)
    return Image.fromarray(cv2.cvtColor(init, cv2.COLOR_GRAY2RGB)), ink, (ox, oy, ox + nw, oy + nh)


# ------------------------------------------------------------------ registration + re-inking
def alignment(ink, art_rgb, tol=3.0):
    """0..1: how many ink centre-line pixels sit within `tol` px of an edge of the colour art, chance level removed.
    (colour art that merely resembles the panel scores ~0; art that follows the lines scores > 0.4)"""
    art = np.asarray(art_rgb.convert("RGB")) if isinstance(art_rgb, Image.Image) else art_rgb
    if art.shape[:2] != ink.shape:
        art = cv2.resize(art, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(art, cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
    edges = cv2.Canny(g, 35, 95) > 0
    dist = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    skel = thin(strokes_only(ink))
    if skel.sum() < 40:
        return 0.0
    hit = float((dist[skel] <= tol).mean())
    chance = float((dist <= tol).mean())
    return float(np.clip((hit - chance) / max(1e-6, 1.0 - chance), 0.0, 1.0))


def overlay_ink(art_rgb, ink, strength=0.85, tint=0.14):
    """Re-ink: darken the colour art along the author's lines (line colour = the art's own colour x `tint`, so the ink stays warm/cold
    with the scene instead of dead black)."""
    art = np.asarray(art_rgb.convert("RGB"), np.float32)
    m = ink.astype(np.float32)
    if m.shape != art.shape[:2]:
        m = cv2.resize(m, (art.shape[1], art.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None] * strength
    out = art * (1 - m) + (art * tint) * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ thinning + tracing
_N8 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))             # N NE E SE S SW W NW


def thin(mask, max_iter=40):
    """Zhang-Suen thinning, vectorised. mask bool -> skeleton bool (1 px wide centre lines)."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in _N8]               # p2..p9
            p2, p3, p4, p5, p6, p7, p8, p9 = p
            B = sum(p)
            A = sum(((p[i] == 0) & (p[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _walk(skel):
    """skeleton -> list of pixel paths [(y, x), ...] (ends / junctions split the paths)."""
    H, W = skel.shape
    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return []
    S = np.pad(skel, 1).astype(np.uint8)
    deg = sum(np.roll(np.roll(S, -dy, 0), -dx, 1) for dy, dx in _N8)[1:-1, 1:-1] * skel
    node = skel & (deg != 2)
    seen = set()
    paths = []

    def nbrs(y, x):
        for dy, dx in _N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and skel[yy, xx]:
                yield yy, xx

    def walk(start, first):
        path, prev, cur = [start], start, first
        while True:
            path.append(cur)
            if node[cur] or cur == start:
                break
            nxt = [n for n in nbrs(*cur) if n != prev and (n not in path[-3:])]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        return path

    for y, x in zip(ys, xs):
        if not node[y, x]:
            continue
        for n in nbrs(y, x):
            key = ((y, x), n)
            if key in seen:
                continue
            p = walk((y, x), n)
            seen.add(key)
            seen.add((p[-1], p[-2]) if len(p) > 1 else key)
            for a, b in zip(p[:-1], p[1:]):
                seen.add((a, b))
                seen.add((b, a))
            paths.append(p)
    used = np.zeros_like(skel)
    for p in paths:
        for y, x in p:
            used[y, x] = True
    left = skel & ~used                                                        # closed loops without any junction
    ys, xs = np.nonzero(left)
    for y, x in zip(ys, xs):
        if not left[y, x]:
            continue
        p, cur, prev = [(y, x)], (y, x), None
        while True:
            left[cur] = False
            nx = [n for n in nbrs(*cur) if left[n]]
            if not nx:
                break
            cur = nx[0]
            p.append(cur)
        if len(p) >= 4:
            paths.append(p)
    return paths


def trace(ink, *, min_len=0.012, max_strokes=1800, eps=0.9):
    """ink mask -> {"size": [W, H], "strokes": [{"p": [[x, y]...] (0..1 of the width / height), "w": [stroke width as a fraction of W ...]}]}.
    Centre lines come from thinning, widths from the distance transform; strokes are simplified (Douglas-Peucker) and the longest
    `max_strokes` are kept (short bits are noise)."""
    H, W = ink.shape
    ink = strokes_only(ink)
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    skel = thin(ink)
    paths = _walk(skel)
    strokes = []
    for p in paths:
        a = np.asarray(p, np.float32)[:, ::-1]                                   # (x, y)
        length = float(np.sum(np.hypot(*np.diff(a, axis=0).T))) if len(a) > 1 else 0.0
        if length < min_len * W:
            continue
        w = 2.0 * dt[np.asarray(p)[:, 0], np.asarray(p)[:, 1]]
        if len(a) >= 3:
            k = min(len(w), 7) | 1
            w = np.convolve(np.pad(w, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        simp = cv2.approxPolyDP(a.reshape(-1, 1, 2), eps, False).reshape(-1, 2)
        if len(simp) < 2:
            continue
        idx = [int(np.argmin(np.abs(a[:, 0] - q[0]) + np.abs(a[:, 1] - q[1]))) for q in simp]
        strokes.append((length, simp, w[idx]))
    strokes.sort(key=lambda s: -s[0])
    out = []
    for length, pts, w in strokes[:max_strokes]:
        out.append({"p": [[round(float(x) / W, 5), round(float(y) / H, 5)] for x, y in pts],
                    "w": [round(float(v) / W, 6) for v in w]})
    return {"version": 1, "size": [int(W), int(H)], "strokes": out}


def save_ink(data, path):
    pathlib.Path(path).write_text(json.dumps(data, separators=(",", ":")), "utf-8")
    return str(path)


def load_ink(path):
    return json.loads(pathlib.Path(path).read_text("utf-8"))


def ink_stats(data):
    s = data["strokes"]
    return {"strokes": len(s), "points": int(sum(len(x["p"]) for x in s))}
