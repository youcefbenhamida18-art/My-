"""From one key-art image to everything the 2.5D renderer needs: plate, depth, matte-derived masks, backdrop, palette.

Heavy models are optional and lazily imported; every step has a CPU/numpy/OpenCV fallback and says which path it used:
    depth  : Depth-Anything-V2 (transformers + torch)      -> heuristic (vertical prior + subject + local contrast)
    matte  : rembg isnet-anime (onnxruntime)               -> saliency + GrabCut -> centre prior
Output depth maps are 8-bit, 255 = nearest. Masks are 8-bit. All geometry is in image coordinates with y pointing down.
"""
from __future__ import annotations

import functools
import json
import pathlib

import cv2
import numpy as np
from PIL import Image

from ..config import cfg, log

ANALYSIS_W = 960                      # working width for all analysis (cheap, plenty for smooth masks)
MASK_MAX_W = 1920                     # masks/depth are saved at most this wide
BACKDROP_SCALE = 1.5                  # backdrop plane covers 1.5x the plate (see SPEC)


# ------------------------------------------------------------------ io / geometry helpers
def load_rgb(path):
    im = Image.open(path)
    im.load()
    return np.asarray(im.convert("RGB"))


def fit_16x9(rgb):
    """Exact 16:9. Wider -> centre crop; taller (manga panels) -> sharp panel in the middle, blurred copy fills the sides."""
    h, w = rgb.shape[:2]
    target = 16 / 9
    ar = w / h
    if abs(ar - target) < 0.012:
        nw = int(round(h * target))
        x0 = max(0, (w - nw) // 2)
        out = rgb[:, x0:x0 + nw]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA) if out.shape[1] != w else out
    if ar > target:
        nw = int(round(h * target))
        x0 = (w - nw) // 2
        return np.ascontiguousarray(rgb[:, x0:x0 + nw])
    # taller than 16:9
    ch = h
    cw = int(round(ch * target))
    bg = cv2.resize(rgb, (cw, int(round(cw / ar))), interpolation=cv2.INTER_AREA)
    y0 = max(0, (bg.shape[0] - ch) // 2)
    bg = bg[y0:y0 + ch]
    bg = cv2.GaussianBlur(bg, (0, 0), max(6, cw / 90))
    bg = (bg * 0.62).astype(np.uint8)                              # darken so the real panel reads as the subject
    out = bg.copy()
    pw = int(round(ch * ar)) if ar * ch <= cw else cw
    ph = int(round(pw / ar))
    panel = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA if pw < w else cv2.INTER_CUBIC)
    x0, y0 = (cw - pw) // 2, (ch - ph) // 2
    out[y0:y0 + ph, x0:x0 + pw] = panel
    return out


def upscale_to(rgb, target_w):
    h, w = rgb.shape[:2]
    if w == target_w:
        return rgb
    th = int(round(target_w * 9 / 16))
    if w > target_w:
        return cv2.resize(rgb, (target_w, th), interpolation=cv2.INTER_AREA)
    out = cv2.resize(rgb, (target_w, th), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(out, (0, 0), 1.4)
    return np.clip(cv2.addWeighted(out, 1.55, blur, -0.55, 0), 0, 255).astype(np.uint8)   # unsharp: crisper lineart


def _small(rgb, w=ANALYSIS_W):
    h0, w0 = rgb.shape[:2]
    return cv2.resize(rgb, (w, int(round(h0 * w / w0))), interpolation=cv2.INTER_AREA)


def _norm(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(1e-6, b - a), 0, 1).astype(np.float32)


def _srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ------------------------------------------------------------------ matte (subject silhouette)
@functools.lru_cache(1)
def _rembg_session():
    import rembg  # noqa: F401  (optional dependency)
    return rembg.new_session("isnet-anime")


def _spectral_residual(gray):
    g = cv2.resize(gray, (256, 144), interpolation=cv2.INTER_AREA).astype(np.float32)
    f = np.fft.fft2(g)
    log_amp, phase = np.log(np.abs(f) + 1e-8), np.angle(f)
    sr = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(sr + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    return cv2.resize(_norm(sal, 1, 99), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)


def _prior(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-(((xx / w - 0.5) / 0.26) ** 2 + ((yy / h - 0.55) / 0.34) ** 2) / 2)


def _matte(small, no_humans):
    """-> (float32 HxW 0..1, method)"""
    h, w = small.shape[:2]
    try:
        import rembg
        out = rembg.remove(Image.fromarray(small), session=_rembg_session(), only_mask=True)
        m = np.asarray(out, np.float32) / 255.0
        if 0.01 < m.mean() < 0.85:
            return m, "rembg"
    except Exception as e:  # noqa: BLE001
        log.debug("rembg unavailable (%s)", e)
    if no_humans:
        return np.zeros((h, w), np.float32), "none(scenery)"
    try:
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        col = np.sqrt(((lab - cv2.GaussianBlur(lab, (0, 0), w / 6)) ** 2).sum(-1))
        score = 0.34 * _spectral_residual(gray) + 0.20 * _norm(col) + 0.46 * _prior(h, w)
        score = np.clip(score - 0.45 * _foliage(small), 0, None)       # leaves/grass are scenery, not the subject
        sw = 320
        sh = int(h * sw / w)
        s_small = cv2.resize(score, (sw, sh), interpolation=cv2.INTER_AREA)
        img_small = cv2.cvtColor(cv2.resize(small, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
        mask[s_small > np.quantile(s_small, 0.45)] = cv2.GC_PR_BGD
        mask[s_small > np.quantile(s_small, 0.62)] = cv2.GC_PR_FGD
        mask[s_small > np.quantile(s_small, 0.84)] = cv2.GC_FGD
        cv2.grabCut(img_small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n > 1:                                               # keep the component with most saliency mass
            best = max(range(1, n), key=lambda i: float(s_small[lab_cc == i].sum()))
            fg = (lab_cc == best).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        area = fg.mean()
        if 0.02 < area < 0.70:
            m = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.2)
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1), "grabcut"
    except Exception as e:  # noqa: BLE001
        log.warning("saliency/grabcut matte failed (%s) - using centre prior", e)
    return np.zeros((h, w), np.float32), "none"


# ------------------------------------------------------------------ depth
@functools.lru_cache(1)
def _depth_pipe():
    from transformers import pipeline          # optional: pip install transformers torch
    return pipeline("depth-estimation", model=cfg.depth_model)


def _depth_model(small):
    try:
        out = _depth_pipe()(Image.fromarray(small))
        d = np.asarray(out["depth"], np.float32)
        d = cv2.resize(d, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_CUBIC)
        return _norm(d, 1, 99), "depth-anything-v2"
    except Exception as e:  # noqa: BLE001
        log.info("Depth-Anything unavailable (%s) - using heuristic depth", str(e)[:100])
        return None, ""


def _depth_heuristic(small, matte, no_humans):
    h, w = small.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F))
    detail = _norm(cv2.GaussianBlur(lap, (0, 0), w / 40), 5, 95)         # sharp/detailed regions read as near
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    sky = ((hsv[..., 2] > 150) & (hsv[..., 1] < 90) & (yy < 0.45)).astype(np.float32)   # bright desaturated top = far
    sky = cv2.GaussianBlur(sky, (0, 0), w / 60)
    d = 0.58 * yy + 0.22 * detail
    d = d * (1 - 0.5 * sky)
    if matte.max() > 0:
        mb = cv2.GaussianBlur(matte, (0, 0), w / 90)
        ground = float(np.clip(np.percentile(yy[matte > 0.5], 95) if (matte > 0.5).any() else 0.8, 0.4, 0.98))
        d = d * (1 - 0.8 * mb) + 0.8 * mb * (0.72 + 0.28 * ground)         # subject sits clearly in front of its surroundings
    return _norm(d, 1, 99)


def _refine_depth(depth, small):
    """Edge-aware smoothing so depth discontinuities follow image edges (less tearing in the 2.5D mesh)."""
    d8 = (depth * 255).astype(np.uint8)
    g = cv2.resize(d8, (480, int(480 * d8.shape[0] / d8.shape[1])), interpolation=cv2.INTER_AREA)
    for _ in range(3):
        g = cv2.bilateralFilter(g, 9, 28, 7)
    d = cv2.resize(g, (d8.shape[1], d8.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    return cv2.GaussianBlur(d, (0, 0), 1.2)


# ------------------------------------------------------------------ masks
def _foliage(small):
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 24) & (h <= 95) & (s > 95) & (v > 60)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), small.shape[1] / 120)


def _sway(small, matte, depth, ground_y):
    h, w = small.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    fol = _foliage(small)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex = _norm(cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), w / 80), 10, 95)
    parts = [fol * (0.30 + 0.62 * tex)]                       # flat green fills sway less than leafy texture
    if matte.max() > 0:
        k = max(3, int(w / 60)) | 1
        edge = cv2.dilate(matte, np.ones((k, k), np.uint8)) - cv2.erode(matte, np.ones((k, k), np.uint8))
        edge = cv2.GaussianBlur(edge, (0, 0), w / 160)
        ys = np.where(matte.max(1) > 0.5)[0]
        if len(ys):
            top, bot = ys[0] / h, ys[-1] / h
            rel = np.clip((yy - top) / max(1e-3, bot - top), 0, 1)
            head = np.clip(1 - rel / 0.45, 0, 1)             # hair: upper 45 % of the figure
            skirt = np.clip((rel - 0.5) / 0.5, 0, 1)          # cape/skirt hem: lower half, softer
            parts += [edge * head * 0.9, edge * skirt * 0.55, cv2.GaussianBlur(matte, (0, 0), w / 100) * head * 0.22]
    grass = ((yy > ground_y - 0.02) & (fol > 0.3)).astype(np.float32) * 0.2
    m = np.max(parts + [grass], axis=0)
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), w / 200)


def _rim(matte):
    if matte.max() <= 0:
        return np.zeros_like(matte)
    w = matte.shape[1]
    k = max(3, int(w / 150)) | 1
    band = cv2.dilate(matte, np.ones((k * 2 + 1, k * 2 + 1), np.uint8)) - cv2.erode(matte, np.ones((k, k), np.uint8))
    return np.clip(cv2.GaussianBlur(band, (0, 0), w / 400) * 1.4, 0, 1)


def _subject(matte, depth, saliency_hint, no_humans):
    h, w = matte.shape
    if matte.max() > 0 and (matte > 0.5).mean() > 0.01:
        ys, xs = np.where(matte > 0.5)
        y0, y1 = np.percentile(ys, [3, 97])
        top = (ys <= y0 + 0.35 * (y1 - y0))
        sx = float(np.median(xs[top])) / w if top.any() else float(np.median(xs)) / w
        sy = float((y0 + 0.2 * (y1 - y0)) / h)                  # about face / eye level
    else:
        yy, xx = np.unravel_index(int(np.argmax(cv2.GaussianBlur(saliency_hint, (0, 0), w / 30))), saliency_hint.shape)
        sx, sy = xx / w, yy / h
        if no_humans:
            sx, sy = 0.5 * sx + 0.25, 0.5 * sy + 0.25               # keep the focus point sane on scenery
    sx, sy = float(np.clip(sx, 0.1, 0.9)), float(np.clip(sy, 0.12, 0.88))
    r = max(2, int(w / 60))
    x, y = int(sx * w), int(sy * h)
    sd = float(np.median(depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]))
    return {"x": round(sx, 4), "y": round(sy, 4), "depth": round(float(np.clip(sd, 0, 1)), 4)}


def _ground_y(small, matte):
    h = small.shape[0]
    if matte.max() > 0 and (matte > 0.5).mean() > 0.01:
        ys = np.where(matte.max(1) > 0.5)[0]
        return float(np.clip(ys[-1] / h, 0.62, 0.95))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (0, 0), 2).astype(np.float32)
    row = np.abs(np.diff(gray.mean(1)))
    lo, hi = int(h * 0.55), int(h * 0.92)
    if hi - lo > 4 and row[lo:hi].max() > 6:
        return float(np.clip((lo + int(np.argmax(row[lo:hi]))) / h, 0.6, 0.93))
    return 0.85


def _palette(small):
    f = small.astype(np.float32) / 255.0
    lin = _srgb_to_linear(f)
    h, w = lin.shape[:2]
    mean = lambda a: tuple(round(float(x), 4) for x in a.reshape(-1, 3).mean(0))   # noqa: E731
    data = f.reshape(-1, 3)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    score = (hsv[:, 1] / 255.0) * (hsv[:, 2] / 255.0) ** 1.5
    top = np.argsort(score)[-max(50, len(score) // 200):]
    acc = _srgb_to_linear(data[top].mean(0))
    return {"avg": mean(lin), "top": mean(lin[: h // 5]), "bottom": mean(lin[-h // 5:]),
            "left": mean(lin[:, : w // 5]), "right": mean(lin[:, -w // 5:]),
            "accent": tuple(round(float(x), 4) for x in acc)}


def _backdrop(rgb_small):
    """Blurred, reflect-padded 1.5x canvas: the camera never sees an edge of the plate."""
    h, w = rgb_small.shape[:2]
    bw, bh = int(round(w * BACKDROP_SCALE)), int(round(h * BACKDROP_SCALE))
    px, py = (bw - w) // 2, (bh - h) // 2
    tiny = cv2.resize(rgb_small, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    tiny = cv2.GaussianBlur(tiny, (0, 0), 3.0)
    big = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_CUBIC)
    pad = cv2.copyMakeBorder(big, py, bh - h - py, px, bw - w - px, cv2.BORDER_REFLECT_101)
    near = cv2.GaussianBlur(pad, (0, 0), 6.0).astype(np.float32)
    far = cv2.GaussianBlur(pad, (0, 0), 26.0).astype(np.float32) * 0.82          # outside the plate: no recognisable mirror copies
    m = np.zeros((bh, bw), np.float32)
    m[py:py + h, px:px + w] = 1.0
    m = 1.0 - cv2.GaussianBlur(m, (0, 0), w * 0.04)                                # 0 inside the plate, 1 far outside
    out = near * (1 - m[..., None]) + far * m[..., None]
    return (out * 0.92).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------ public
def _save_gray(arr01, path, max_w=MASK_MAX_W):
    a = (np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.shape[1] > max_w:
        a = cv2.resize(a, (max_w, int(round(a.shape[0] * max_w / a.shape[1]))), interpolation=cv2.INTER_AREA)
    Image.fromarray(a, "L").save(path, optimize=True)
    return str(path)


def build_layers(image_path, out_dir, stem, *, no_humans=False, target_w=None, force=False):
    """-> {"image","backdrop","depth","sway","rim","subject","palette","ground_y","methods"} (absolute paths)."""
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ext = "png" if cfg.art_plate_format == "png" else "jpg"
    meta_p = out / f"{stem}_layers.json"
    paths = {"image": out / f"{stem}.{ext}", "backdrop": out / f"{stem}_backdrop.jpg", "depth": out / f"{stem}_depth.png",
             "sway": out / f"{stem}_sway.png", "rim": out / f"{stem}_rim.png"}
    if not force and meta_p.exists() and all(p.exists() for p in paths.values()):
        return json.loads(meta_p.read_text("utf-8"))

    rgb = fit_16x9(load_rgb(image_path))
    plate = upscale_to(rgb, int(target_w or cfg.art_target_w))
    small = _small(plate)
    matte, m_how = _matte(small, no_humans)
    depth, d_how = _depth_model(small)
    if depth is None:
        depth, d_how = _depth_heuristic(small, matte, no_humans), "heuristic"
    depth = _refine_depth(depth, small)
    ground = _ground_y(small, matte)
    sal = _spectral_residual(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    subject = _subject(matte, depth, sal, no_humans)
    sway = _sway(small, matte, depth, ground)
    rim = _rim(matte)
    palette = _palette(small)

    if ext == "png":
        Image.fromarray(plate).save(paths["image"], compress_level=3)
    else:
        Image.fromarray(plate).save(paths["image"], quality=94, subsampling=0, optimize=True)
    Image.fromarray(_backdrop(small)).save(paths["backdrop"], quality=88)
    _save_gray(depth, paths["depth"])
    _save_gray(sway, paths["sway"])
    _save_gray(rim, paths["rim"])
    meta = {**{k: str(v) for k, v in paths.items()}, "subject": subject, "palette": palette,
            "ground_y": round(ground, 4), "methods": {"matte": m_how, "depth": d_how},
            "size": [plate.shape[1], plate.shape[0]]}
    meta_p.write_text(json.dumps(meta, indent=1), "utf-8")
    log.info("layers %s: matte=%s depth=%s subject=(%.2f,%.2f) ground=%.2f", stem, m_how, d_how,
             subject["x"], subject["y"], ground)
    return meta
