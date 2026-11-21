"""2.5D motion-comic engine: depth-driven parallax + cinematic camera + procedural VFX, piped straight into x264.

Per frame: depth-displaced remap (real parallax, not a flat Ken Burns) -> colour grade -> bloom -> particles
-> speed lines / impact flash / chromatic aberration -> vignette -> film grain -> Arabic subtitles.
"""
import functools
import math
import subprocess

import cv2
import numpy as np

from .config import FPS, PRESET, log
from . import subs

cv2.setNumThreads(1)          # we parallelise across shots, not inside OpenCV
PX, PZ = 0.030, 0.8           # lateral parallax strength / dolly parallax strength

MOODS = {
    "embers": dict(col=(30, 110, 255), n=160, vel=(-8, -120), sway=35, kind="dot", tail=0, gain=(0.90, 1.0, 1.10), bloom=0.55),
    "shadow": dict(col=(210, 50, 150), n=140, vel=(0, -70), sway=45, kind="dot", tail=0, gain=(1.12, 0.90, 1.04), bloom=0.45),
    "rain":   dict(col=(200, 190, 170), n=260, vel=(-90, 1300), sway=0, kind="streak", tail=0.02, gain=(1.08, 1.02, 0.92), bloom=0.25),
    "aura":   dict(col=(255, 160, 90), n=170, vel=(0, -240), sway=25, kind="streak", tail=0.035, gain=(1.12, 0.98, 1.02), bloom=0.65),
    "dust":   dict(col=(150, 160, 170), n=110, vel=(14, -16), sway=22, kind="dot", tail=0, gain=(1.0, 1.0, 1.0), bloom=0.25),
    "snow":   dict(col=(235, 235, 235), n=170, vel=(-20, 120), sway=40, kind="dot", tail=0, gain=(1.06, 1.03, 0.97), bloom=0.30),
}


# ------------------------------------------------------------------ cached building blocks
@functools.lru_cache(16)
def grade_lut(mood):
    x = np.arange(256) / 255.0
    y = 0.55 * (x * x * (3 - 2 * x)) + 0.45 * x                      # gentle S-curve
    lut = np.stack([np.clip(y * g * 0.98 + 0.012, 0, 1) for g in MOODS[mood]["gain"]], -1)
    return (lut * 255).astype(np.uint8).reshape(256, 1, 3)


@functools.lru_cache(8)
def vignette(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx / w - 0.5) * 1.1) ** 2 + ((yy / h - 0.5) * 1.1) ** 2
    v = np.clip(1 - 2.3 * r2, 0.28, 1)
    return cv2.merge([(v * 255).astype(np.uint8)] * 3)


@functools.lru_cache(8)
def speed_lines(w, h):
    rng = np.random.default_rng(5)
    R, out = math.hypot(w, h), []
    for _ in range(3):
        m = np.zeros((h, w), np.uint8)
        for _ in range(110):
            a, wid, r0 = rng.random() * 2 * math.pi, rng.uniform(0.004, 0.02), rng.uniform(0.17, 0.30) * R
            pts = np.array([[w / 2 + math.cos(a) * r0, h / 2 + math.sin(a) * r0],
                            [w / 2 + math.cos(a - wid) * R, h / 2 + math.sin(a - wid) * R],
                            [w / 2 + math.cos(a + wid) * R, h / 2 + math.sin(a + wid) * R]], np.int32)
            cv2.fillConvexPoly(m, pts, int(rng.uniform(120, 255)))
        out.append(cv2.cvtColor(cv2.GaussianBlur(m, (0, 0), 1.2), cv2.COLOR_GRAY2BGR))
    return tuple(out)


@functools.lru_cache(8)
def grain_pool(w, h):
    rng = np.random.default_rng(9)
    pool = []
    for _ in range(6):
        g = rng.integers(0, 256, (h // 2, w // 2), dtype=np.uint8)
        pool.append(cv2.cvtColor(cv2.resize(g, (w, h)), cv2.COLOR_GRAY2BGR))
    return pool


@functools.lru_cache(3)
def load_asset(img_path, depth_path):
    img = cv2.imread(img_path)
    d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if d.shape != img.shape[:2]:
        d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    ys, xs = np.nonzero(d >= np.percentile(d, 82))                    # nearest 18% = the subject
    h, w = d.shape
    fx = 0.5 + 0.6 * ((float(xs.mean()) / w) - 0.5) if len(xs) else 0.5
    fy = 0.5 + 0.6 * ((float(ys.mean()) / h) - 0.5) if len(ys) else 0.5
    r = max(4, int(0.05 * min(h, w)))
    cx, cy = int(fx * w), int(fy * h)
    df = float(np.median(d[max(0, cy - r):cy + r, max(0, cx - r):cx + r]))
    return img, d, fx, fy, df


class Particles:
    def __init__(self, mood, w, h, seed):
        m, rng = MOODS[mood], np.random.default_rng(seed)
        n = m["n"]
        self.m, self.w, self.h = m, w, h
        self.p0 = rng.random((n, 2)) * [w, h]
        self.v = np.array(m["vel"], np.float32)[None, :] * rng.uniform(0.5, 1.5, n)[:, None]
        self.size = rng.choice([1, 1, 2, 2, 3, 4], n)
        self.phase = rng.random(n) * 6.28
        self.amp = m["sway"] * rng.uniform(0.3, 1.2, n)
        self.bright = rng.uniform(0.45, 1.0, n)

    def draw(self, t, layer):
        m = self.m
        pos = self.p0 + self.v * t
        pos[:, 0] += np.sin(self.phase + t * 1.7) * self.amp
        pos[:, 0] %= self.w
        pos[:, 1] %= self.h
        flick = 0.65 + 0.35 * np.sin(self.phase * 3 + t * 9)
        for (x, y), s, b, f, v in zip(pos, self.size, self.bright, flick, self.v):
            col = tuple(int(c * b * f) for c in m["col"])
            if m["kind"] == "streak":
                cv2.line(layer, (int(x), int(y)), (int(x - v[0] * m["tail"]), int(y - v[1] * m["tail"])), col, 1)
            else:
                cv2.circle(layer, (int(x), int(y)), int(s), col, -1)


def cam_path(move, t):
    """-> zoom, window offset x, y (fractions of the image), lateral parallax in [-1, 1]."""
    e = t * t * (3 - 2 * t)
    if move == "push_in":
        return 1 + 0.17 * e, 0, 0, -0.5 + e
    if move == "pull_out":
        return 1.18 - 0.17 * e, 0, 0, 0.5 - e
    if move == "pan_l":
        return 1.10, 0.05 - 0.10 * e, 0, 1 - 2 * e
    if move == "pan_r":
        return 1.10, -0.05 + 0.10 * e, 0, -1 + 2 * e
    if move == "tilt_up":
        return 1.12, 0, 0.05 - 0.10 * e, 0.3
    if move == "tilt_dn":
        return 1.12, 0, -0.05 + 0.10 * e, -0.3
    if move == "orbit":
        return 1.12, 0, 0, -1 + 2 * e
    if move == "punch":                                               # slow-slow-FAST speed ramp
        return 1 + 0.40 * t ** 2.6, 0, 0, 0.2 * t
    return 1.05 + 0.03 * t, -0.015 + 0.03 * t, 0, 0                   # drift


def chroma(fr, s):
    h, w = fr.shape[:2]
    b, g, r = cv2.split(fr)
    b = cv2.warpAffine(b, np.float32([[1, 0, s], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    r = cv2.warpAffine(r, np.float32([[1, 0, -s], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b, g, r])


# ------------------------------------------------------------------ renderer
def _render(t):
    img, depth, fx, fy, df = load_asset(t["img"], t["depth"])
    if t.get("simple"):
        depth = np.full_like(depth, df)
    Hi, Wi = img.shape[:2]
    ow, oh = t["size"]
    n, fps = t["frames"], t.get("fps", FPS)
    dur = n / fps
    asp = ow / oh
    bh = Hi * 0.90 if Wi / Hi > asp else Wi * 0.90 / asp
    bw = bh * asp
    gw, gh = ow // 4, oh // 4
    U = np.tile(np.linspace(-.5, .5, gw, dtype=np.float32)[None, :], (gh, 1))
    V = np.tile(np.linspace(-.5, .5, gh, dtype=np.float32)[:, None], (1, gw))
    rng = np.random.default_rng(t["seed"])
    raw = rng.standard_normal((n + 16, 2))
    k7 = np.ones(7) / 7
    shake = np.stack([np.convolve(raw[:, j], k7, "same") for j in (0, 1)], 1) * 1.8

    mood = t["mood"] if t["mood"] in MOODS else "dust"
    grade, vg, grain = grade_lut(mood), vignette(ow, oh), grain_pool(ow, oh)
    parts = Particles(mood, ow // 2, oh // 2, t["seed"])
    heavy = t["intensity"] >= 4 or t["impact"]
    lines = speed_lines(ow, oh) if heavy else None
    sub = subs.render(t["ar"], ow, oh) if t.get("ar") else None
    zb, foc = t.get("zoom_bias", 1.0), t.get("foc_off", (0, 0))
    bloom_k = MOODS[mood]["bloom"] * 2.2

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", PRESET, "-crf", "17", "-threads", "2",
           "-pix_fmt", "yuv420p", t["out"]]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            ts, p = i / fps, i / max(1, n - 1)
            zoom, ox, oy, lat = cam_path(t["move"], p)
            zoom *= zb
            amp = 0.0010 + (0.012 * math.exp(-ts * 5.0) if t["impact"] else 0.0)
            ws, hs = bw / zoom, bh / zoom
            cx = float(np.clip((fx + foc[0] + ox + shake[i, 0] * amp) * Wi, ws / 2, Wi - ws / 2))
            cy = float(np.clip((fy + foc[1] + oy + shake[i, 1] * amp) * Hi, hs / 2, Hi - hs / 2))

            ds = cv2.remap(depth, cx + U * ws, cy + V * hs, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            k = ds - df
            zk = 1 - PZ * k * (zoom - 1)
            mx = cv2.resize(cx + U * ws * zk + lat * PX * bw * k, (ow, oh))
            my = cv2.resize(cy + V * hs * zk, (ow, oh))
            fr = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

            fr = cv2.LUT(fr, grade)
            sm = cv2.resize(fr, (ow // 4, oh // 4), interpolation=cv2.INTER_AREA)
            br = cv2.GaussianBlur(cv2.subtract(sm, (135, 135, 135, 0)), (0, 0), 6)
            fr = cv2.addWeighted(fr, 1, cv2.resize(br, (ow, oh)), bloom_k, 0)

            layer = np.zeros((oh // 2, ow // 2, 3), np.uint8)
            parts.draw(((i // 2) * 2) / fps, layer)                   # particles "on twos" = anime stepping
            layer = cv2.add(layer, cv2.GaussianBlur(layer, (0, 0), 2.2))
            fr = cv2.add(fr, cv2.resize(layer, (ow, oh)))

            if lines is not None:
                a = (0.8 * max(0.25, 1 - ts / 0.9)) if t["impact"] else 0.30
                fr = cv2.addWeighted(fr, 1, lines[(i // 2) % 3], a, 0)
            if mood in ("embers", "aura"):
                fr = cv2.convertScaleAbs(fr, alpha=1 + 0.04 * math.sin(ts * 13))
            if t["impact"] and ts < 0.5:
                s = int(14 * math.exp(-ts * 9))
                if s > 0:
                    fr = chroma(fr, s)
                v = int(210 * max(0.0, 1 - ts / 0.18))
                if v > 0:
                    fr = cv2.add(fr, (v, v, v, 0))

            tr = t.get("trans", "cut")
            if tr == "flash" and ts < 0.25:
                v = int(230 * (1 - ts / 0.25))
                fr = cv2.add(fr, (v, v, v, 0))
            elif tr == "whip" and ts < 0.25:
                kk = int(1 + 120 * (1 - ts / 0.25) ** 2)
                if kk > 1:
                    fr = cv2.blur(fr, (kk, 1))
            elif tr == "fade" and ts < 0.6:
                fr = cv2.convertScaleAbs(fr, alpha=ts / 0.6)

            fr = cv2.multiply(fr, vg, scale=1 / 255.0)
            fr = cv2.addWeighted(fr, 1, grain[i % len(grain)], 0.07, -0.07 * 128)

            if sub is not None:
                col, alpha, y0 = sub
                hh = alpha.shape[0]
                a = alpha[..., None] * min(1.0, ts / 0.15)
                reg = fr[y0:y0 + hh]
                reg[:] = (reg * (1 - a) + col * a).astype(np.uint8)
            if t.get("fade_out") and ts > dur - 0.6:
                fr = cv2.convertScaleAbs(fr, alpha=max(0.0, (dur - ts) / 0.6))
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")


def render_shot(t):
    """Worker entry. A broken shot degrades to a flat-depth render instead of killing the episode."""
    try:
        _render(t)
    except Exception as e:  # noqa: BLE001
        log.warning("shot %s failed (%r) - retrying in simple mode", t["out"], e)
        _render(dict(t, simple=True))
    return t["out"]
