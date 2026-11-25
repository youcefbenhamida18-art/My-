"""No-Blender renderer: the same shot dict, the same camera language, the same plate/camera model (plate_geometry) - done with
numpy + OpenCV. Used when Blender is missing/fails, for fast previews (RENDER_ENGINE=cv) and by the tests.

Depth-based parallax (iterated backward mapping), shader-style wind/hair sway and heat haze, analytic particles, fog, rays,
speed lines, slash/shockwave/lightning, rim glow, depth of field with rack focus, directional motion blur on whips.
Writes contiguous PNG drawings `f_0001.png ...` and `done.json`, exactly like blender_scene.py.
"""
from __future__ import annotations

import json
import math
import pathlib

import cv2
import numpy as np

from .. import camera_paths as cp
from ..config import log
from . import plate_geometry as pg
from .eye_rig import EyeRig

TAU = 2 * math.pi


def _noise(h, w, cell, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
    n = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (np.abs(n).max() + 1e-6)


def _scroll(a, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _load_gray(path, w, h):
    if not path:
        return None
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


class Layers:
    """Per-drawing scratch layers: `over` (BGRA, alpha composite), `add` (BGR, additive light)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.over = np.zeros((h, w, 4), np.uint8)
        self.add = np.zeros((h, w, 3), np.uint8)
        self.front = np.zeros((h, w, 4), np.uint8)               # blurred foreground (petals close to camera, drops...)

    def px(self, f):
        return max(1, int(round(f * self.h)))


def _bgr(rgb):
    r, g, b = [int(max(0, min(1, c)) * 255) for c in rgb]
    return (b, g, r)


def png_complete(path):
    """True only for a fully written PNG (ends with the IEND chunk): a job killed mid-write leaves a truncated file that must be redrawn."""
    try:
        with open(path, "rb") as fh:
            fh.seek(-12, 2)
            return fh.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
    except OSError:
        return False


class Engine:
    def __init__(self, shot, settings):
        self.shot, self.set = shot, settings or {}
        sc = float(self.set.get("scale", 1.0))
        self.W = max(2, int(shot["size"][0] * sc) // 2 * 2)
        self.H = max(2, int(shot["size"][1] * sc) // 2 * 2)
        self.fps, self.step, self.frames = shot["fps"], shot["step"], shot["frames"]
        self.seed = int(shot.get("seed", 1))
        self.rng = np.random.default_rng(self.seed)
        pl = shot["plate"]
        fr = shot["framing"]
        self.zoom, self.cx, self.cy = fr["zoom"], fr["cx"], fr["cy"]
        self.fw = cp.PLATE_MARGIN / self.zoom
        cam = shot["camera"]
        self.cam = cam
        self.k = cam.get("focal", 35) / pg.SENSOR_W
        sub = pl.get("subject") or {"x": 0.5, "y": 0.5, "depth": 0.8}
        self.subject = sub
        self.subject_uv = ((sub["x"] - self.cx) / self.fw, (sub["y"] - self.cy) / self.fw)      # where the subject sits at the rest camera
        self.subject_w = pg.subject_world(self.subject_uv, cam.get("focal", 35))
        self.parallax = float(pl.get("parallax", 0.35))
        # plate resolution adapted to the deepest zoom this shot reaches (keeps remap sharp without aliasing)
        need = self.W * 1.15 * self.zoom / cp.PLATE_MARGIN / max(0.55, 1 - 0.32 * cam.get("strength", 0.5))
        plate = cv2.imread(str(pl["image"]), cv2.IMREAD_COLOR)
        if plate is None:
            raise FileNotFoundError(f"plate image unreadable: {pl['image']}")
        pw = plate.shape[1]
        if need < pw:
            plate = cv2.resize(plate, (int(need), int(need * plate.shape[0] / pw)), interpolation=cv2.INTER_AREA)
        self.plate = plate
        self.seq = pl.get("sequence")
        dh, dw = 270, 480
        self.depth = _load_gray(pl["depth"], dw, dh)
        if self.depth is None:
            self.depth = np.full((dh, dw), 0.5, np.float32)
        self.sway = _load_gray(pl.get("sway"), dw, dh)
        self.rim = _load_gray(pl.get("rim"), self.W // 2, self.H // 2)
        bd = cv2.imread(str(pl["backdrop"]), cv2.IMREAD_COLOR) if pl.get("backdrop") else None
        self.backdrop = bd if bd is not None else cv2.GaussianBlur(cv2.resize(self.plate, (480, 270)), (0, 0), 12)
        pal = pl.get("palette") or {}
        self.accent = tuple(pal.get("accent") or (1.0, 0.8, 0.5))
        self.avg = tuple(pal.get("avg") or (0.3, 0.3, 0.3))
        self.light = tuple(pal.get("top") or pal.get("avg") or (0.6, 0.6, 0.7))
        # the rim light hugs the subject's silhouette: only trust it when the matte is accurate (rembg), fade it for heuristics
        self.rim_scale = {"rembg": 1.0, "grabcut": 0.3}.get(pl.get("matte_method", "rembg"), 0.0)
        fx = shot.get("fx") or []
        self.fx = {f["kind"]: f for f in fx}
        self.ok = {}
        self.noiseA = _noise(self.H // 2, self.W // 2, 24, self.seed)
        self.noiseB = _noise(self.H // 2, self.W // 2, 40, self.seed + 1)
        self.fognoise = [_noise(self.H // 2, self.W // 2, 90, self.seed + 5), _noise(self.H // 2, self.W // 2, 150, self.seed + 6)]
        self.grid_u = ((np.arange(self.W, dtype=np.float32) + 0.5) / self.W - 0.5)[None, :] * np.ones((self.H, 1), np.float32)
        self.grid_v = ((np.arange(self.H, dtype=np.float32) + 0.5) / self.H - 0.5)[:, None] * np.ones((1, self.W), np.float32)
        self.wind = shot.get("wind") or [0.4, 0.05]
        self.hit_at = cam.get("hit_at")
        self.seqcache = {}
        self._prev_pose = None
        self.eyes = None if self.seq else EyeRig(shot, self.plate.shape)        # blink / gaze / catch-light / glow / tears on the plate
        if self.eyes and self.eyes.on:
            self.ok.update(self.eyes.report)
        self.title = self._load_title(shot.get("title3d"))
        self._tgrid = None

    # ---------------------------------------------------------------- helpers
    def active(self, kind, t):
        f = self.fx.get(kind)
        if not f:
            return 0.0
        at, dur = f.get("at") or 0.0, f.get("dur")
        if t < at or (dur is not None and t > at + dur):
            return 0.0
        return float(f.get("amount", 0.6))

    def mark(self, kind):
        self.ok[kind] = "ok"

    def plate_at(self, t):
        if not self.seq:
            return self.plate
        n = int(self.seq.get("frames", 0))
        if n <= 0:
            return self.plate
        i = min(n, 1 + int(t * self.seq.get("fps", self.fps)))
        if i not in self.seqcache:
            p = pathlib.Path(self.seq["dir"]) / (self.seq["pattern"] % i)
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                im = cv2.resize(im, (self.plate.shape[1], self.plate.shape[0]), interpolation=cv2.INTER_AREA)
            self.seqcache = {i: im if im is not None else self.plate}
        return self.seqcache[i]

    def sample_depth(self, x, y):
        """x, y plate coords in 0..1 -> depth 0..1 (bilinear)."""
        dh, dw = self.depth.shape
        return cv2.remap(self.depth, (x * (dw - 1)).astype(np.float32), (y * (dh - 1)).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---------------------------------------------------------------- core warp
    def warp(self, t, f, pose):
        """Backward map for one drawing: output pixel -> plate coords (x, y in 0..1) and relative depth z. Rays are cast through the
        SAME exact pinhole camera the Blender scene uses (plate_geometry.camera_pose_world) and intersected with the depth surface
        by a short fixed-point iteration on z (the plate point at depth z lies on the world plane y = z * D)."""
        f_mm = self.k * pg.SENSOR_W
        R, C = pg.camera_pose_world(pose, self.subject_w, f_mm)
        fs = pose["focal_scale"]
        u, v = self.grid_u, self.grid_v
        sd = self.subject["depth"]
        z = np.ones_like(u)
        for _ in range(3):
            u0, v0 = pg.hit_plate(u, v, z, R, C, f_mm, fs)
            z = pg.zeta(self.sample_depth(self.cx + u0 * self.fw, self.cy + v0 * self.fw), sd, self.parallax).astype(np.float32)
        u0, v0 = pg.hit_plate(u, v, z, R, C, f_mm, fs)
        return (self.cx + u0 * self.fw).astype(np.float32), (self.cy + v0 * self.fw).astype(np.float32), z

    def wind_field(self, t, amp_px):
        """Pixel displacement from the sway mask (hair / foliage / grass / cloth)."""
        want = ("wind_sway" in self.fx) or ("hair_sway" in self.fx) or ("grass" in self.fx) or ("cloth_flag" in self.fx)
        if not want or self.sway is None:
            return None
        W, H = self.W, self.H
        sway = cv2.resize(self.sway, (W, H), interpolation=cv2.INTER_LINEAR)
        gust = 0.55 + 0.45 * math.sin(TAU * 0.33 * t + self.seed % 7)
        wx = self.wind[0]
        a = cv2.resize(_scroll(self.noiseA, -wx * 90 * t, 6 * t), (W, H))
        b = cv2.resize(_scroll(self.noiseB, -wx * 60 * t, -9 * t), (W, H))
        ux = (0.7 * a + 0.3 * math.sin(TAU * 0.6 * t)) * sway * gust * amp_px
        uy = (0.6 * b) * sway * gust * amp_px * 0.55
        for k in ("wind_sway", "hair_sway", "grass", "cloth_flag"):
            if k in self.fx:
                self.mark(k)
        return ux, uy

    def haze_field(self, t, amp_px):
        out_x = out_y = None
        if "heat_haze" in self.fx:
            n = cv2.resize(_scroll(self.noiseA, 0, -140 * t), (self.W, self.H))
            m = cv2.resize(_scroll(self.noiseB, 70 * t, 0), (self.W, self.H))
            out_x, out_y = n * amp_px * 0.55, m * amp_px * 0.35
            self.mark("heat_haze")
        if "water_ripple" in self.fx:
            yy = (self.grid_v + 0.5)
            band = np.clip((yy - 0.55) / 0.2, 0, 1)
            ripple = np.sin(self.grid_u * 90 + t * 3.0 + yy * 40) * band * amp_px * 0.5
            out_x = ripple if out_x is None else out_x + ripple
            out_y = np.zeros_like(ripple) if out_y is None else out_y
            self.mark("water_ripple")
        return None if out_x is None else (out_x, out_y)

    # ---------------------------------------------------------------- render one drawing
    def drawing(self, f):
        W, H = self.W, self.H
        t = (f - 1) / self.fps
        tn = (f - 1) / max(1, self.frames - 1)
        hit_frame = None if self.hit_at is None else 1 + int(round(self.hit_at * self.fps))
        pose = cp.total_pose(self.cam["move"], tn, self.cam["strength"], self.seed, f, self.fps,
                             self.cam.get("shake", 0.0), hit_frame)
        x, y, z = self.warp(t, f, pose)
        plate = self.plate_at(t)
        if self.eyes is not None and self.eyes.on:
            plate = self.eyes.apply(plate, t)
        ph, pw = plate.shape[:2]
        scale_px = (self.fw * pw) / W                            # plate pixels per output pixel
        amp = 0.0055 * W * (self.cam.get("focal", 35) / 35.0 * 0 + 1)
        wf = self.wind_field(t, amp)
        hz = self.haze_field(t, 0.004 * W)
        for fld in (wf, hz):
            if fld is not None:
                x = x + fld[0] * (scale_px / pw)
                y = y + fld[1] * (scale_px / ph)
        shock = self.shock_field(t)
        if shock is not None:
            x = x + shock[0] * (scale_px / pw)
            y = y + shock[1] * (scale_px / ph)
        img = cv2.remap(plate, x * (pw - 1), y * (ph - 1), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        out_of = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        if out_of.any():                                          # beyond the artwork: blurred backdrop, never an edge
            bh, bw = self.backdrop.shape[:2]
            bx = (0.5 + (x - 0.5) / 1.5) * (bw - 1)
            by = (0.5 + (y - 0.5) / 1.5) * (bh - 1)
            bd = cv2.remap(self.backdrop, bx.astype(np.float32), by.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            m = cv2.GaussianBlur(out_of.astype(np.float32), (0, 0), 6)[..., None]
            img = (img * (1 - m) + bd * m).astype(np.uint8)
        self.mark("plate_depth_parallax")
        img = self.dof(img, z, t, tn)
        img = self.motion_blur(img, pose, f)
        fimg = img.astype(np.float32) / 255.0
        fimg = self.rim_glow(fimg, t)
        L = Layers(W, H)
        self.draw_fx(L, t, f, tn, pose)
        a = L.over[..., 3:4].astype(np.float32) / 255.0
        fimg = fimg * (1 - a) + (L.over[..., :3].astype(np.float32) / 255.0) * a
        if L.front.any():
            fa = cv2.GaussianBlur(L.front, (0, 0), self.H / 220.0)
            fa_a = fa[..., 3:4].astype(np.float32) / 255.0
            fimg = fimg * (1 - fa_a) + (fa[..., :3].astype(np.float32) / 255.0) * fa_a
        glow = L.add.astype(np.float32) / 255.0
        if glow.any():
            g2 = cv2.GaussianBlur(glow, (0, 0), self.H / 90.0)
            fimg = fimg + glow * 0.9 + g2 * 0.8
        self._prev_pose = pose
        if self.title:
            fimg = self.draw_title(fimg, t)
        return np.clip(fimg * 255 + 0.5, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- title card (3 layers: glow / floating extrusion / face + light sweep)
    def _load_title(self, t3):
        if not t3 or not t3.get("layers"):
            return None
        out = {}
        for k in ("face", "extrude", "glow"):
            im = cv2.imread(str(t3["layers"][k]), cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                return None
            im = cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32)
            out[k] = np.dstack([im[..., :3], im[..., 3:4] / 255.0 * 255.0])          # BGR + alpha(0..255)
        out["at"], out["dur"] = float(t3["at"]), float(t3["dur"])
        self.ok["title3d"] = "ok"
        return out

    def draw_title(self, fimg, t):
        T = self.title
        k, D = t - T["at"], T["dur"]
        if k < 0 or k > D:
            return fimg
        W, H = self.W, self.H
        x = min(1.0, k / 0.6)
        c1 = 1.70158
        back = 1.0 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2
        scale = (0.86 + 0.14 * back) * (1.0 + 0.04 * k / D)
        out_k = min(1.0, max(0.0, (k - (D - 0.5)) / 0.5))
        alpha = min(1.0, k / 0.25) * (1.0 - out_k * out_k * (3 - 2 * out_k))
        M = cv2.getRotationMatrix2D((W / 2, H * 0.5), 1.6 * math.sin(k * 0.9), scale)
        sk = W / 1920.0

        def tf(layer, off=(0.0, 0.0)):
            m = M.copy()
            m[0, 2] += off[0] * sk
            m[1, 2] += off[1] * sk
            return cv2.warpAffine(layer, m, (W, H), flags=cv2.INTER_LINEAR)

        g = tf(T["glow"])
        fimg = fimg + g[..., :3] / 255.0 * (g[..., 3:4] / 255.0) * alpha * (0.65 + 0.35 * math.sin(k * 3.0))
        e = tf(T["extrude"], (6 * math.sin(k * 1.3 + 1), 7 * math.cos(k * 1.1)))
        ea = e[..., 3:4] / 255.0 * alpha
        fimg = fimg * (1 - ea) + e[..., :3] / 255.0 * ea
        f = tf(T["face"])
        if self._tgrid is None:
            self._tgrid = (self.grid_u + 0.5) * 0.85 + (self.grid_v + 0.5) * 0.25
        pos = (k - 0.7) / 1.1 * 1.5 - 0.25
        band = np.exp(-(((self._tgrid - pos) / 0.07) ** 2))[..., None]
        fa = f[..., 3:4] / 255.0 * alpha
        face = np.clip(f[..., :3] + band * 230.0, 0, 255) / 255.0
        return fimg * (1 - fa) + face * fa

    # ---------------------------------------------------------------- lens / light
    def dof(self, img, z, t, tn):
        if not self.cam.get("dof"):
            self.ok["dof"] = "off"
            return img
        rack = self.cam.get("rack")
        sd = self.subject["depth"]
        if rack:
            e = cp.ease_in_out(tn)
            fd = rack["from"] + (rack["to"] - rack["from"]) * e
            self.mark("rack_focus")
        else:
            fd = sd
        fz = 1.0 - self.parallax * (fd - sd)
        coc = np.clip(np.abs(z - fz) * 5.0, 0, 1)
        sm = cv2.resize(img, (self.W // 2, self.H // 2), interpolation=cv2.INTER_AREA)
        lv = [img, cv2.resize(cv2.GaussianBlur(sm, (0, 0), 2.0), (self.W, self.H)),
              cv2.resize(cv2.GaussianBlur(sm, (0, 0), 5.0), (self.W, self.H)),
              cv2.resize(cv2.GaussianBlur(sm, (0, 0), 10.0), (self.W, self.H))]
        s = coc * 3.0
        i0 = np.clip(np.floor(s).astype(np.int32), 0, 2)
        fr = (s - i0)[..., None]
        stack = np.stack(lv, 0).astype(np.float32)
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        a, b = stack[i0, yy, xx], stack[np.minimum(i0 + 1, 3), yy, xx]
        self.mark("dof")
        return (a * (1 - fr) + b * fr).astype(np.uint8)

    def motion_blur(self, img, pose, f):
        if self._prev_pose is None:
            return img
        p = self._prev_pose
        f_mm = self.k * pg.SENSOR_W
        a = pg.project(0.0, 0.0, 1.0, p, f_mm, self.subject_uv)                # screen motion of the subject plane's centre
        b = pg.project(0.0, 0.0, 1.0, pose, f_mm, self.subject_uv)
        du, dv = float(b[0] - a[0]), float(b[1] - a[1])
        mag = math.hypot(du * self.W, dv * self.H)
        n = int(min(41, mag * 0.5)) | 1
        if n < 5:
            return img
        ker = np.zeros((n, n), np.float32)
        ang = math.atan2(dv * self.H, du * self.W)
        c = n // 2
        cv2.line(ker, (int(c - math.cos(ang) * c), int(c - math.sin(ang) * c)), (int(c + math.cos(ang) * c), int(c + math.sin(ang) * c)), 1.0, 1)
        ker /= ker.sum() + 1e-6
        self.mark("motion_blur")
        return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REPLICATE)

    def rim_glow(self, fimg, t):
        if self.rim is None or self.rim.max() < 0.02 or self.rim_scale <= 0:
            return fimg
        amt = 0.17 + 0.09 * math.sin(TAU * 0.5 * t) + (0.30 if "aura" in self.fx else 0.0) + 0.2 * self.active("glow_pulse", t)
        r = cv2.resize(self.rim, (self.W, self.H), interpolation=cv2.INTER_LINEAR)[..., None] * self.rim_scale
        top = np.clip(np.array(self.light, np.float32) ** (1 / 2.2), 0, 1)           # light colour: the sky/key-light side of the plate
        col = (top / (top.max() + 1e-6) * 0.65 + 0.35)[::-1][None, None, :]          # RGB -> BGR, pushed towards white
        self.mark("rim_glow")
        if "aura" in self.fx:
            self.mark("aura")
        return fimg + r * col * amt

    def shock_field(self, t):
        a = self.active("shockwave", t)
        if not a or self.hit_at is None:
            return None
        age = t - (self.fx["shockwave"].get("at") or self.hit_at)
        if age < 0 or age > 0.8:
            return None
        cx, cy = self.subject["x"], self.subject["y"]
        # centre in frame pixels via the framing
        u = 0.5 + (cx - self.cx) / self.fw
        v = 0.5 + (cy - self.cy) / self.fw
        R = age * 1.6 * self.W
        dx_ = self.grid_u * self.W - (u - 0.5) * self.W
        dy_ = self.grid_v * self.H - (v - 0.5) * self.H
        r = np.sqrt(dx_ ** 2 + dy_ ** 2) + 1e-3
        ring = np.exp(-((r - R) / (0.05 * self.W)) ** 2) * (1 - age / 0.8) * a
        k = ring * 0.03 * self.W
        self.mark("shockwave")
        return (dx_ / r * k, dy_ / r * k)

    # ---------------------------------------------------------------- effects
    def to_px(self, px_, py_):
        u = 0.5 + (px_ - self.cx) / self.fw
        v = 0.5 + (py_ - self.cy) / self.fw
        return u * self.W, v * self.H

    def draw_fx(self, L, t, f, tn, pose):
        W, H = self.W, self.H
        rng = np.random.default_rng(self.seed + 13)
        ac = _bgr(self.accent)
        wx = self.wind[0]
        for kind in ("petals", "leaves"):
            a = self.active(kind, t)
            if a:
                n = int(40 + 70 * a)
                col = (180, 170, 255) if kind == "petals" else (50, 120, 70)
                for i in range(n):
                    r1, r2, r3, r4 = rng.random(4)
                    x = (r1 * 1.3 - 0.15 + wx * 0.08 * t + 0.03 * math.sin(TAU * (0.4 + r3) * t + r4 * 6)) % 1.3 - 0.15
                    y = (r2 + (0.07 + 0.08 * r3) * t) % 1.12 - 0.06
                    sz = L.px(0.006 + 0.010 * r3)
                    ang = (r4 * 360 + t * 90 * (0.5 + r1)) % 360
                    lay = L.front if r3 > 0.86 else L.over
                    cv2.ellipse(lay, (int(x * W), int(y * H)), (sz, max(1, sz // 2)), ang, 0, 360,
                                (*col, 230), -1, cv2.LINE_AA)
                self.mark(kind)
        a = self.active("snow", t)
        if a:
            for i in range(int(90 + 120 * a)):
                r1, r2, r3, r4 = rng.random(4)
                x = (r1 + wx * 0.05 * t + 0.02 * math.sin(TAU * 0.3 * t + r4 * 6)) % 1.0
                y = (r2 + (0.05 + 0.1 * r3) * t) % 1.0
                lay = L.front if r3 > 0.9 else L.over
                cv2.circle(lay, (int(x * W), int(y * H)), L.px(0.002 + 0.005 * r3), (245, 245, 255, 210), -1, cv2.LINE_AA)
            self.mark("snow")
        a = self.active("rain", t)
        if a:
            ang = math.radians(100 + 20 * wx)
            for i in range(int(160 + 260 * a)):
                r1, r2, r3 = rng.random(3)
                ln = 0.03 + 0.05 * r3
                x = (r1 + 0.1 * math.cos(ang) * t * 4) % 1.1
                y = (r2 + 1.4 * t * (0.8 + r3)) % 1.1
                x2, y2 = x - ln * math.cos(ang) * 0.4, y - ln
                cv2.line(L.over, (int(x * W), int(y * H)), (int(x2 * W), int(y2 * H)), (235, 225, 215, int(90 + 70 * r3)), 1, cv2.LINE_AA)
            self.mark("rain")
        a = self.active("dust", t)
        if a:
            for i in range(int(60 + 80 * a)):
                r1, r2, r3, r4 = rng.random(4)
                x = (r1 + 0.02 * t * (r3 + wx) + 0.01 * math.sin(TAU * 0.2 * t + r4 * 6)) % 1.0
                y = (r2 - 0.012 * t * (0.5 + r3) + 0.01 * math.sin(TAU * 0.25 * t + r1 * 6)) % 1.0
                tw = 0.6 + 0.4 * math.sin(TAU * (0.3 + r3) * t + r4 * 9)
                cv2.circle(L.add, (int(x * W), int(y * H)), L.px(0.0016 + 0.003 * r3), tuple(int(c * 0.7 * tw) for c in (230, 235, 255)), -1, cv2.LINE_AA)
            self.mark("dust")
        for kind, base in (("embers", (40, 140, 255)), ("sparks", (120, 220, 255)), ("fireflies", (90, 250, 190))):
            a = self.active(kind, t)
            if not a:
                continue
            n = int((50 + 80 * a) if kind != "sparks" else 40)
            for i in range(n):
                r1, r2, r3, r4 = rng.random(4)
                if kind == "embers":
                    x = (r1 + 0.03 * math.sin(TAU * 0.4 * t + r4 * 6) + wx * 0.03 * t) % 1.0
                    y = (r2 - (0.06 + 0.08 * r3) * t) % 1.0
                    fl = 0.5 + 0.5 * math.sin(TAU * (2 + 3 * r3) * t + r4 * 9)
                elif kind == "fireflies":
                    x = (r1 + 0.05 * math.sin(TAU * 0.13 * t + r4 * 6) + 0.02 * t * (r3 - .5)) % 1.0
                    y = (r2 + 0.05 * math.sin(TAU * 0.17 * t + r3 * 6)) % 1.0
                    fl = max(0.0, math.sin(TAU * (0.35 + 0.4 * r3) * t + r4 * 9)) ** 2
                else:
                    age = t - (self.fx[kind].get("at") or 0)
                    if age < 0 or age > 0.9:
                        continue
                    ox, oy = self.to_px(self.subject["x"], self.subject["y"])
                    ang = r1 * TAU
                    sp = (0.25 + 0.9 * r3) * W
                    px_, py_ = ox + math.cos(ang) * sp * age, oy + math.sin(ang) * sp * age + 0.8 * H * age ** 2 * r2
                    fl = 1.0 - age / 0.9
                    cv2.line(L.add, (int(px_), int(py_)), (int(px_ - math.cos(ang) * sp * 0.05), int(py_ - math.sin(ang) * sp * 0.05)),
                             tuple(int(c * fl) for c in base), 2, cv2.LINE_AA)
                    continue
                cv2.circle(L.add, (int(x * W), int(y * H)), L.px(0.0025 + 0.004 * r3), tuple(int(c * fl) for c in base), -1, cv2.LINE_AA)
            self.mark(kind)
        for kind in ("mist", "volume_fog"):
            a = self.active(kind, t)
            if a:
                m = 0
                for i, n in enumerate(self.fognoise):
                    s = cv2.resize(_scroll(n, (8 + 10 * i) * t * (1 if i else -1), 0), (W, H))
                    m = m + (s * 0.5 + 0.5) * (0.6 if i == 0 else 0.4)
                m = np.clip((m - 0.25) * 1.5, 0, 1) * a * 0.55
                col = np.clip(np.array(self.avg) ** (1 / 2.2) * 0.6 + 0.35, 0, 1)
                layer = (m[..., None] * np.array(col[::-1])[None, None, :] * 255).astype(np.uint8)
                L.add[:] = np.minimum(255, L.add.astype(np.int32) + (layer * 0.55).astype(np.int32)).astype(np.uint8)
                self.mark(kind)
        a = self.active("light_rays", t)
        if a:
            ox, oy = self.to_px(0.72, 0.1)
            for i in range(14):
                r1, r2 = rng.random(2)
                ang = math.radians(100 + 70 * r1 + 3 * math.sin(TAU * 0.1 * t + i))
                ln = W * 1.3
                p1 = (int(ox), int(oy))
                p2 = (int(ox + math.cos(ang) * ln), int(oy + math.sin(ang) * ln))
                cv2.line(L.add, p1, p2, tuple(int(c * (0.18 + 0.12 * r2) * a) for c in (210, 235, 255)), L.px(0.02 + 0.03 * r2), cv2.LINE_AA)
            self.mark("light_rays")
        a = self.active("speed_lines", t)
        if a:
            cx_, cy_ = W * 0.5, H * 0.5
            for i in range(int(40 + 40 * a)):
                ang = rng.random() * TAU
                r_in = (0.28 + 0.2 * rng.random()) * W
                r_out = r_in + (0.2 + 0.3 * rng.random()) * W
                p1 = (int(cx_ + math.cos(ang) * r_in), int(cy_ + math.sin(ang) * r_in * 0.75))
                p2 = (int(cx_ + math.cos(ang) * r_out), int(cy_ + math.sin(ang) * r_out * 0.75))
                cv2.line(L.over, p1, p2, (255, 255, 255, int(140 + 100 * rng.random())), 1 + int(2 * rng.random()), cv2.LINE_AA)
            self.mark("speed_lines")
        a = self.active("slash", t)
        if a and self.hit_at is not None:
            age = t - (self.fx["slash"].get("at") or self.hit_at)
            if 0 <= age < 0.28:
                p = min(1.0, age / 0.1)
                fade = 1.0 - max(0.0, (age - 0.1) / 0.18)
                ang = math.radians(-28 + 20 * ((self.seed % 5) - 2) / 2)
                cx_, cy_ = W * 0.5, H * 0.48
                R = W * 0.62
                pts = []
                for s in np.linspace(-1, -1 + 2 * p, 24):
                    px_ = cx_ + math.cos(ang) * R * s
                    py_ = cy_ + math.sin(ang) * R * s + 0.16 * H * (1 - s * s) * 0.0
                    pts.append((int(px_), int(py_)))
                for w_, c_ in ((L.px(0.05), (255, 220, 150)), (L.px(0.02), (255, 255, 255))):
                    cv2.polylines(L.add, [np.array(pts, np.int32)], False, tuple(int(c * fade) for c in c_), w_, cv2.LINE_AA)
                self.mark("slash")
        a = self.active("lightning", t)
        if a and self.hit_at is not None:
            age = t - (self.fx["lightning"].get("at") or self.hit_at)
            if 0 <= age < 0.2 and int(age * self.fps) % 2 == 0:
                lr = np.random.default_rng(self.seed + int(age * self.fps))
                x_, y_ = W * (0.3 + 0.4 * lr.random()), 0
                for _ in range(26):
                    nx_, ny_ = x_ + lr.normal(0, W * 0.02), y_ + H * 0.04
                    for wdt, c_ in ((L.px(0.012), (255, 160, 120)), (2, (255, 255, 255))):
                        cv2.line(L.add, (int(x_), int(y_)), (int(nx_), int(ny_)), c_, wdt, cv2.LINE_AA)
                    x_, y_ = nx_, ny_
                self.mark("lightning")
        a = self.active("swirl", t)
        if a:
            cx_, cy_ = self.to_px(self.subject["x"], self.subject["y"] + 0.12)
            for k_ in range(3):
                pts = []
                for s in np.linspace(0, 1, 40):
                    th = TAU * (1.5 * s + 0.3 * t + k_ / 3)
                    rr = (0.04 + 0.26 * s) * W
                    pts.append((int(cx_ + math.cos(th) * rr), int(cy_ - s * 0.5 * H + math.sin(th) * rr * 0.35)))
                cv2.polylines(L.add, [np.array(pts, np.int32)], False, tuple(int(c * 0.8 * a) for c in ac), L.px(0.008), cv2.LINE_AA)
            self.mark("swirl")
        a = self.active("lens_flare", t)
        if a:
            ox, oy = self.to_px(0.74, 0.14)
            cv2.circle(L.add, (int(ox), int(oy)), L.px(0.03), (200, 230, 255), -1, cv2.LINE_AA)
            cv2.line(L.add, (0, int(oy)), (W, int(oy)), (110, 85, 70), 2, cv2.LINE_AA)
            for k_, c_ in enumerate(((90, 140, 200), (130, 90, 160), (80, 160, 120))):
                gx, gy = W * 0.5 + (ox - W * 0.5) * (-0.5 - 0.4 * k_), H * 0.5 + (oy - H * 0.5) * (-0.5 - 0.4 * k_)
                cv2.circle(L.add, (int(gx), int(gy)), L.px(0.025 + 0.012 * k_), tuple(int(c * 0.35) for c in c_), 2, cv2.LINE_AA)
            self.mark("lens_flare")
        a = self.active("debris", t) or self.active("glass_shards", t)
        if a and self.hit_at is not None:
            kind = "debris" if "debris" in self.fx else "glass_shards"
            age = t - (self.fx[kind].get("at") or self.hit_at)
            if 0 <= age < 1.2:
                ox, oy = self.to_px(self.subject["x"], self.subject["y"] + 0.15)
                for i in range(26):
                    r1, r2, r3 = rng.random(3)
                    ang = -math.pi * (0.1 + 0.8 * r1)
                    sp = (0.2 + 0.7 * r2) * W
                    px_ = ox + math.cos(ang) * sp * age * (1 if r3 > .5 else -1)
                    py_ = oy + math.sin(ang) * sp * age + 0.9 * H * age ** 2
                    s_ = L.px(0.008 + 0.012 * r3)
                    tri = np.array([(px_, py_), (px_ + s_, py_ + s_ * 0.4), (px_ + s_ * 0.3, py_ + s_)], np.int32)
                    col = (235, 240, 255, 220) if kind == "glass_shards" else (70, 80, 95, 235)
                    cv2.fillConvexPoly(L.over, tri, col, cv2.LINE_AA)
                self.mark(kind)
        a = self.active("lens_drops", t)
        if a:
            lr = np.random.default_rng(self.seed + 91)
            for i in range(7):
                x, y = lr.random(), (lr.random() * 0.7 + (t * 0.01 * lr.random())) % 1.0
                cv2.circle(L.front, (int(x * W), int(y * H)), L.px(0.012 + 0.02 * lr.random()), (255, 255, 255, 70), -1, cv2.LINE_AA)
                cv2.circle(L.front, (int(x * W), int(y * H)), L.px(0.012), (255, 255, 255, 120), 2, cv2.LINE_AA)
            self.mark("lens_drops")
        a = self.active("pebbles", t)
        if a:
            lr = np.random.default_rng(self.seed + 5)
            for i in range(14):
                x, r = lr.random(), L.px(0.012 + 0.03 * lr.random())
                cv2.ellipse(L.front, (int(x * W), int(H * (0.97 + 0.03 * lr.random()))), (r * 2, r), 0, 0, 360, (38, 42, 48, 255), -1, cv2.LINE_AA)
            self.mark("pebbles")
        for k_ in ("smoke", "fire", "splash"):
            if k_ in self.fx and k_ not in self.ok:
                self.ok[k_] = "skipped:needs_blender_physics"

    def run(self, out_dir, resume=True):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        drawings = int(math.ceil(self.frames / self.step))
        for i in range(drawings):
            fp = out / f"f_{i + 1:04d}.png"
            f = 1 + i * self.step
            if resume and png_complete(fp):
                continue
            cv2.imwrite(str(fp), self.drawing(f), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        (out / "done.json").write_text(json.dumps({"drawings": drawings, "engine": "cv", "blender": None, "warnings": [],
                                                   "features": self.ok}, indent=1), "utf-8")
        return out


def render_shot(shot, settings=None):
    """-> frames_dir (Path). Same contract as the Blender scene script."""
    eng = Engine(shot, settings)
    out = eng.run(shot["paths"]["frames_dir"], resume=(settings or {}).get("resume", True))
    log.info("cv engine: shot %s done (%d drawings, features: %s)", shot.get("idx"), len(list(out.glob('f_*.png'))), sorted(eng.ok))
    return out
