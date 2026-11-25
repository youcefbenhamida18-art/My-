"""Runs INSIDE Blender (loaded by blender_scene.py) - billboard "quad clouds": particles and light effects as REAL 3D geometry.

Because they live in the 3D scene they get true parallax from the camera moves, depth of field / bokeh, occlusion by nearer
parts of the plate, and the same lens as the artwork.  They are invisible to GI / shadow / glossy rays, so they never light or
shade the plate.

    setup(ctx) -> [callable(t, f)]        (ctx is built by blender_scene.main)

* Only two meshes exist (one additive, one alpha blended); every emitter fills a slice of their quads each frame.
* Sprites come from one procedurally drawn atlas (no external files): dot, petal, leaf, streak, ring, star, mist, shard.
* Materials are unlit emission:  additive = Transparent + Emission,  alpha = Mix(Transparent, Emission).  Cycles and EEVEE.
* Every effect is a pure function of time (no stepped simulation), so a re-rendered shot is bit-identical and resumable.
"""
import math
import zlib

import bpy
import numpy as np

ASPECT = 16.0 / 9.0
SENSOR = 36.0
TILE, COLS, ROWS = 96, 4, 2
TILES = {"dot": 0, "petal": 1, "leaf": 2, "streak": 3, "ring": 4, "star": 5, "mist": 6, "shard": 7}
TAU = 2.0 * math.pi


# ------------------------------------------------------------------ small maths
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def wrap(x, lo, hi):
    return (x - lo) % (hi - lo) + lo


def to_linear(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ------------------------------------------------------------------ procedural sprite atlas
def _grid():
    ax = (np.arange(TILE, dtype=np.float32) + 0.5) / TILE * 2.0 - 1.0
    return np.meshgrid(ax, ax)                                          # X grows right, Y grows downward


def _noise(size, cells, seed):
    r = np.random.default_rng(seed).random((cells + 1, cells + 1)).astype(np.float32)
    xs = np.linspace(0, cells, size, dtype=np.float32)
    i0 = np.clip(np.floor(xs).astype(int), 0, cells - 1)
    f = xs - i0
    f = f * f * (3 - 2 * f)
    rows = r[i0, :] * (1 - f)[:, None] + r[i0 + 1, :] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


def _tri(X, Y, p1, p2, p3, soft=0.07):
    area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    sgn = 1.0 if area > 0 else -1.0
    d = None
    for a, b in ((p1, p2), (p2, p3), (p3, p1)):
        ln = math.hypot(b[0] - a[0], b[1] - a[1])
        e = sgn * ((X - a[0]) * (b[1] - a[1]) - (Y - a[1]) * (b[0] - a[0])) / ln
        d = e if d is None else np.minimum(d, e)
    return smooth(d / soft)


def sprite_tiles():
    """name -> (alpha, shade) float32 arrays (TILE, TILE); shade multiplies the per-quad colour (gives petals/leaves/shards volume)."""
    X, Y = _grid()
    R = np.sqrt(X * X + Y * Y)
    out = {"dot": (np.clip(1.0 - R, 0, 1) ** 1.8, np.ones_like(X))}
    e = np.sqrt((X / 0.46) ** 2 + (Y / 0.86) ** 2)
    notch = np.exp(-((X / 0.12) ** 2 + ((Y + 0.86) / 0.20) ** 2))
    out["petal"] = (smooth((1.0 - e) / 0.14) * (1.0 - 0.85 * notch),
                    np.clip(0.72 + 0.28 * (1 - np.abs(X) / 0.46) - 0.12 * np.clip(Y, 0, 1), 0.3, 1.0))
    w = 0.52 * np.clip(1.0 - Y * Y, 0, 1) ** 0.75
    veins = 0.5 + 0.5 * np.sin(Y * 22.0 + np.abs(X) * 14.0)
    out["leaf"] = (smooth((w - np.abs(X)) / 0.07) * (np.abs(Y) < 0.95),
                   np.clip(0.80 + 0.12 * veins * (np.abs(X) > 0.04) - 0.35 * np.exp(-((X / 0.035) ** 2)), 0.3, 1.0))
    out["streak"] = (np.exp(-((Y / 0.30) ** 2)) * np.clip(1.0 - np.abs(X) ** 2.4, 0, 1), np.ones_like(X))
    out["ring"] = (0.9 * np.exp(-(((R - 0.78) / 0.085) ** 2)) + 0.07 * np.clip(1 - R, 0, 1), np.ones_like(X))
    h = np.exp(-((Y / 0.05) ** 2)) * np.clip(1 - np.abs(X), 0, 1) ** 2
    v = np.exp(-((X / 0.05) ** 2)) * np.clip(1 - np.abs(Y), 0, 1) ** 2
    out["star"] = (np.clip(np.maximum(h, v) + np.exp(-((R / 0.18) ** 2)), 0, 1), np.ones_like(X))
    n = _noise(TILE, 6, 3) * 0.6 + _noise(TILE, 12, 4) * 0.4
    out["mist"] = (np.clip(n * 1.35 - 0.12, 0, 1) * np.clip(1 - R, 0, 1) ** 1.2, np.ones_like(X))
    out["shard"] = (_tri(X, Y, (0.0, -0.9), (0.62, 0.8), (-0.5, 0.7)), np.clip(0.55 + 0.45 * (X * 0.6 - Y * 0.4 + 0.5), 0.3, 1.0))
    return {k: (np.asarray(a, np.float32), np.asarray(s, np.float32)) for k, (a, s) in out.items()}


def make_atlas():
    buf = np.zeros((ROWS * TILE, COLS * TILE, 4), np.float32)
    for name, (a, s) in sprite_tiles().items():
        r, c = divmod(TILES[name], COLS)
        sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
        buf[sl + (slice(0, 3),)] = s[::-1][..., None]                    # image rows start at the bottom: flip so tile "up" is +v
        buf[sl + (3,)] = a[::-1]
    img = bpy.data.images.new("mf_sprites", COLS * TILE, ROWS * TILE, alpha=True)
    img.pixels.foreach_set(buf.ravel())
    try:
        img.colorspace_settings.name = "Non-Color"
        img.alpha_mode = "STRAIGHT"
    except Exception:  # noqa: BLE001
        pass
    img.update()
    return img


# ------------------------------------------------------------------ camera state (same maths as blender_scene.cam_matrix)
class CamState:
    def __init__(self, M, focal):
        A = np.array(M, dtype=np.float64)
        self.R, self.pos = A[:3, :3], A[:3, 3]
        self.right, self.up, self.fwd = self.R[:, 0], self.R[:, 1], -self.R[:, 2]
        self.focal = float(focal)

    def wl(self, L):
        """World width of the frame at distance L along the view axis."""
        return np.asarray(L, float) * SENSOR / self.focal

    def screen(self, u, v, L):
        """Point at screen coords (u frame widths right, v frame heights down) at distance L in front of the camera."""
        u, v, L = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(L, float))
        wl = L * SENSOR / self.focal
        return (self.pos + self.fwd * L[..., None] + self.right * (u * wl)[..., None] + self.up * (-v * wl / ASPECT)[..., None])


# ------------------------------------------------------------------ emitters
class E:
    """One effect. setup() fills self.n and self.tiles; step(t, f, cam) -> (centre (n,3), hw, hh, angle, rgb (n,3) 0-1, alpha)
    where hw/hh are WORLD half sizes of the quad and the colour is display (sRGB) colour."""
    blend = "alpha"

    def __init__(self, ctx, kind, fx):
        self.ctx, self.kind, self.fx = ctx, kind, fx
        self.a = float(fx.get("amount") or 0.6)
        self.at = float(fx.get("at") or 0.0)
        self.dur = fx.get("dur")
        self.rs = np.random.default_rng(int(ctx["seed"]) * 7919 + zlib.crc32(kind.encode()) % 100003)
        self.wind = float((ctx["shot"].get("wind") or [0.4, 0.05])[0])
        SUB, CX, CY, FW = ctx["SUB"], ctx["CX"], ctx["CY"], ctx["FW"]
        self.su, self.sv = (SUB["x"] - CX) / FW, (SUB["y"] - CY) / FW          # subject in frame units
        self.vg = (float(ctx["plate"].get("ground_y") or 0.85) - CY) / FW        # ground line in frame heights
        self.pal, self.WF0, self.HF0, self.D = ctx["pal"], ctx["WF0"], ctx["HF0"], ctx["D"]
        self.n, self.tiles = 0, np.zeros(0, int)
        self.setup()

    # helpers
    def count(self, lo, hi):
        return max(1, int(round(lo + (hi - lo) * min(1.0, max(0.0, self.a)))))

    def world(self, u, v, z):
        return self.ctx["s2w"](u, v, z)

    def sz(self, frac, z):
        return np.asarray(frac, float) * self.WF0 * np.asarray(z, float)

    def tile_ids(self, name, n=None):
        return np.full(self.n if n is None else n, TILES[name], int)

    def rgb(self, c):
        return np.tile(np.asarray(c, float)[None, :], (self.n, 1))

    def window(self, t, life):
        """-> (age 0..1, alive mask) for a one-shot event starting at self.at."""
        k = (t - self.at) / np.maximum(life, 1e-3)
        return k, (k >= 0) & (k <= 1)

    def setup(self):
        raise NotImplementedError

    def step(self, t, f, cam):
        raise NotImplementedError


class Faller(E):
    """Petals / leaves / snow: drift down with sway, wrap around the frame so the shot is always populated."""
    tile_name, size, spin_k, flutter, colors = "petal", (0.008, 0.017), 1.6, True, ((1.0, 0.72, 0.80),)
    count_range, fall, sway, zr, alpha = (40, 110), (0.06, 0.15), (0.01, 0.05), (0.58, 1.22), 0.95

    def setup(self):
        rs, n = self.rs, self.count(*self.count_range)
        self.n, self.tiles = n, self.tile_ids(self.tile_name, n)
        self.z = rs.uniform(*self.zr, n)
        self.u0, self.v0 = rs.uniform(-0.8, 0.8, n), rs.uniform(-0.75, 0.75, n)
        self.fall_s, self.sw, self.f1 = rs.uniform(*self.fall, n), rs.uniform(*self.sway, n), rs.uniform(0.25, 0.7, n)
        self.ph = rs.uniform(0, TAU, (n, 3))
        self.size_a = rs.uniform(*self.size, n)
        self.spin = rs.uniform(-self.spin_k, self.spin_k, n)
        self.flut = rs.uniform(1.2, 3.2, n)
        cols = np.asarray(self.colors, float)
        base = cols[rs.integers(0, len(cols), n)]
        self.col = np.clip(base * rs.uniform(0.86, 1.06, (n, 1)), 0, 1)

    def step(self, t, f, cam):
        u = wrap(self.u0 + self.wind * 0.085 * t * (1.3 - 0.5 * self.z) + self.sw * np.sin(TAU * self.f1 * t + self.ph[:, 0]), -0.8, 0.8)
        v = wrap(self.v0 + self.fall_s * t, -0.75, 0.75)
        edge = smooth(np.minimum(v + 0.75, 0.75 - v) / 0.12) * smooth(np.minimum(u + 0.8, 0.8 - u) / 0.12)
        fl = (0.35 + 0.65 * np.abs(np.cos(TAU * 0.18 * self.flut * t + self.ph[:, 1]))) if self.flutter else 1.0
        z = self.z
        return (self.world(u, v, z), self.sz(self.size_a * fl, z), self.sz(self.size_a * 1.25, z), self.ph[:, 2] + self.spin * t,
                self.col, self.alpha * edge)


class Petals(Faller):
    pass


class Leaves(Faller):
    tile_name, size, spin_k = "leaf", (0.012, 0.026), 1.2
    colors = ((0.86, 0.46, 0.12), (0.74, 0.27, 0.10), (0.58, 0.52, 0.12), (0.80, 0.62, 0.15))
    count_range, fall, sway = (22, 60), (0.05, 0.11), (0.03, 0.09)


class Snow(Faller):
    tile_name, size, spin_k, flutter, colors = "dot", (0.0035, 0.009), 0.0, False, ((0.96, 0.98, 1.0),)
    count_range, fall, sway, zr, alpha = (90, 250), (0.05, 0.11), (0.006, 0.03), (0.56, 1.25), 0.85

    def setup(self):
        super().setup()
        near = self.z < 0.7                                           # a few big soft flakes near the lens -> bokeh when DoF is on
        self.size_a = np.where(near, self.size_a * 3.2, self.size_a)


class Dust(E):
    def setup(self):
        rs, n = self.rs, self.count(50, 140)
        self.n, self.tiles = n, self.tile_ids("dot", n)
        self.z, self.u0, self.v0 = rs.uniform(0.6, 1.2, n), rs.uniform(-0.8, 0.8, n), rs.uniform(-0.7, 0.7, n)
        self.size_a, self.ph, self.vel = rs.uniform(0.0025, 0.0065, n), rs.uniform(0, TAU, (n, 3)), rs.uniform(-0.012, 0.012, (n, 2))
        top = np.clip(np.asarray(self.pal("top", (1.0, 0.95, 0.85)), float), 0, 1)
        self.col = np.tile(np.clip(0.55 * top + 0.45 * np.array([1.0, 0.95, 0.82]), 0, 1)[None, :], (n, 1))

    def step(self, t, f, cam):
        u = wrap(self.u0 + self.vel[:, 0] * t + 0.012 * np.sin(0.6 * t + self.ph[:, 0]), -0.8, 0.8)
        v = wrap(self.v0 + self.vel[:, 1] * t - 0.006 * t + 0.01 * np.sin(0.5 * t + self.ph[:, 1]), -0.7, 0.7)
        tw = 0.25 + 0.4 * (0.5 + 0.5 * np.sin(TAU * 0.4 * t + self.ph[:, 2]))
        return self.world(u, v, self.z), self.sz(self.size_a, self.z), self.sz(self.size_a, self.z), self.ph[:, 2], self.col, tw * (0.5 + 0.5 * self.a)


class Rain(E):
    def setup(self):
        rs, n = self.rs, self.count(140, 380)
        self.n, self.tiles = n, self.tile_ids("streak", n)
        self.z, self.u0, self.v0 = rs.uniform(0.6, 1.25, n), rs.uniform(-0.85, 0.85, n), rs.uniform(-0.8, 0.8, n)
        self.speed, self.len_, self.alpha = rs.uniform(1.3, 2.0, n), rs.uniform(0.03, 0.065, n), rs.uniform(0.22, 0.5, n)
        self.col = np.tile(np.array([0.82, 0.9, 1.0])[None, :], (n, 1))
        self.dir = np.array([np.clip(self.wind, -1, 1) * 0.35, -1.0])           # (right, up) in screen space; falls down and slightly sideways
        self.ang = math.atan2(self.dir[1], self.dir[0])

    def step(self, t, f, cam):
        d = self.dir / np.hypot(*self.dir)
        s = self.speed * t
        u = wrap(self.u0 + d[0] * s, -0.85, 0.85)
        v = wrap(self.v0 - d[1] * s, -0.8, 0.8)
        z = self.z
        return self.world(u, v, z), self.sz(self.len_, z), self.sz(0.0034, z), np.full(self.n, self.ang), self.col, self.alpha * (0.6 + 0.4 * self.a)


class Embers(E):
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(40, 130)
        self.n, self.tiles = n, np.where(rs.random(n) < 0.35, TILES["star"], TILES["dot"])
        self.z, self.u0 = rs.uniform(0.62, 1.15, n), rs.uniform(-0.7, 0.7, n)
        self.life, self.phase = rs.uniform(2.2, 4.4, n), rs.random(n)
        self.rise, self.size_a, self.ph = rs.uniform(0.18, 0.42, n), rs.uniform(0.003, 0.009, n), rs.uniform(0, TAU, (n, 2))

    def step(self, t, f, cam):
        age = (t / self.life + self.phase) % 1.0
        v = 0.62 - self.rise * age * self.life * 0.55
        u = self.u0 + 0.05 * np.sin(TAU * 0.45 * t + self.ph[:, 0]) + self.wind * 0.04 * age * self.life
        a = np.sin(np.pi * age) ** 0.7 * (0.65 + 0.35 * np.sin(TAU * 4.5 * t + self.ph[:, 1]))
        k = age[:, None]
        col = (1.0 - k) * np.array([1.0, 0.82, 0.32]) + k * np.array([1.0, 0.24, 0.05])
        return self.world(u, v, self.z), self.sz(self.size_a * (1.2 - 0.6 * age), self.z), self.sz(self.size_a * (1.2 - 0.6 * age), self.z), \
            self.ph[:, 0], col, np.clip(a, 0, 1) * (0.7 + 0.3 * self.a) * 1.6


class Fireflies(E):
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(12, 36)
        self.n, self.tiles = n, self.tile_ids("dot", n)
        self.z, self.u0, self.v0 = rs.uniform(0.62, 1.15, n), rs.uniform(-0.55, 0.55, n), rs.uniform(-0.2, 0.45, n)
        self.ph, self.f = rs.uniform(0, TAU, (n, 3)), rs.uniform(0.25, 0.8, (n, 3))
        self.size_a, self.pf = rs.uniform(0.007, 0.014, n), rs.uniform(0.25, 0.6, n)
        self.col = np.tile(np.array([0.85, 1.0, 0.35])[None, :], (n, 1)) * rs.uniform(0.85, 1.0, (n, 1))

    def step(self, t, f, cam):
        u = self.u0 + 0.08 * np.sin(self.f[:, 0] * t + self.ph[:, 0]) + 0.04 * np.sin(self.f[:, 1] * 2.3 * t + self.ph[:, 1])
        v = self.v0 + 0.05 * np.sin(self.f[:, 2] * t + self.ph[:, 2]) + 0.03 * np.sin(self.f[:, 0] * 1.7 * t + self.ph[:, 0])
        pulse = np.clip(np.sin(TAU * self.pf * t + self.ph[:, 1]), 0, 1) ** 2
        return self.world(u, v, self.z), self.sz(self.size_a, self.z), self.sz(self.size_a, self.z), self.ph[:, 2], self.col, pulse * 1.4


class Sparks(E):
    """Burst at `at`: ballistic streaks that cool from white-yellow to red."""
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(60, 180)
        self.n, self.tiles = n, self.tile_ids("streak", n)
        th = rs.uniform(0, TAU, n)
        up = rs.random(n) < 0.55
        th = np.where(up, rs.uniform(math.radians(35), math.radians(145), n), th)           # biased upward
        sp = rs.uniform(0.5, 1.7, n)
        self.vel = np.stack([np.cos(th) * sp, -np.sin(th) * sp], 1)                           # (u, v down) frame units / s
        self.life, self.z = rs.uniform(0.35, 0.95, n), rs.uniform(0.85, 1.0, n)
        self.size_a = rs.uniform(0.012, 0.03, n)

    def step(self, t, f, cam):
        dt = t - self.at
        if dt < 0:
            return None
        g = 1.5
        u = self.su + self.vel[:, 0] * dt
        v = self.sv + 0.02 + self.vel[:, 1] * dt + 0.5 * g * dt * dt
        vv = self.vel[:, 1] + g * dt
        ang = np.arctan2(-vv, self.vel[:, 0])                                                 # screen up is +y of the billboard
        k = np.clip(dt / self.life, 0, 1)
        a = (1.0 - k) ** 1.5
        col = (1.0 - k)[:, None] * np.array([1.0, 0.92, 0.55]) + k[:, None] * np.array([1.0, 0.42, 0.1])
        sp = np.hypot(self.vel[:, 0], vv)
        return self.world(u, v, self.z), self.sz(self.size_a * (0.6 + 0.5 * sp), self.z), self.sz(self.size_a * 0.22, self.z), ang, col, a * 2.2


class Mist(E):
    def setup(self):
        rs, n = self.rs, self.count(7, 18)
        self.n, self.tiles = n, self.tile_ids("mist", n)
        self.z, self.u0 = rs.uniform(0.72, 1.25, n), rs.uniform(-0.8, 0.8, n)
        self.v0 = np.clip(self.vg - 0.06, -0.15, 0.42) + rs.uniform(-0.12, 0.10, n)
        self.size_a, self.ph, self.sp = rs.uniform(0.25, 0.5, n), rs.uniform(0, TAU, n), rs.uniform(0.012, 0.03, n)
        avg = np.clip(np.asarray(self.pal("avg", (0.5, 0.5, 0.55)), float), 0, 1)
        self.col = np.tile(np.clip(0.45 * avg + 0.55 * np.array([0.88, 0.9, 0.95]), 0, 1)[None, :], (n, 1))
        self.base = rs.uniform(0.10, 0.22, n) * (0.7 + 0.6 * self.a)

    def step(self, t, f, cam):
        u = wrap(self.u0 + self.sp * t * (1.0 if self.wind >= 0 else -1.0), -0.95, 0.95)
        v = self.v0 + 0.012 * np.sin(0.5 * t + self.ph)
        edge = smooth(np.minimum(u + 0.95, 0.95 - u) / 0.3)
        return self.world(u, v, self.z), self.sz(self.size_a, self.z), self.sz(self.size_a * 0.42, self.z), self.ph * 0.05, self.col, self.base * edge


class Smoke(Mist):
    """Sprite smoke (the Mantaflow version in blender_extras replaces it when PHYSICS=1 works)."""
    def setup(self):
        super().setup()
        rs, n = self.rs, self.n
        self.rise, self.size_a = rs.uniform(0.05, 0.12, n), rs.uniform(0.07, 0.16, n)
        self.u0 = self.su + rs.uniform(-0.12, 0.12, n)
        self.col = np.tile(np.array([0.30, 0.30, 0.33])[None, :], (n, 1)) * rs.uniform(0.7, 1.2, (n, 1))
        self.life = rs.uniform(2.0, 3.6, n)
        self.phase = rs.random(n)

    def step(self, t, f, cam):
        age = (t / self.life + self.phase) % 1.0
        u = self.u0 + self.wind * 0.05 * age * self.life + 0.03 * np.sin(0.7 * t + self.ph)
        v = (self.vg - 0.04) - self.rise * age * self.life
        s = self.size_a * (0.5 + 1.8 * age)
        return self.world(u, v, self.z), self.sz(s, self.z), self.sz(s * 0.95, self.z), self.ph + age, self.col, np.sin(np.pi * age) * 0.42 * (0.6 + 0.8 * self.a)


class Fire(E):
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(30, 70)
        self.n, self.tiles = n, np.where(rs.random(n) < 0.3, TILES["star"], TILES["mist"])
        self.z, self.phase, self.life = rs.uniform(0.9, 1.0, n), rs.random(n), rs.uniform(0.5, 1.1, n)
        self.u0, self.size_a, self.ph = self.su + rs.uniform(-0.09, 0.09, n), rs.uniform(0.02, 0.05, n), rs.uniform(0, TAU, n)

    def step(self, t, f, cam):
        age = (t / self.life + self.phase) % 1.0
        u = self.u0 + 0.025 * np.sin(TAU * 1.6 * t + self.ph) * age
        v = (self.vg - 0.02) - 0.34 * age * (0.6 + 0.4 * self.a)
        s = self.size_a * (1.0 - 0.7 * age)
        k = age[:, None]
        col = (1 - k) * np.array([1.0, 0.78, 0.25]) + k * np.array([0.9, 0.18, 0.04])
        return self.world(u, v, self.z), self.sz(s, self.z), self.sz(s * 1.5, self.z), np.zeros(self.n), col, np.sin(np.pi * np.clip(age, 0, 1)) ** 0.6 * 1.5


class VolumeFog(Mist):
    """Big slow fog banks (the Principled-Volume version in blender_extras replaces it when VOLUMETRICS=1 works)."""
    def setup(self):
        super().setup()
        self.size_a, self.base = self.rs.uniform(0.45, 0.8, self.n), self.rs.uniform(0.07, 0.15, self.n) * (0.7 + 0.6 * self.a)


class LightRays(E):
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(6, 14)
        self.n, self.tiles = n, self.tile_ids("streak", n)
        side = -1.0 if self.wind > 0 else 1.0                                                    # light comes from the windward-opposite top corner
        self.src = np.stack([side * rs.uniform(0.1, 0.45, n), np.full(n, -0.62)], 1)
        self.dirs = np.radians(rs.uniform(62, 112, n))                                          # downward fan (screen angle)
        self.width, self.base, self.ph = rs.uniform(0.025, 0.09, n), rs.uniform(0.04, 0.11, n), rs.uniform(0, TAU, n)
        top = np.clip(np.asarray(self.pal("top", (1.0, 0.93, 0.78)), float), 0, 1)
        self.col = np.tile(np.clip(0.4 * top + 0.6 * np.array([1.0, 0.93, 0.75]), 0, 1)[None, :], (n, 1))

    def step(self, t, f, cam):
        L = self.D * 0.55
        c = np.stack([np.cos(self.dirs), np.sin(self.dirs)], 1)                                  # screen dir: x right, y DOWN
        half = 0.55
        u = self.src[:, 0] + c[:, 0] * half
        v = self.src[:, 1] + c[:, 1] * half
        a = self.base * (0.7 + 0.3 * np.sin(0.45 * t + self.ph)) * (0.7 + 0.6 * self.a)
        wl = float(cam.wl(L))
        return cam.screen(u, v, np.full(self.n, L)), np.full(self.n, half * wl), self.width * wl * 0.5 * 2.2, -self.dirs, self.col, a * 1.5


class SpeedLines(E):
    """Manga focus lines: radial streaks around the subject that re-roll on every drawing (hand-drawn flicker)."""
    def setup(self):
        self.n = self.count(36, 110)
        self.tiles = self.tile_ids("streak")
        self.step_n = max(1, int(self.ctx["shot"].get("step", 2)))

    def step(self, t, f, cam):
        k = (int(f) - 1) // self.step_n
        rs = np.random.default_rng(self.ctx["seed"] * 31 + k * 7919)
        n = self.n
        th = rs.uniform(0, TAU, n)
        cs, sn = np.cos(th), np.sin(th)
        # distance from the centre to the frame edge along each direction (frame widths; the frame is 1 x 1/ASPECT)
        edge = np.minimum(0.5 / np.maximum(np.abs(cs), 1e-3), (0.5 / ASPECT) / np.maximum(np.abs(sn), 1e-3))
        r_in = edge * rs.uniform(0.42, 0.80, n)
        ln = (edge - r_in) * rs.uniform(0.7, 1.35, n)
        rc = r_in + ln * 0.5
        ox, oy = self.su * 0.4, self.sv * 0.4
        u, v = ox + cs * rc, oy - sn * rc * ASPECT                  # physical (circle-true) layout -> frame units (v points down)
        L = self.D * 0.6
        wl = float(cam.wl(L))
        a = rs.uniform(0.5, 0.95, n) * (0.5 + 0.7 * self.a) * (1.0 if self.at <= 0 else float(smooth((t - self.at + 0.05) / 0.05)))
        return cam.screen(u, v, np.full(n, L)), ln * 0.5 * wl, rs.uniform(0.0016, 0.006, n) * wl * 2.2, th, \
            np.ones((n, 3)), a


class Shockwave(E):
    blend = "add"

    def setup(self):
        self.n = 3
        self.tiles = self.tile_ids("ring")

    def step(self, t, f, cam):
        out = []
        for i in range(self.n):
            dt = t - self.at - 0.06 * i
            life = 0.55 + 0.1 * i
            out.append((dt, life))
        dts = np.array([o[0] for o in out])
        life = np.array([o[1] for o in out])
        k = np.clip(dts / life, 0, 1)
        grow = 1.0 - (1.0 - k) ** 3
        size = (0.04 + (0.85 - 0.18 * np.arange(3)) * grow)
        a = np.where(dts < 0, 0.0, (1.0 - k) ** 1.2) * np.array([1.0, 0.7, 0.45])
        L = self.D * 0.8
        wl = float(cam.wl(L))
        ctr = cam.screen(np.full(3, self.su), np.full(3, self.sv), np.full(3, L))
        col = np.tile(self.pal("accent", (1.0, 0.85, 0.6)), (3, 1)) * 0.5 + 0.5
        return ctr, size * wl * 0.5, size * wl * 0.5, np.zeros(3), np.clip(col, 0, 1), a * 1.3


class Debris(E):
    def setup(self):
        rs, n = self.rs, self.count(16, 48)
        self.n, self.tiles = n, self.tile_ids("shard", n)
        th = rs.uniform(math.radians(20), math.radians(160), n)
        sp = rs.uniform(0.3, 1.1, n)
        self.vel = np.stack([np.cos(th) * sp * rs.choice([-1, 1], n), -np.sin(th) * sp], 1)
        self.z = np.where(rs.random(n) < 0.25, rs.uniform(0.55, 0.7, n), rs.uniform(0.85, 1.0, n))
        self.size_a, self.spin, self.ph = rs.uniform(0.006, 0.026, n), rs.uniform(-6, 6, n), rs.uniform(0, TAU, (n, 2))
        g = np.clip(np.asarray(self.pal("bottom", (0.35, 0.3, 0.25)), float), 0, 1)
        self.col = np.clip(g[None, :] * rs.uniform(0.45, 0.95, (n, 1)) + 0.04, 0, 1)

    def step(self, t, f, cam):
        dt = t - self.at
        if dt < 0:
            return None
        u = self.su + self.vel[:, 0] * dt
        v = self.sv + 0.06 + self.vel[:, 1] * dt + 0.5 * 1.3 * dt * dt
        a = 1.0 - smooth((dt - 1.1) / 0.5)
        fl = 0.4 + 0.6 * np.abs(np.cos(self.spin * dt + self.ph[:, 0]))
        return self.world(u, v, self.z), self.sz(self.size_a * fl, self.z), self.sz(self.size_a, self.z), self.ph[:, 1] + self.spin * dt, self.col, a


class GlassShards(E):
    """Slow-motion glass: shards tumble and flash when a facet catches the light."""
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(22, 60)
        self.n, self.tiles = n, self.tile_ids("shard", n)
        th = rs.uniform(0, TAU, n)
        sp = rs.uniform(0.12, 0.5, n)
        self.vel = np.stack([np.cos(th) * sp, np.sin(th) * sp * 0.8], 1)
        self.z, self.size_a = rs.uniform(0.7, 1.0, n), rs.uniform(0.008, 0.03, n)
        self.spin, self.ph = rs.uniform(-3.5, 3.5, n), rs.uniform(0, TAU, (n, 2))
        self.col = np.tile(np.array([0.72, 0.92, 1.0])[None, :], (n, 1)) * rs.uniform(0.8, 1.0, (n, 1))

    def step(self, t, f, cam):
        dt = t - self.at
        if dt < 0:
            return None
        u = self.su + self.vel[:, 0] * dt
        v = self.sv + self.vel[:, 1] * dt + 0.5 * 0.25 * dt * dt
        fl = np.abs(np.cos(self.spin * dt * 0.7 + self.ph[:, 0]))
        glint = fl ** 8
        a = (0.12 + 1.5 * glint) * (1.0 - smooth((dt - 1.6) / 0.6))
        return self.world(u, v, self.z), self.sz(self.size_a * (0.3 + 0.7 * fl), self.z), self.sz(self.size_a, self.z), \
            self.ph[:, 1] + self.spin * dt, self.col, a


class LensDrops(E):
    def setup(self):
        rs, n = self.rs, self.count(6, 16)
        self.n, self.tiles = n, self.tile_ids("ring", n)
        self.u0, self.v0 = rs.uniform(-0.45, 0.45, n), rs.uniform(-0.25, 0.3, n)
        self.size_a, self.slide, self.ph = rs.uniform(0.014, 0.04, n), rs.uniform(0.002, 0.012, n), rs.uniform(0, TAU, n)

    def step(self, t, f, cam):
        L = self.D * 0.28
        wl = float(cam.wl(L))
        v = self.v0 + self.slide * t
        a = (0.30 + 0.12 * np.sin(0.7 * t + self.ph)) * (0.5 + 0.8 * self.a)
        return cam.screen(self.u0, v, np.full(self.n, L)), self.size_a * wl, self.size_a * wl, np.zeros(self.n), \
            np.tile(np.array([0.9, 0.95, 1.0])[None, :], (self.n, 1)), a


class LensFlare(E):
    blend = "add"

    def setup(self):
        self.n = 6
        self.tiles = np.array([TILES["star"], TILES["streak"], TILES["ring"], TILES["ring"], TILES["dot"], TILES["dot"]])
        self.side = -1.0 if self.wind > 0 else 1.0

    def step(self, t, f, cam):
        lu, lv = self.side * 0.34, -0.2
        L = self.D * 0.5
        wl = float(cam.wl(L))
        pulse = (0.7 + 0.3 * math.sin(TAU * 0.35 * t)) * (0.5 + 0.7 * self.a)
        pts = [(lu, lv), (lu, lv), (-0.35 * lu, -0.35 * lv), (-0.8 * lu, -0.8 * lv), (0.55 * lu, 0.55 * lv), (-1.3 * lu, -1.3 * lv)]
        size = np.array([0.10, 0.55, 0.05, 0.09, 0.03, 0.04]) * wl
        hh = size.copy()
        hh[1] = 0.006 * wl * 2.2
        cols = np.array([[1.0, 0.95, 0.8], [0.5, 0.7, 1.0], [0.6, 1.0, 0.8], [1.0, 0.6, 0.5], [1.0, 0.9, 0.6], [0.6, 0.7, 1.0]])
        a = np.array([1.0, 0.8, 0.5, 0.35, 0.5, 0.4]) * pulse
        ctr = cam.screen(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), np.full(6, L))
        return ctr, size, hh, np.zeros(6), cols, a


class Aura(E):
    """Wisps of energy rising around the subject (additive, tinted by the palette accent)."""
    blend = "add"

    def setup(self):
        rs, n = self.rs, self.count(36, 100)
        self.n, self.tiles = n, np.where(rs.random(n) < 0.5, TILES["streak"], TILES["mist"])
        self.u0, self.phase, self.life = rs.uniform(-0.13, 0.13, n), rs.random(n), rs.uniform(1.1, 2.2, n)
        self.size_a, self.ph, self.z = rs.uniform(0.012, 0.03, n), rs.uniform(0, TAU, n), rs.uniform(0.95, 1.0, n)
        acc = np.clip(np.asarray(self.pal("accent", (0.5, 0.8, 1.0)), float), 0, 1)
        self.col = np.tile(np.clip(0.75 * acc + 0.25, 0, 1)[None, :], (n, 1))

    def step(self, t, f, cam):
        age = (t / self.life + self.phase) % 1.0
        u = self.su + self.u0 * (1.0 + 0.5 * age) + 0.02 * np.sin(TAU * 0.8 * t + self.ph)
        v = self.sv + 0.22 - 0.5 * age
        s = self.size_a * (1.0 - 0.4 * age)
        return self.world(u, v, self.z), self.sz(s * 1.0, self.z), self.sz(s * 3.2, self.z), np.full(self.n, math.pi / 2) + 0.15 * np.sin(self.ph + t), \
            self.col, np.sin(np.pi * age) ** 0.8 * (0.5 + 0.5 * self.a) * 1.4


class Splash(E):
    """Droplets + ripple ring where something hits the water/ground (ballistic sprites, no fluid sim)."""
    def setup(self):
        rs, n = self.rs, self.count(30, 80)
        self.n, self.tiles = n + 2, np.concatenate([self.tile_ids("dot", n), self.tile_ids("ring", 2)])
        self.nd = n
        th = rs.uniform(math.radians(40), math.radians(140), n)
        sp = rs.uniform(0.35, 1.0, n)
        self.vel = np.stack([np.cos(th) * sp * rs.choice([-1, 1], n) * 0.6, -np.sin(th) * sp], 1)
        self.size_a, self.z = rs.uniform(0.003, 0.009, n), rs.uniform(0.88, 1.0, n)

    def step(self, t, f, cam):
        dt = t - self.at
        n = self.nd
        ctr = np.zeros((n + 2, 3))
        hw, hh, ang, col, a = np.zeros(n + 2), np.zeros(n + 2), np.zeros(n + 2), np.ones((n + 2, 3)), np.zeros(n + 2)
        if dt >= 0:
            u = self.su + self.vel[:, 0] * dt
            v = self.vg + self.vel[:, 1] * dt + 0.5 * 1.8 * dt * dt
            ctr[:n] = self.world(u, v, self.z)
            hw[:n], hh[:n] = self.sz(self.size_a, self.z), self.sz(self.size_a * 1.4, self.z)
            col[:n] = np.array([0.8, 0.92, 1.0])
            a[:n] = np.where(v > self.vg + 0.02, 0.0, 0.85) * (1.0 - smooth((dt - 0.8) / 0.4))
            for i in range(2):
                k = np.clip((dt - 0.08 * i) / 0.9, 0, 1)
                ctr[n + i] = self.world(self.su, self.vg + 0.01, 0.97)
                hw[n + i] = self.sz(0.03 + 0.2 * k, 0.97)
                hh[n + i] = hw[n + i] * 0.32
                a[n + i] = (1.0 - k) ** 1.3 * 0.8 if dt >= 0.08 * i else 0.0
            col[n:] = np.array([0.85, 0.95, 1.0])
        return ctr, hw, hh, ang, col, a


EMITTERS = {"petals": Petals, "leaves": Leaves, "snow": Snow, "dust": Dust, "rain": Rain, "embers": Embers, "fireflies": Fireflies,
            "sparks": Sparks, "mist": Mist, "smoke": Smoke, "fire": Fire, "volume_fog": VolumeFog, "light_rays": LightRays,
            "speed_lines": SpeedLines, "shockwave": Shockwave, "debris": Debris, "glass_shards": GlassShards, "lens_drops": LensDrops,
            "lens_flare": LensFlare, "aura": Aura, "splash": Splash}
# handled elsewhere: wind_sway/hair_sway/heat_haze/water_ripple/glow_pulse (plate material), grass/pebbles (blender_extras),
# slash/lightning/swirl (curve strokes in blender_extras), cloth_flag (blender_extras, physics tier)


# ------------------------------------------------------------------ the two meshes
def sprite_material(ctx, name, atlas, additive):
    m = ctx["new_mat"](name)
    n, l = m.node_tree.nodes, m.node_tree.links
    uv = n.new("ShaderNodeTexCoord")
    tex = n.new("ShaderNodeTexImage")
    tex.image, tex.interpolation, tex.extension = atlas, "Linear", "CLIP"
    l.new(uv.outputs["UV"], tex.inputs["Vector"])
    col = n.new("ShaderNodeAttribute")
    col.attribute_name = "Col"
    mul_c = n.new("ShaderNodeVectorMath")
    mul_c.operation = "MULTIPLY"
    l.new(tex.outputs["Color"], mul_c.inputs[0])
    l.new(col.outputs["Color"], mul_c.inputs[1])
    mul_a = n.new("ShaderNodeMath")
    mul_a.operation = "MULTIPLY"
    l.new(tex.outputs["Alpha"], mul_a.inputs[0])
    l.new(col.outputs["Alpha"], mul_a.inputs[1])
    em = n.new("ShaderNodeEmission")
    l.new(mul_c.outputs["Vector"], em.inputs["Color"])
    tr = n.new("ShaderNodeBsdfTransparent")
    out = n.new("ShaderNodeOutputMaterial")
    if additive:
        k = n.new("ShaderNodeMath")
        k.operation = "MULTIPLY"
        l.new(mul_a.outputs[0], k.inputs[0])
        k.inputs[1].default_value = 2.2                                                   # >1: bright enough to bloom in the compositor
        l.new(k.outputs[0], em.inputs["Strength"])
        add = n.new("ShaderNodeAddShader")
        l.new(tr.outputs["BSDF"], add.inputs[0])
        l.new(em.outputs["Emission"], add.inputs[1])
        l.new(add.outputs["Shader"], out.inputs["Surface"])
    else:
        em.inputs["Strength"].default_value = 1.0
        mix = n.new("ShaderNodeMixShader")
        l.new(mul_a.outputs[0], mix.inputs[0])
        l.new(tr.outputs["BSDF"], mix.inputs[1])
        l.new(em.outputs["Emission"], mix.inputs[2])
        l.new(mix.outputs["Shader"], out.inputs["Surface"])
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND"), ("show_transparent_back", False)):
        try:
            setattr(m, attr, val)
        except Exception:  # noqa: BLE001
            pass
    return m


class Pool:
    CORN = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    CUV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)

    def __init__(self, ctx, name, additive, emitters, atlas):
        self.ctx, self.name, self.em = ctx, name, emitters
        self.n = int(sum(e.n for e in emitters))
        n = self.n
        tiles = np.concatenate([np.asarray(e.tiles, int) for e in emitters])
        c, r = tiles % COLS, tiles // COLS
        uv = np.zeros((n, 4, 2))
        for k in range(4):
            uv[:, k, 0] = (c + 0.02 + 0.96 * self.CUV[k, 0]) / COLS
            uv[:, k, 1] = (r + 0.02 + 0.96 * self.CUV[k, 1]) / ROWS
        me = bpy.data.meshes.new(name)
        me.from_pydata(np.zeros((4 * n, 3)).tolist(), [], np.arange(4 * n).reshape(n, 4).tolist())
        uvl = me.uv_layers.new(name="UV")
        uvl.data.foreach_set("uv", uv.reshape(-1, 2).astype(np.float32).ravel())
        self.ca = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
        me.update()
        self.me = me
        self.ob = bpy.data.objects.new(name, me)
        ctx["scene"].collection.objects.link(self.ob)
        me.materials.append(sprite_material(ctx, name + "_mat", atlas, additive))
        for attr in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter", "visible_shadow"):
            try:
                setattr(self.ob, attr, False)
            except Exception:  # noqa: BLE001
                pass
        self.failed = set()

    def update(self, t, f):
        ctx, n = self.ctx, self.n
        M, p = ctx["cam_matrix"](f)
        cam = CamState(M, ctx["FOCAL"] * p["focal_scale"])
        ctr, hw, hh, ang = np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n)
        rgb, al = np.ones((n, 3)), np.zeros(n)
        i = 0
        for e in self.em:
            sl = slice(i, i + e.n)
            i += e.n
            if e.kind in self.failed or e.kind in ctx.get("fx_disabled", ()):    # blender_extras may replace sprites by a real sim
                continue
            try:
                r = e.step(t, f, cam)
            except Exception as ex:  # noqa: BLE001
                self.failed.add(e.kind)
                ctx["mark"](e.kind, f"failed:{type(ex).__name__}: {str(ex)[:120]}")
                ctx["say"]("fx failed", e.kind, ex)
                continue
            if r is None:
                continue
            ctr[sl], hw[sl], hh[sl], ang[sl], rgb[sl], al[sl] = r
        cs, sn = np.cos(ang), np.sin(ang)
        px = self.CORN[None, :, 0] * hw[:, None] * cs[:, None] - self.CORN[None, :, 1] * hh[:, None] * sn[:, None]
        py = self.CORN[None, :, 0] * hw[:, None] * sn[:, None] + self.CORN[None, :, 1] * hh[:, None] * cs[:, None]
        verts = ctr[:, None, :] + px[..., None] * cam.right + py[..., None] * cam.up
        self.me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        cols = np.concatenate([to_linear(rgb), np.clip(al, 0, 1)[:, None]], 1)
        self.ca.data.foreach_set("color", np.repeat(cols, 4, axis=0).astype(np.float32).ravel())
        self.me.update()


def setup(ctx):
    """-> [update(t, f)]  (empty list when the shot asks for no billboard effect)."""
    FXS, mark = ctx["FXS"], ctx["mark"]
    add, alpha = [], []
    for kind, fx in FXS.items():
        cls = EMITTERS.get(kind)
        if cls is None:
            continue
        try:
            e = cls(ctx, kind, fx)
        except Exception as ex:  # noqa: BLE001
            mark(kind, f"failed:{type(ex).__name__}: {str(ex)[:120]}")
            ctx["say"]("fx setup failed", kind, ex)
            continue
        (add if e.blend == "add" else alpha).append(e)
        mark(kind, "ok")
    if not add and not alpha:
        mark("fx_billboards", "off")
        return []
    atlas = make_atlas()
    pools = []
    if alpha:
        pools.append(Pool(ctx, "fx_alpha", False, alpha, atlas))
    if add:
        pools.append(Pool(ctx, "fx_add", True, add, atlas))
    mark("fx_billboards", "ok")
    mark("sprite_atlas_procedural")

    def update(t, f):
        for p in pools:
            p.update(t, f)
    update.__name__ = "fx_pools"
    return [update]
