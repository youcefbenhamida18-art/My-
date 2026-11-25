"""Eye rig for the cv engine: composites the per-eye layers made by eyes.py onto the PLATE (before the depth warp, so every
drawing keeps parallax/DoF/motion blur) following the shot's eye plan (eye_anim.py):

    gaze drift + pupil breathing  (local backward remap inside the iris)       blink (3 closed-lid drawings, on twos)
    catch-light twinkle (additive)  glow (additive, demon / power eyes)        tears (welling + running streak)
"""
from __future__ import annotations

import hashlib
import math
import pathlib

import cv2
import numpy as np

from .. import eye_anim


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _rgba(path, s):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    if abs(s - 1.0) > 1e-3:
        im = cv2.resize(im, (max(2, int(round(im.shape[1] * s))), max(2, int(round(im.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    f = im.astype(np.float32)
    return f[..., :3], f[..., 3:4] / 255.0


class Eye:
    def __init__(self, e, s, plate_hw):
        L = e["layers"]
        self.face, self.side, self.s = e["face"], e["side"], s
        ox, oy = L["origin"]
        wc, hc = L["size"]
        self.x0, self.y0 = int(round(ox * s)), int(round(oy * s))
        self.closed = [_rgba(p, s) for p in L["closed"]]
        self.shine, self.glow = _rgba(L["shine"], s), _rgba(L["glow"], s)
        ref = self.shine or self.glow or next((c for c in self.closed if c), None)
        if ref is None:
            self.ok = False
            return
        self.h, self.w = ref[0].shape[:2]
        H, W = plate_hw
        self.x0, self.y0 = max(0, min(self.x0, W - self.w)), max(0, min(self.y0, H - self.h))
        cx, cy, rx, ry = [v * s for v in e["iris"]]
        self.cx, self.cy, self.rx, self.ry = cx - self.x0, cy - self.y0, max(2.0, rx), max(2.0, ry)
        bx0, by0, bx1, by1 = [v * s for v in e["box_px"]]
        self.box = (bx0 - self.x0, by0 - self.y0, bx1 - self.x0, by1 - self.y0)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.X, self.Y = xx, yy
        r = np.sqrt(((xx - self.cx) / self.rx) ** 2 + ((yy - self.cy) / self.ry) ** 2)
        self.w1 = _smooth((1.32 - r) / 0.34)                       # moves with the gaze
        self.w2 = _smooth((0.78 - r) / 0.33)                       # pupil area (breathing / shock shrink)
        self.ok = True

    def remap(self, st):
        dx, dy = st["gx"] * 0.28 * self.rx, st["gy"] * 0.18 * self.ry
        ps = 1.0 + (st["pupil"] - 1.0) * self.w2
        mx = self.cx + (self.X - dx * self.w1 - self.cx) / ps
        my = self.cy + (self.Y - dy * self.w1 - self.cy) / ps
        return mx.astype(np.float32), my.astype(np.float32)


class EyeRig:
    def __init__(self, shot, plate_shape):
        self.eyes, self.on, self.report = [], False, {}
        info = (shot.get("plate") or {}).get("eyes")
        self.plan = shot.get("eye_plan")
        if not info or not info.get("eyes") or not self.plan:
            return
        W0 = info["size"][0]
        s = plate_shape[1] / float(W0)
        for e in info["eyes"]:
            try:
                eye = Eye(e, s, plate_shape[:2])
            except Exception:  # noqa: BLE001
                continue
            if eye.ok:
                self.eyes.append(eye)
        self.on = bool(self.eyes)
        self.report = {"eye_blink": "ok", "eye_gaze": "ok", "eye_pupil": "ok", "eye_catchlight": "ok"}
        if self.plan.get("glow", {}).get("on"):
            self.report["eye_glow"] = "ok"
        if self.plan.get("tears", {}).get("on"):
            self.report["eye_tears"] = "ok"

    def state(self, face, t):
        return eye_anim.state_at(self.plan, max(0.0, t + 0.11 * face))      # faces are slightly out of phase

    def apply(self, plate, t):
        if not self.on:
            return plate
        out = plate.copy()
        states = {}
        for e in self.eyes:
            st = states.get(e.face) or states.setdefault(e.face, self.state(e.face, t))
            roi = out[e.y0:e.y0 + e.h, e.x0:e.x0 + e.w]
            f = roi.astype(np.float32)
            if abs(st["gx"]) > 0.01 or abs(st["gy"]) > 0.01 or abs(st["pupil"] - 1.0) > 0.01:
                mx, my = e.remap(st)
                f = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            lid_i = eye_anim.lid_index(st["lid"])
            cover = 0.0 if lid_i == 0 else 1.0
            if e.shine and st["glint"] > 0.01:
                c, a = e.shine                                    # cv2.imread gives BGR like the plate; the shine layer is white
                f += 255.0 * a * st["glint"] * (1.0 - cover)
            if lid_i > 0 and e.closed[lid_i - 1]:
                c, a = e.closed[lid_i - 1]
                f = f * (1 - a) + c * a
            if e.glow and st["glow"] > 0.01:
                c, a = e.glow
                f += c * a * st["glow"] * (1.0 - 0.8 * cover)
            roi[...] = np.clip(f, 0, 255).astype(np.uint8)
        if self.plan.get("tears", {}).get("on"):
            for e in self.eyes:
                self._tears(out, e, states.get(e.face) or self.state(e.face, t))
        return out

    @staticmethod
    def _tears(out, e, st):
        t = st["tear"]
        if t <= 0.0:
            return
        bx0, by0, bx1, by1 = e.box
        bw, bh = bx1 - bx0, by1 - by0
        H, W = out.shape[:2]
        x = int(e.x0 + (bx0 + bx1) / 2 + (0.18 if e.side == "left" else -0.18) * bw)
        y0 = int(e.y0 + by1 - 0.12 * bh)
        run = min(1.0, t) * 2.2 * bh
        x0, y1 = max(0, x - int(bw)), min(H, int(y0 + run + 0.3 * bh))
        xa, xb, ya = x0, min(W, x + int(bw)), max(0, y0 - int(0.4 * bh))
        if y1 - ya < 4 or xb - xa < 4:
            return
        m = np.zeros((y1 - ya, xb - xa), np.float32)
        tw = max(2, int(0.075 * bw))
        well = min(1.0, t * 5.0)
        cv2.ellipse(m, (x - xa, y0 - ya), (int(0.34 * bw), max(1, int(0.06 * bh * (0.5 + well)))), 0, 0, 360, 0.55 * well, -1, cv2.LINE_AA)
        if run > 2:
            cv2.line(m, (x - xa, y0 - ya), (x - xa, int(y0 + run) - ya), 0.7, tw, cv2.LINE_AA)
            cv2.circle(m, (x - xa, int(y0 + run) - ya), int(tw * 1.25), 0.9, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), max(0.8, 0.012 * bw))[..., None]
        reg = out[ya:y1, xa:xb].astype(np.float32)
        tint = np.array([255, 245, 235], np.float32)                   # BGR: slightly cool-white water
        reg = reg * (1 - 0.45 * m) + tint * 0.45 * m
        hl = np.zeros(m.shape[:2], np.float32)
        if run > 2:
            cv2.line(hl, (x - xa - max(1, tw // 4), y0 - ya), (x - xa - max(1, tw // 4), int(y0 + run) - ya), 0.8, max(1, tw // 3), cv2.LINE_AA)
        reg += 120.0 * hl[..., None]
        out[ya:y1, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ baking for the Blender scene
def _patch_alpha(diff, feather):
    """Opaque only where the rig really changed the plate (lids, iris, glints, glow, tears), softly feathered. Everywhere else the
    patch is transparent, so hair / foliage that the plate shader sways around the eye is never frozen by the patch."""
    m = np.clip(diff.astype(np.float32) / 10.0, 0.0, 1.0)
    k = max(1, int(feather))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return np.clip(cv2.GaussianBlur(m, (0, 0), max(0.6, k * 0.8)) * 1.35, 0.0, 1.0)


def bake_patches(shot, out_dir, plate=None):
    """Bake the eye animation of one shot into per-eye RGBA patches (one PNG per DRAWING) for the Blender scene.

    The Blender process has no OpenCV, and the eye rig works on pixels, so it runs HERE (same code, same plan, same timing as the
    cv engine) and the result is handed over as pictures: `blender_extras.EyePatches` puts each patch on a quad that conforms
    to the plate's depth surface and swaps the texture per drawing. Identical drawings share one file.

    Returns [{"files": [path per drawing], "box": [x0, y0, x1, y1] (plate coords 0..1), "side", "face", "report"}] - [] when
    the shot has no eye plan / eye data, the plate is an AI-animated sequence (the eyes already move there) or anything fails.
    """
    pl = shot.get("plate") or {}
    if not pl.get("eyes") or not shot.get("eye_plan") or pl.get("sequence"):
        return []
    img = plate if plate is not None else cv2.imread(str(pl["image"]), cv2.IMREAD_COLOR)
    if img is None:
        return []
    rig = EyeRig(shot, img.shape)
    if not rig.on:
        return []
    H, W = img.shape[:2]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, fps = max(1, int(shot["step"])), float(shot["fps"])
    n = int(math.ceil(shot["frames"] / step))
    tears = bool(rig.plan.get("tears", {}).get("on"))
    boxes = []
    for e in rig.eyes:
        bx0, by0, bx1, by1 = e.box
        bw, bh = bx1 - bx0, by1 - by0
        x0, y0, x1, y1 = e.x0, e.y0, e.x0 + e.w, e.y0 + e.h
        if tears:                                                           # the streak runs far below the eye box
            x0, x1 = x0 - int(1.1 * bw), x1 + int(1.1 * bw)
            y0, y1 = y0 - int(0.5 * bh), y1 + int(2.9 * bh)
        boxes.append((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    files = [[] for _ in rig.eyes]
    seen = {}
    feather = max(2, int(round(0.012 * W / 4)))
    for i in range(n):
        frame = rig.apply(img, i * step / fps)
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            a, b = img[y0:y1, x0:x1], frame[y0:y1, x0:x1]
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
            alpha = _patch_alpha(diff, feather)
            bgra = np.dstack([b, np.round(alpha * 255).astype(np.uint8)])
            key = hashlib.md5(bgra.tobytes()).hexdigest()[:12]
            path = seen.get((k, key))
            if path is None:
                path = out_dir / f"eye{k}_{key}.png"
                if not path.exists():
                    cv2.imwrite(str(path), bgra)
                seen[(k, key)] = path
            files[k].append(str(path))
    res = []
    for k, e in enumerate(rig.eyes):
        x0, y0, x1, y1 = boxes[k]
        res.append({"files": files[k], "box": [x0 / W, y0 / H, x1 / W, y1 / H], "side": e.side, "face": e.face,
                    "report": dict(rig.report)})
    return res
