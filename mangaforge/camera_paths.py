"""Pure-python camera language shared by the Blender scene and the no-Blender fallback engine (stdlib only).

`pose(move, t, strength)` answers: where is the camera at normalised shot time t in [0, 1]?
Units are engine-neutral (fractions), so Blender converts them to metres/degrees and the cv2 engine to crop windows:

  dx, dy        truck / pedestal as a FRACTION OF THE VISIBLE FRAME WIDTH at the subject plane (+x = camera right, +y = up)
  dz            dolly toward the subject as a FRACTION OF THE SUBJECT DISTANCE (0.2 = 20 % closer); never negative
  yaw/pitch/roll degrees (yaw + = turn camera left, pitch + = tilt up, roll + = counter-clockwise)
  orbit_deg     camera revolves around the subject by this angle while keeping it centred (+ = camera moves right)
  focal_scale   multiply the lens focal length (dolly-zoom uses it; 1.0 otherwise)
  shake         handheld/impact amplitude (fraction of frame width); apply with shake_offset()

Design rule: the plate is built larger than the frame (the frame covers PLATE_MARGIN of it at dz = 0), so with
|dx| <= 0.045, |dy| <= 0.045 and dz >= 0 the camera never sees past the edge of the artwork. Whip moves may exceed it
for a few frames: engines must put a blurred backdrop behind the plate.
"""
from __future__ import annotations

import math

PLATE_MARGIN = 0.90            # at dz = 0 the frame covers 90 % of the plate width/height
MAX_LATERAL = 0.045


def ease_in_out(t):
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (6 * t - 15) + 10)          # smootherstep


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = min(1.0, max(0.0, t))
    return t ** 3


def _zero():
    return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "orbit_deg": 0.0,
            "focal_scale": 1.0, "shake": 0.0}


def pose(move, t, strength=0.5):
    """-> dict of the keys above. `t` is linear shot time in [0, 1]; easing is applied inside."""
    s = 0.35 + 0.65 * min(1.0, max(0.0, strength))                # 0.35 .. 1.0
    e = ease_in_out(t)
    p = _zero()
    m = move or "drift"
    if m == "static":
        pass
    elif m == "drift":
        p.update(dx=0.020 * s * (2 * t - 1), dy=0.006 * s * (1 - 2 * t), dz=0.025 * s * t)
    elif m == "push_in":
        p.update(dz=0.24 * s * e, dx=0.012 * s * (2 * e - 1))
    elif m == "pull_out":
        p.update(dz=0.24 * s * (1 - e), dx=-0.012 * s * (2 * e - 1))
    elif m == "pan_l":
        p.update(dx=0.040 * s * (1 - 2 * e), yaw=1.6 * s * (2 * e - 1), dz=0.03 * s)
    elif m == "pan_r":
        p.update(dx=-0.040 * s * (1 - 2 * e), yaw=-1.6 * s * (2 * e - 1), dz=0.03 * s)
    elif m == "tilt_up":
        p.update(dy=0.035 * s * (2 * e - 1), pitch=1.4 * s * (2 * e - 1), dz=0.03 * s)
    elif m == "tilt_dn":
        p.update(dy=-0.035 * s * (2 * e - 1), pitch=-1.4 * s * (2 * e - 1), dz=0.03 * s)
    elif m == "orbit_l":
        p.update(orbit_deg=-9.0 * s * (2 * e - 1), dz=0.07 * s * e)
    elif m == "orbit_r":
        p.update(orbit_deg=9.0 * s * (2 * e - 1), dz=0.07 * s * e)
    elif m == "crane_up":
        p.update(dy=0.040 * s * (2 * e - 1), pitch=-1.8 * s * (e - 0.5) * 2, dz=0.10 * s * e)
    elif m == "crane_dn":
        p.update(dy=-0.040 * s * (2 * e - 1), pitch=1.8 * s * (e - 0.5) * 2, dz=0.10 * s * e)
    elif m == "whip_l":
        k = ease_in(t)
        p.update(dx=-0.12 * s * k, yaw=-6.0 * s * k, dz=0.04 * s)
    elif m == "whip_r":
        k = ease_in(t)
        p.update(dx=0.12 * s * k, yaw=6.0 * s * k, dz=0.04 * s)
    elif m == "handheld":
        p.update(dz=0.035 * s * t, shake=0.0035 * s)
    elif m == "punch_in":                                           # slow-slow-FAST speed ramp (hero beats)
        p.update(dz=0.30 * s * t ** 2.6, dx=0.006 * s * t, shake=0.004 * s * t ** 3)
    elif m == "dolly_zoom":                                         # vertigo: move in, widen the lens
        dz = 0.22 * s * e
        p.update(dz=dz, focal_scale=max(0.55, 1.0 - dz))
    else:
        p.update(dz=0.02 * s * t)
    return p


# ------------------------------------------------------------------ deterministic smooth noise (no numpy needed)
def _phases(seed, n=6):
    x = (seed * 2654435761 + 97) & 0xFFFFFFFF
    out = []
    for _ in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append((x / 0xFFFFFFFF) * 2 * math.pi)
    return out


def shake_offset(seed, frame, amp, fps=24):
    """Smooth pseudo-random hand-held motion -> (dx, dy, roll_deg). `amp` is a fraction of frame width."""
    if amp <= 0:
        return 0.0, 0.0, 0.0
    ph = _phases(seed)
    t = frame / float(fps)
    dx = sum(math.sin(2 * math.pi * f * t + p) / (i + 1) for i, (f, p) in enumerate(zip((0.9, 2.3, 4.1), ph[0:3])))
    dy = sum(math.sin(2 * math.pi * f * t + p) / (i + 1) for i, (f, p) in enumerate(zip((1.1, 2.9, 4.7), ph[3:6])))
    roll = math.sin(2 * math.pi * 0.7 * t + ph[0]) * 0.6 + math.sin(2 * math.pi * 2.1 * t + ph[3]) * 0.25
    return amp * dx * 0.6, amp * dy * 0.6, amp * roll * 40.0


def impact_shake(seed, frame, amp, hit_frame=0, fps=24, decay=9.0):
    """Decaying shake after a hit at `hit_frame` -> (dx, dy, roll_deg)."""
    if frame < hit_frame or amp <= 0:
        return 0.0, 0.0, 0.0
    t = (frame - hit_frame) / float(fps)
    k = math.exp(-decay * t)
    ph = _phases(seed + 17)
    dx = math.sin(2 * math.pi * 17 * t + ph[0]) * k
    dy = math.sin(2 * math.pi * 13 * t + ph[1]) * k
    roll = math.sin(2 * math.pi * 11 * t + ph[2]) * k
    return amp * dx, amp * dy, amp * roll * 25.0


def total_pose(move, t, strength, seed, frame, fps=24, shake_amp=0.0, hit_frame=None):
    """pose() + hand-held shake + impact shake folded into dx/dy/roll. The one call engines should use."""
    p = pose(move, t, strength)
    sx, sy, sr = shake_offset(seed, frame, p["shake"] + shake_amp, fps)
    p["dx"] += sx
    p["dy"] += sy
    p["roll"] += sr
    if hit_frame is not None:
        hx, hy, hr = impact_shake(seed, frame, max(shake_amp, 0.012), hit_frame, fps)
        p["dx"] += hx
        p["dy"] += hy
        p["roll"] += hr
    return p


def default_move_for(shot_type, intensity=2, index=0):
    """Fallback when neither the script nor the director chose a move."""
    pools = {
        "establishing": ["pull_out", "crane_up", "pan_l", "pan_r", "drift"],
        "wide": ["pan_r", "pan_l", "drift", "orbit_r", "crane_up"],
        "medium": ["push_in", "drift", "orbit_l", "orbit_r", "tilt_up"],
        "closeup": ["push_in", "drift", "handheld", "tilt_dn"],
        "insert": ["push_in", "tilt_dn", "drift"],
        "reaction": ["push_in", "handheld", "drift"],
        "action": ["handheld", "orbit_r", "orbit_l", "whip_r", "punch_in"],
        "impact": ["punch_in", "handheld"],
        "transition": ["whip_r", "whip_l"],
    }
    pool = pools.get(shot_type, pools["medium"])
    if intensity >= 5 and shot_type in ("action", "impact", "closeup"):
        return "punch_in"
    return pool[index % len(pool)]
