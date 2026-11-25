"""Eye animation: a deterministic per-shot PLAN (blinks, saccades, pupil, glint, glow, tears) and `state_at(plan, t)`.

Pure standard library: the cv engine imports it normally, Blender's scene script loads it by file path.
Anime eyes live on twos, so engines sample the state once per drawing; everything here is a smooth function of time.
"""
import math
import random

LID_LEVELS = (0.0, 0.45, 0.8, 1.0)          # the closed-lid drawings that exist (index 0 = open, 1..3 = layers closed_1..3)

# emotion -> (blink interval s (min, max), pupil scale, permanent lid droop, gaze activity)
EMOTION_EYES = {
    "neutral": ((2.6, 5.2), 1.00, 0.00, 1.0), "calm": ((3.0, 6.0), 1.00, 0.04, 0.6), "angry": ((4.5, 8.0), 0.90, 0.00, 0.5),
    "sad": ((2.2, 4.0), 1.02, 0.10, 0.4), "happy": ((2.2, 4.2), 1.08, 0.00, 1.0), "shout": ((5.0, 9.0), 0.88, 0.00, 0.4),
    "whisper": ((2.5, 4.5), 1.00, 0.04, 0.5), "fear": ((1.4, 2.6), 0.72, 0.00, 1.6), "cold": ((4.0, 7.5), 0.95, 0.05, 0.3),
    "tender": ((2.4, 4.4), 1.10, 0.07, 0.7),
}


def _smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def plan(duration, *, seed=1, emotion="neutral", intensity=3, mode=None, hit_at=None, glow_at=None):
    """-> JSON-able plan. mode: None | 'closeup' | 'glow' | 'tears'."""
    rnd = random.Random(int(seed) * 7919 + 13)
    (b0, b1), pupil, droop, act = EMOTION_EYES.get(emotion, EMOTION_EYES["neutral"])
    intens = max(1, min(5, int(intensity or 3)))
    duration = max(0.2, float(duration))
    blinks, t = [], rnd.uniform(0.45, min(b1, max(0.6, duration * 0.6)))
    while t < duration - 0.25:
        near_hit = hit_at is not None and hit_at - 0.2 < t < hit_at + 0.45
        if not near_hit and not (mode == "closeup" and t < 0.35):
            d = rnd.uniform(0.17, 0.24) * (1.25 if mode == "closeup" or emotion in ("sad", "tender") else 1.0)
            blinks.append([round(t, 3), round(d, 3)])
        t += rnd.uniform(b0, b1)
    if mode == "closeup" and not blinks and duration > 1.2:                 # one deliberate slow blink for drama
        blinks.append([round(duration * rnd.uniform(0.55, 0.7), 3), 0.27])
    gaze, t = [[0.0, rnd.uniform(-0.3, 0.3), rnd.uniform(-0.2, 0.2)]], rnd.uniform(0.5, 1.4) / max(0.3, act)
    amp = 0.55 if mode == "closeup" else 1.0
    while t < duration - 0.2:
        gaze.append([round(t, 3), round(rnd.uniform(-1, 1) * amp, 3), round(rnd.uniform(-0.6, 0.6) * amp, 3)])
        t += rnd.uniform(0.7, 2.2) / max(0.3, act)
    ev = []                                                                   # sparkle events (peak catch-light)
    t = rnd.uniform(0.3, 1.2)
    while t < duration:
        ev.append(round(t, 3))
        t += rnd.uniform(0.9, 2.4)
    return {"duration": duration, "blinks": blinks, "gaze": gaze, "pupil": pupil, "droop": droop, "sparkles": ev,
            "phase": rnd.uniform(0, 6.28), "hit_at": hit_at, "mode": mode,
            "glow": {"on": mode == "glow", "at": float(glow_at if glow_at is not None else (hit_at if hit_at is not None else 0.0)),
                     "ramp": 0.45},
            "tears": {"on": mode == "tears", "at": rnd.uniform(0.15, 0.4), "run": rnd.uniform(1.6, 2.6)},
            "intensity": intens}


def _blink(b, t):
    t0, d = b
    x = (t - t0) / d
    if x < 0 or x > 1:
        return 0.0
    if x < 0.35:
        return _smooth(x / 0.35)
    if x < 0.5:
        return 1.0
    return 1.0 - _smooth((x - 0.5) / 0.5)


def state_at(p, t):
    lid = max([_blink(b, t) for b in p["blinks"]] + [p["droop"]])
    g = p["gaze"]
    gx, gy = g[0][1], g[0][2]
    for i in range(1, len(g)):
        if t >= g[i][0]:
            gx, gy = g[i][1], g[i][2]
        else:
            break
    for i in range(1, len(g)):                                                  # fast saccade (70 ms) into each new target
        k = (t - g[i][0]) / 0.07
        if 0 <= k < 1:
            a = _smooth(k)
            gx, gy = g[i - 1][1] + (g[i][1] - g[i - 1][1]) * a, g[i - 1][2] + (g[i][2] - g[i - 1][2]) * a
            break
    pupil = p["pupil"] * (1.0 + 0.025 * math.sin(2 * math.pi * t / 2.6 + p["phase"]))
    h = p.get("hit_at")
    if h is not None and t >= h:
        k = t - h
        pupil *= 1.0 - 0.28 * (_smooth(k / 0.05) if k < 0.05 else (1.0 - _smooth((k - 0.2) / 0.5) if k > 0.2 else 1.0))
    glint = 0.78 + 0.1 * math.sin(2 * math.pi * t / 1.7 + p["phase"])
    for s in p["sparkles"]:
        k = abs(t - s) / 0.12
        if k < 1:
            glint = max(glint, 0.78 + 0.22 * (1 - k))
    gl = p["glow"]
    glow = 0.0
    if gl["on"] and t >= gl["at"]:
        glow = _smooth((t - gl["at"]) / gl["ramp"]) * (0.92 + 0.08 * math.sin(2 * math.pi * t * 1.3))
    tr = p["tears"]
    tear = 0.0
    if tr["on"] and t >= tr["at"]:
        tear = min(1.0, (t - tr["at"]) / tr["run"])
    return {"lid": round(lid, 4), "gx": gx, "gy": gy, "pupil": pupil, "glint": min(1.0, glint) * (1.0 - 0.9 * lid), "glow": glow,
            "tear": tear}


def lid_index(lid):
    """Which closed-lid drawing to show: 0 = eye open, 1..3 = closed_1..closed_3 (nearest level)."""
    best, bi = 9.0, 0
    for i, lv in enumerate(LID_LEVELS):
        d = abs(lv - lid)
        if d < best:
            best, bi = d, i
    return bi
