"""Procedural score, SFX and the final mix (numpy + scipy.signal, no sample libraries needed).

User assets win: tracks in assets/music/<mood>/ and sounds in assets/sfx/<kind>*.(wav|mp3|ogg|flac|m4a) replace the
procedural versions. The procedural score uses plucked strings (Karplus-Strong, koto/shamisen-like) in a Japanese
miyako-bushi-style scale, taiko hits, a breathy flute and pads - enough for atmosphere, not a replacement for an OST.
"""
from __future__ import annotations

import pathlib
import subprocess
import wave

import numpy as np

from .config import ASSETS, log

SR = 44100
AUDIO_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
SCALES = {"in": [0, 1, 5, 7, 8], "major_pent": [0, 2, 4, 7, 9], "minor_pent": [0, 3, 5, 7, 10]}


# ------------------------------------------------------------------ io
def decode_audio(path, sr=SR, channels=1):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    return x.reshape(-1, channels) if channels > 1 else x.copy()


def write_wav(path, x, sr=SR):
    x = np.asarray(x, dtype=np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if x.ndim == 1 else x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return pathlib.Path(path)


# ------------------------------------------------------------------ dsp helpers
def _sos(kind, fc, order=2):
    from scipy.signal import butter
    fc = np.atleast_1d(fc) / (SR / 2)
    fc = np.clip(fc, 1e-4, 0.99)
    return butter(order, fc if len(fc) > 1 else fc[0], btype=kind, output="sos")


def lp(x, fc):
    from scipy.signal import sosfilt
    return sosfilt(_sos("low", fc), x)


def hp(x, fc):
    from scipy.signal import sosfilt
    return sosfilt(_sos("high", fc), x)


def bp(x, lo, hi):
    from scipy.signal import sosfilt
    return sosfilt(_sos("band", [lo, hi]), x)


def reverb(x, rng, t=1.3, wet=0.28):
    from scipy.signal import fftconvolve
    n = int(SR * t)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 6.5, n))
    ir = lp(ir, 5500)
    ir /= np.abs(ir).sum() * 0.35 + 1e-9
    y = fftconvolve(x, ir)[: len(x)]
    return (1 - wet) * x + wet * y


def t_axis(dur):
    return np.arange(int(SR * dur)) / SR


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    env = np.ones(n) * s
    a, d, r = min(a, n), min(d, max(0, n - a)), min(r, n)
    if a:
        env[:a] = np.linspace(0, 1, a)
    if d:
        env[a:a + d] = np.linspace(1, s, d)
    if r:
        env[n - r:] *= np.linspace(1, 0, r)
    return env


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def pan(x, p):
    """mono -> stereo with equal-power pan p in [-1,1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1)


def place(dst, src, t0):
    i = int(t0 * SR)
    if i >= len(dst) or i + len(src) <= 0:
        return
    s0 = max(0, -i)
    i = max(0, i)
    n = min(len(src) - s0, len(dst) - i)
    if n > 0:
        dst[i:i + n] += src[s0:s0 + n]


# ------------------------------------------------------------------ instruments
def ks_pluck(freq, dur, rng, decay=0.997, bright=0.55):
    """Karplus-Strong plucked string, block-vectorised."""
    n = int(SR * dur)
    N = max(8, int(round(SR / freq)))
    buf = rng.uniform(-1, 1, N)
    buf = lp(buf, max(600.0, freq * (2 + 8 * bright)))
    out = np.empty(n + N)
    out[:N] = buf
    carry = 0.0
    prev = buf
    pos = N
    while pos < n:
        shifted = np.concatenate(([carry], prev[:-1]))
        blk = decay * 0.5 * (prev + shifted)
        out[pos:pos + N] = blk
        carry = prev[-1]
        prev = blk
        pos += N
    y = out[:n]
    return y / (np.abs(y).max() + 1e-9)


def bell(freq, dur, rng):
    t = t_axis(dur)
    y = sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t * dcy)
            for r, a, dcy in ((1, 1, 1.4), (2.76, .5, 2.2), (5.4, .25, 3.5), (8.9, .12, 5)))
    return y / (np.abs(y).max() + 1e-9)


def taiko(dur=1.3, f0=120.0, f1=48.0, rng=None):
    rng = rng or np.random.default_rng(1)
    t = t_axis(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.07)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.33)
    skin = bp(rng.standard_normal(len(t)), 180, 2200) * np.exp(-t / 0.03)
    y = body + 0.5 * skin
    return y / (np.abs(y).max() + 1e-9)


def flute(freq, dur, rng):
    t = t_axis(dur)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.5)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    y = np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.10 * np.sin(3 * ph)
    breath = bp(rng.standard_normal(len(t)), freq * 1.5, freq * 4.5) * 0.35
    y = (y + breath) * adsr(len(t), 0.22, 0.1, 0.8, 0.45)
    return y / (np.abs(y).max() + 1e-9)


def pad(freqs, dur, rng):
    t = t_axis(dur)
    y = np.zeros_like(t)
    for f in freqs:
        for det in (-0.004, 0.0, 0.004):
            y += sum(np.sin(2 * np.pi * f * (1 + det) * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 7))
    y = lp(y, 950) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6)))
    y *= adsr(len(t), 1.8, 0.3, 0.9, 2.0)
    return y / (np.abs(y).max() + 1e-9)


def riser(dur, rng):
    t = t_axis(dur)
    n = rng.standard_normal(len(t))
    sweep = np.linspace(300, 7000, len(t))
    y = np.zeros_like(n)
    for k in range(6):                                  # crude time-varying band by chunked filtering
        a, b = k * len(t) // 6, (k + 1) * len(t) // 6
        c = float(sweep[a])
        y[a:b] = bp(n, c, c * 1.8)[a:b]
    y = y * (t / dur) ** 2 + 0.3 * np.sin(2 * np.pi * np.cumsum(np.linspace(110, 880, len(t))) / SR) * (t / dur) ** 3
    return y / (np.abs(y).max() + 1e-9)


# ------------------------------------------------------------------ SFX library
def _noise(dur, rng):
    return rng.standard_normal(int(SR * dur))


def sfx(kind, rng):
    if kind == "boom":
        y = taiko(1.8, 90, 32, rng) + 0.4 * lp(_noise(1.8, rng), 300) * np.exp(-t_axis(1.8) / 0.5)
    elif kind == "hit":
        t = t_axis(0.45)
        y = 0.9 * bp(_noise(0.45, rng), 150, 3500) * np.exp(-t / 0.05) + taiko(0.45, 140, 55, rng)
    elif kind == "slash":
        t = t_axis(0.7)
        y = bp(_noise(0.7, rng), 2500, 9000) * np.exp(-t / 0.12) * np.minimum(1, t / 0.03)
        y += 0.5 * sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((3100, 7), (4700, 9), (6300, 12)))
    elif kind == "whoosh":
        n = _noise(1.0, rng)
        t = t_axis(1.0)
        y = np.zeros_like(n)
        for k in range(8):
            a, b = k * len(t) // 8, (k + 1) * len(t) // 8
            c = 500 + 3500 * np.sin(np.pi * (k + .5) / 8)
            y[a:b] = bp(n, c, c * 1.7)[a:b]
        y *= np.sin(np.pi * t / 1.0) ** 2
    elif kind == "footsteps":
        y = np.zeros(int(SR * 1.8))
        for i in range(4):
            th = lp(_noise(0.18, rng), 420) * np.exp(-t_axis(0.18) / 0.045)
            place(y, th, 0.12 + i * 0.42 + rng.uniform(-.02, .02))
    elif kind == "thunder":
        t = t_axis(3.2)
        y = lp(_noise(3.2, rng), 170) * np.exp(-t / 1.2) * (1 + 0.6 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6)))
        y += 0.3 * lp(_noise(3.2, rng), 900) * np.exp(-t / 0.35)
    elif kind == "wind":
        t = t_axis(3.0)
        y = bp(_noise(3.0, rng), 200, 1400) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t + rng.uniform(0, 6)))
        y *= np.minimum(1, t / 0.6) * np.minimum(1, (3.0 - t) / 0.8)
    elif kind == "riser":
        y = riser(2.6, rng)
    elif kind == "shimmer":
        t = t_axis(1.6)
        y = sum(np.sin(2 * np.pi * f * t) for f in (2093, 2637, 3136, 3951)) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
        y *= np.exp(-t / 0.9) * np.minimum(1, t / 0.05)
    elif kind == "clash":
        t = t_axis(1.2)
        y = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t * d)
                for f, d in ((1180, 3.2), (1930, 4.5), (2750, 6), (4310, 8), (5900, 10)))
        y += hp(_noise(1.2, rng), 3000) * np.exp(-t / 0.04)
    elif kind == "heartbeat":
        y = np.zeros(int(SR * 1.0))
        for off, a in ((0.0, 1.0), (0.22, 0.7)):
            place(y, a * lp(np.sin(2 * np.pi * 52 * t_axis(0.25)) * np.exp(-t_axis(0.25) / 0.06), 160), off)
    elif kind == "rain":
        y = hp(_noise(3.0, rng), 1800) * 0.35 * np.minimum(1, t_axis(3.0) / 0.5)
    elif kind == "fire":
        t = t_axis(2.5)
        y = lp(_noise(2.5, rng), 700) * 0.4
        for _ in range(60):
            c = hp(_noise(0.012, rng), 2500) * rng.uniform(0.3, 1)
            place(y, c, rng.uniform(0, 2.4))
    elif kind == "glass":
        t = t_axis(1.3)
        y = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t * d)
                for f, d in ((2900, 4), (4200, 6), (5600, 9), (7400, 12))) + hp(_noise(1.3, rng), 5000) * np.exp(-t / 0.05)
    else:
        y = np.zeros(int(SR * 0.1))
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def get_sfx(kind, rng):
    d = ASSETS / "sfx"
    files = sorted(p for p in d.glob(f"{kind}*") if p.suffix.lower() in AUDIO_EXT) if d.exists() else []
    if files:
        try:
            x = decode_audio(files[int(rng.integers(len(files)))], channels=1)
            return x / (np.abs(x).max() + 1e-9)
        except Exception as e:  # noqa: BLE001
            log.warning("sfx file for %s unreadable (%s) - using procedural", kind, e)
    return sfx(kind, rng)


# ------------------------------------------------------------------ music
MOODS = {  # tempo, root midi, scale, density of plucks (per beat), has taiko, taiko pattern, drone, shimmer
    "calm": dict(bpm=66, root=50, scale="in", pluck=.35, taiko=0, drone=1, flute=.3),
    "warm": dict(bpm=72, root=52, scale="major_pent", pluck=.5, taiko=0, drone=1, flute=.2),
    "tense": dict(bpm=84, root=45, scale="in", pluck=.25, taiko=0, drone=2, heart=True),
    "dark": dict(bpm=58, root=43, scale="in", pluck=.15, taiko=.12, drone=2, flute=.2),
    "battle": dict(bpm=152, root=47, scale="in", pluck=.9, taiko=1, drone=1, riser=True),
    "sad": dict(bpm=54, root=50, scale="minor_pent", pluck=.3, taiko=0, drone=1, flute=.5, long=True),
    "epic": dict(bpm=96, root=45, scale="in", pluck=.4, taiko=.6, drone=1, flute=.5, riser=True),
    "mystic": dict(bpm=60, root=52, scale="in", pluck=.2, taiko=0, drone=1, bells=True, flute=.25),
}


def procedural_cue(mood, dur, rng):
    m = MOODS.get(mood) or MOODS["calm"]
    n = int(SR * dur)
    out = np.zeros((n, 2), dtype=np.float32)
    spb = 60.0 / m["bpm"]
    sc = SCALES[m["scale"]]
    root = m["root"]
    f_root = midi_hz(root)
    freqs = [f_root, f_root * 1.5] if m.get("drone", 1) == 1 else [f_root, f_root * 1.5, midi_hz(root + 1)]
    out += pan(pad(freqs, dur, rng), 0).astype(np.float32) * 0.22
    t = 0.0
    beat = 0
    while t < dur - 0.3:
        if rng.random() < m["pluck"]:
            deg = sc[int(rng.integers(len(sc)))] + 12 * int(rng.integers(1, 3))
            note = ks_pluck(midi_hz(root + deg), min(3.2 if m.get("long") else 2.2, dur - t),
                            rng, decay=0.9985 if m.get("long") else 0.997)
            note = reverb(note, rng, 1.6, 0.35) * rng.uniform(0.25, 0.45)
            place(out, pan(note, rng.uniform(-.7, .7)).astype(np.float32), t)
        if m.get("bells") and rng.random() < 0.18:
            place(out, pan(reverb(bell(midi_hz(root + 36 + sc[int(rng.integers(len(sc)))]), 3.0, rng), rng, 2.2, .5) * .16,
                           rng.uniform(-.8, .8)).astype(np.float32), t)
        tk = m.get("taiko", 0)
        if tk and (rng.random() < tk) and beat % (2 if tk >= 1 else 4) == 0:
            hit = taiko(1.3, 125, 46, rng) * (0.7 if beat % 4 else 1.0)
            place(out, pan(hit, 0).astype(np.float32), t)
            if tk >= 1:
                place(out, pan(taiko(.6, 150, 70, rng) * .35, rng.uniform(-.3, .3)).astype(np.float32), t + spb / 2)
        if m.get("heart") and beat % 2 == 0:
            place(out, pan(get_sfx("heartbeat", rng) * .5, 0).astype(np.float32), t)
        if m.get("flute") and beat % 8 == 0 and rng.random() < m["flute"]:
            deg = sc[int(rng.integers(len(sc)))] + 24
            fl = reverb(flute(midi_hz(root + deg), min(4.5, dur - t), rng), rng, 1.8, .4) * .22
            place(out, pan(fl, rng.uniform(-.4, .4)).astype(np.float32), t)
        t += spb / (2 if m["pluck"] > .6 else 1)
        beat += 1
    if m.get("riser") and dur > 4:
        place(out, pan(riser(min(3.0, dur / 2), rng) * .25, 0).astype(np.float32), dur - min(3.0, dur / 2) - 0.2)
    pk = np.abs(out).max() + 1e-9
    return out / pk * 0.6


def music_cue(mood, dur, rng):
    d = ASSETS / "music" / mood
    files = sorted(p for p in d.glob("*") if p.suffix.lower() in AUDIO_EXT) if d.exists() else []
    if files:
        try:
            x = decode_audio(files[int(rng.integers(len(files)))], channels=2)
            if len(x) < int(dur * SR):
                x = np.tile(x, (int(np.ceil(dur * SR / len(x))), 1))
            return (x[: int(dur * SR)] / (np.abs(x).max() + 1e-9) * 0.6).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            log.warning("music file for %s unreadable (%s) - using procedural", mood, e)
    return procedural_cue(mood, dur, rng)


# ------------------------------------------------------------------ the mix
def _smooth(x, sec):
    k = max(1, int(sec * SR))
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k // 2, y[0]), y, np.full(len(x) - len(y) - k // 2, y[-1])])[: len(x)]


def build_soundtrack(shots, out_wav, *, seed=0):
    """shots: render-ready shot dicts (uses dur and audio{voice,voice_start,sfx[],music,intensity}). -> wav path."""
    rng = np.random.default_rng(seed)
    total = sum(s["dur"] for s in shots)
    N = int(total * SR) + SR // 2
    music = np.zeros((N, 2), np.float32)
    sfxb = np.zeros((N, 2), np.float32)
    voice = np.zeros(N, np.float32)
    # music segments by mood
    segs = []
    t = 0.0
    for s in shots:
        a = s.get("audio") or {}
        mood = a.get("music") or "calm"
        if mood == "none":
            mood = None
        if segs and segs[-1][0] == mood:
            segs[-1][2] += s["dur"]
        else:
            segs.append([mood, t, s["dur"]])
        t += s["dur"]
    for mood, t0, d in segs:
        if not mood:
            continue
        cue = music_cue(mood, d + 1.6, np.random.default_rng(int(rng.integers(1 << 30))))
        fi, fo = int(0.8 * SR), int(1.6 * SR)
        env = np.ones(len(cue), np.float32)
        env[:fi] = np.linspace(0, 1, fi)
        env[-fo:] = np.linspace(1, 0, fo)
        place(music, cue * env[:, None], max(0.0, t0 - 0.4))
    t = 0.0
    for s in shots:
        a = s.get("audio") or {}
        for e in a.get("sfx") or []:
            kind = e.get("kind") if isinstance(e, dict) else e
            x = get_sfx(kind, rng) * (0.75 if kind not in ("boom", "thunder") else 0.95)
            place(sfxb, pan(x, rng.uniform(-.25, .25)).astype(np.float32), t + (e.get("t", 0) if isinstance(e, dict) else 0))
        if a.get("voice"):
            try:
                v = decode_audio(a["voice"], channels=1)
                place(voice, v.astype(np.float32), t + (a.get("voice_start") or 0.15))
            except Exception as e:  # noqa: BLE001
                log.error("voice file unreadable (%s): %s", a.get("voice"), e)
        t += s["dur"]
    env = _smooth(np.abs(voice), 0.18)
    duck = 1.0 - 0.55 * np.clip(env / 0.06, 0, 1)
    duck = _smooth(duck, 0.12)
    mix = music * duck[:, None] * 0.85 + sfxb * 0.9 + np.stack([voice, voice], 1) * 1.15
    mix = np.tanh(mix * 1.05) / np.tanh(1.05)                   # gentle soft limiter
    mix /= max(1.0, float(np.abs(mix).max()) / 0.92)
    mix = mix[: int(total * SR)]
    write_wav(out_wav, mix)
    return pathlib.Path(out_wav)
