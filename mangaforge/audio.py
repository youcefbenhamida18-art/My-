"""Voices (Japanese neural TTS or your own recordings), original dark score, SFX and sidechain-style ducking."""
import asyncio
import subprocess
import wave

import numpy as np
from scipy.signal import butter, sosfilt

from .config import ROOT, log, media_duration, run

SR = 44100
STYLES = {  # style -> gender -> (edge voice, rate, pitch)
    "hero": {"male": ("ja-JP-KeitaNeural", "+4%", "+0Hz"), "female": ("ja-JP-NanamiNeural", "+4%", "+0Hz")},
    "villain": {"male": ("ja-JP-KeitaNeural", "-12%", "-14Hz"), "female": ("ja-JP-NanamiNeural", "-10%", "-10Hz")},
    "elder": {"male": ("ja-JP-KeitaNeural", "-15%", "-8Hz"), "female": ("ja-JP-NanamiNeural", "-14%", "-6Hz")},
    "young": {"male": ("ja-JP-KeitaNeural", "+8%", "+10Hz"), "female": ("ja-JP-NanamiNeural", "+8%", "+8Hz")},
    "narrator": {"male": ("ja-JP-KeitaNeural", "-6%", "-4Hz"), "female": ("ja-JP-NanamiNeural", "-6%", "-4Hz")},
}
HUMAN_DIR = ROOT / "assets" / "voices"      # drop real recordings as <ep>_<scene>_<line>.mp3/.wav to override TTS


# ------------------------------------------------------------------ voices
def _pick(chars, speaker):
    c = chars.get(speaker) or {}
    style = c.get("voice") if c.get("voice") in STYLES else "narrator"
    gender = c.get("gender") if c.get("gender") in ("male", "female") else "male"
    return STYLES[style][gender]


def _silence(path, sec=2.0):
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", sec, "-c:a", "libmp3lame", path])


async def _tts(jobs):
    import edge_tts
    sem = asyncio.Semaphore(4)

    async def one(j):
        async with sem:
            for k in range(4):
                try:
                    await edge_tts.Communicate(j["text"], j["voice"], rate=j["rate"], pitch=j["pitch"]).save(j["out"])
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("tts retry %s: %s", j["out"], e)
                    await asyncio.sleep(2 * (k + 1))
            _silence(j["out"])           # never kill the episode for one line

    await asyncio.gather(*(one(j) for j in jobs))


def synthesize(bible, scenes, ep, outdir):
    """-> {(scene, line): (path, seconds)}"""
    chars = {c["id"]: c for c in bible["characters"]}
    jobs, res = [], {}
    for si, sc in enumerate(scenes):
        for li, ln in enumerate(sc["lines"]):
            human = list(HUMAN_DIR.glob(f"{ep}_{si}_{li}.*")) if HUMAN_DIR.exists() else []
            if human:
                res[(si, li)] = str(human[0])
                continue
            v, rate, pitch = _pick(chars, ln["speaker"])
            out = str(outdir / f"v_{si:03d}_{li}.mp3")
            jobs.append(dict(text=ln["ja"], voice=v, rate=rate, pitch=pitch, out=out))
            res[(si, li)] = out
    asyncio.run(_tts(jobs))
    return {k: (p, media_duration(p)) for k, p in res.items()}


def load(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def load_voice(path):
    x = load(path)
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-6)
    return np.clip(x * (0.14 / rms), -0.95, 0.95)


def save_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ original score + sfx
def music(dur, energy, seed=7):
    """Dark evolving drone (minor chords + sub) with heartbeat pulses that follow the scene intensity."""
    n = int(dur * SR)
    out = np.zeros(n, np.float32)
    roots, seg, fade = [55.0, 43.65, 49.0, 41.2], 14.0, 3.0
    for i in range(int(dur // seg) + 2):
        a = int(i * seg * SR)
        if a >= n:
            break
        L = int((seg + 2 * fade) * SR)
        b = min(n, a + L)
        lt = (np.arange(b - a, dtype=np.float32)) / SR
        env = np.sin(np.pi * np.clip(lt / (seg + 2 * fade), 0, 1)) ** 2
        f0 = roots[i % 4]
        sig = np.zeros_like(lt)
        for ratio in (1.0, 1.5, 2.4):
            for det in (-0.003, 0.003):
                for h in range(1, 5):
                    sig += np.sin(2 * np.pi * f0 * ratio * h * (1 + det) * lt) / h ** 1.4
        sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * lt + i)
        sig = sig / 6 + 0.9 * np.sin(2 * np.pi * f0 * lt)
        out[a:b] += (sig * env).astype(np.float32)
    out /= max(1e-6, np.abs(out).max()) * 2.0
    tk = np.arange(int(0.35 * SR), dtype=np.float32) / SR
    kick = np.sin(2 * np.pi * np.cumsum(42 + 70 * np.exp(-tk * 30)) / SR) * np.exp(-tk * 8)
    for tb in np.arange(0, dur - 0.4, 0.85):
        e = energy[min(int(tb), len(energy) - 1)]
        if e >= 3:
            a = int(tb * SR)
            out[a:a + len(kick)] += kick[:max(0, n - a)] * 0.22 * (e - 2) / 3
    return out


def _band(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def sfx(kind):
    rng = np.random.default_rng(3)
    if kind == "boom":
        t = np.arange(int(1.8 * SR)) / SR
        body = np.sin(2 * np.pi * np.cumsum(38 + 100 * np.exp(-t * 9)) / SR) * np.exp(-t * 2.6)
        hit = _band(rng.standard_normal(len(t)).astype(np.float32), 120, 3000) * np.exp(-t * 14)
        return (0.9 * body + 0.5 * hit).astype(np.float32)
    if kind == "whoosh":
        t = np.arange(int(0.7 * SR)) / SR
        return (_band(rng.standard_normal(len(t)).astype(np.float32), 300, 5000)
                * np.sin(np.pi * t / 0.7) ** 2 * 0.9).astype(np.float32)
    d = 2.5                                                              # riser
    t = np.arange(int(d * SR)) / SR
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1000 * (t / d) ** 2) / SR)
    noise = _band(rng.standard_normal(len(t)).astype(np.float32), 800, 7000)
    return ((0.5 * sweep + 0.6 * noise) * (t / d) ** 3).astype(np.float32)


GAIN = {"boom": 0.9, "whoosh": 0.45, "riser": 0.5}


def mix(total, voice_events, energy, sfx_events, out_wav):
    n = int((total + 1) * SR)
    voice = np.zeros(n, np.float32)
    for t, arr in voice_events:
        a = int(t * SR)
        voice[a:a + len(arr)] += arr[:max(0, n - a)]
    fx = np.zeros(n, np.float32)
    cache = {}
    for t, kind in sfx_events:
        w = cache.setdefault(kind, sfx(kind))
        a = int(t * SR)
        fx[a:a + len(w)] += w[:max(0, n - a)] * GAIN[kind]
    mus = music(total + 1, energy)[:n]
    blk = 441
    m = (n // blk) * blk
    env = np.abs(voice[:m]).reshape(-1, blk).max(1)
    env = np.clip(np.convolve(env, np.ones(30) / 30, "same") * 3, 0, 1)
    duck = np.interp(np.arange(n), np.arange(len(env)) * blk + blk / 2, 1 - 0.7 * env)
    out = voice + mus * 0.38 * duck + fx * 0.6
    out /= max(1.0, float(np.abs(out).max()) / 0.95)
    save_wav(out_wav, out)


def build_mix(shots, fps, out_wav):
    starts, t = [], 0.0
    for s in shots:
        starts.append(t)
        t += s["frames"] / fps
    total = t
    energy = np.zeros(int(total) + 3)
    voice_events, sfx_events = [], []
    for s, a in zip(shots, starts):
        b = a + s["frames"] / fps
        energy[int(a):int(b) + 1] = np.maximum(energy[int(a):int(b) + 1], s["intensity"])
        if s.get("voice"):
            voice_events.append((a + 0.12, load_voice(s["voice"])))
        if s["impact"]:
            sfx_events.append((a, "boom"))
        elif s["trans"] == "whip":
            sfx_events.append((max(0.0, a - 0.05), "whoosh"))
        if s["act_end"]:
            sfx_events.append((max(0.0, b - 2.5), "riser"))
    mix(total, voice_events, energy, sfx_events, out_wav)
    return total
