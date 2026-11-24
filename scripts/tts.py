"""Human-sounding narration with Microsoft Edge neural voices (free), one voice per language.

English  -> TTS_EN_VOICE (default en-US-AndrewMultilingualNeural, a warm natural male voice)
Arabic   -> TTS_AR_VOICE (default ar-DZ-IsmaelNeural, Algerian Arabic male voice)
Change them with env vars; list all voices with:  python -m edge_tts --list-voices
"""
import asyncio
import re
import subprocess
from dataclasses import dataclass

import numpy as np

from common import CACHE, env, log, sha

SR = 24000
LEAD = 0.30   # silence before a step starts talking
GAP = 0.22    # silence between two spoken parts of one step
TAIL = 0.50   # silence after a step

VOICES = {
    "en": env("TTS_EN_VOICE", "en-US-AndrewMultilingualNeural"),
    "ar": env("TTS_AR_VOICE", "ar-DZ-IsmaelNeural"),
}
FALLBACKS = {
    "en": ["en-US-AndrewNeural", "en-US-GuyNeural", "en-GB-RyanNeural"],
    "ar": ["ar-EG-ShakirNeural", "ar-SA-HamedNeural"],
}


def _rate(name, default):
    v = env(name, default)
    return v if re.fullmatch(r"[+-]\d{1,3}%", str(v)) else default


RATES = {"en": _rate("TTS_EN_RATE", "-6%"), "ar": _rate("TTS_AR_RATE", "-4%")}


def speech_text(text: str, lang: str) -> str:
    t = re.sub(r"[*_#`~|<>\[\]{}\\]", " ", text)
    if lang == "en":
        t = (t.replace("->", " to ").replace("=>", " gives ").replace("→", " to ").replace(" = ", " equals ")
              .replace(" + ", " plus ").replace("%", " percent").replace("°C", " degrees Celsius")
              .replace("µ", "micro ").replace("±", " plus or minus "))
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in ".!?؟…:":
        t += "."
    return t


@dataclass
class StepAudio:
    samples: np.ndarray  # int16 mono @ SR, includes LEAD, speech, TAIL and pause
    speech_end: float    # seconds from step start when speech ends (before TAIL/pause)

    @property
    def total(self) -> float:
        return len(self.samples) / SR


async def _synth(text, voice, rate, out, sem):
    async with sem:
        import edge_tts

        for attempt in range(1, 5):
            try:
                await edge_tts.Communicate(text, voice, rate=rate).save(str(out))
                if out.exists() and out.stat().st_size > 800:
                    return True
            except Exception as e:  # network, 403, NoAudioReceived...
                log(f"    tts retry {attempt} ({voice}): {type(e).__name__}: {str(e)[:100]}")
            await asyncio.sleep(min(2 ** attempt, 15))
        return False


def _decode(mp3) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-f", "s16le", "-ar", str(SR), "-ac", "1", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.int16)


async def _run_all(jobs):
    sem = asyncio.Semaphore(env("TTS_CONCURRENCY", 4, int))
    # jobs: key -> (text, lang, path)
    async def one(key, text, lang, path):
        if path.exists() and path.stat().st_size > 800:
            return key, True
        for voice in [VOICES[lang]] + FALLBACKS[lang]:
            if await _synth(text, voice, RATES[lang], path, sem):
                return key, True
            log(f"    voice {voice} failed, trying next")
        return key, False

    res = await asyncio.gather(*[one(k, *v) for k, v in jobs.items()])
    return dict(res)


def build_audio(steps) -> list:
    """Synthesize every spoken part of every step and return a list of StepAudio."""
    d = CACHE / "tts"
    d.mkdir(parents=True, exist_ok=True)
    jobs, plan = {}, []
    for step in steps:
        keys = []
        for lang, text in step.say:
            sp = speech_text(text, lang)
            if not sp:
                continue
            key = sha(VOICES[lang], RATES[lang], sp)
            jobs[key] = (sp, lang, d / f"{key}.mp3")
            keys.append((key, lang))
        plan.append(keys)
    log(f"  synthesizing {len(jobs)} speech segments ...")
    ok = asyncio.run(_run_all(jobs))
    out = []
    silence = lambda sec: np.zeros(int(sec * SR), dtype=np.int16)
    for step, keys in zip(steps, plan):
        chunks, first = [silence(LEAD)], True
        for key, lang in keys:
            if not ok.get(key):
                if lang == "en":
                    raise RuntimeError("English TTS failed repeatedly (edge-tts unreachable?)")
                log("    skipping one Arabic segment (voice unavailable)")
                continue
            if not first:
                chunks.append(silence(GAP))
            first = False
            chunks.append(_decode(d / f"{key}.mp3"))
        speech_end = sum(len(c) for c in chunks) / SR
        chunks.append(silence(TAIL + step.pause))
        out.append(StepAudio(np.concatenate(chunks), speech_end))
    return out
