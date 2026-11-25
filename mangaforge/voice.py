"""Voice acting: VOICEVOX (expressive Japanese anime voices, local engine) -> edge-tts -> human files -> loud silence.

`synthesize(script, work)` -> {shot_id: {"path","dur","speaker","provider","credit"}} for every shot with a line.
VOICEVOX voices need a credit in the video description ("VOICEVOX:<speaker>"); the entries carry it in "credit".
"""
from __future__ import annotations

import asyncio
import pathlib
import re

import numpy as np
import requests

from .config import ASSETS, cfg, log
from .sound import SR, decode_audio, write_wav

MALE = {"hero": ("白上虎太郎", "玄野武宏", "青山龍星"), "villain": ("青山龍星", "剣崎雌雄", "雀松朱司", "玄野武宏"),
        "elder": ("ちび式じい", "麒ヶ島宗麟", "青山龍星"), "young": ("白上虎太郎", "九州そら", "玄野武宏"),
        "boy": ("白上虎太郎", "玄野武宏"), "narrator": ("玄野武宏", "青山龍星", "雀松朱司")}
FEMALE = {"hero": ("四国めたん", "波音リツ", "†聖騎士 紅桜†", "春日部つむぎ"), "villain": ("冥鳴ひまり", "後鬼", "四国めたん"),
          "elder": ("櫻歌ミコ", "冥鳴ひまり"), "young": ("春日部つむぎ", "雨晴はう", "九州そら"),
          "girl": ("春日部つむぎ", "雨晴はう", "冥鳴ひまり", "四国めたん"), "narrator": ("小夜/SAYO", "春日部つむぎ", "冥鳴ひまり")}
STYLE_KEYS = {"angry": ("怒", "おこ", "熱血", "ツンギレ", "不機嫌"), "shout": ("熱血", "怒", "おこ"),
              "sad": ("悲", "かなし", "なみだ", "びえーん", "しっとり"), "whisper": ("囁", "ささやき", "ヒソヒソ"),
              "happy": ("喜", "わーい", "あまあま"), "fear": ("びくびく", "ヘロヘロ", "なみだ"),
              "tender": ("しっとり", "あまあま"), "cold": ("不機嫌", "ツンツン", "ノーマル"),
              "calm": ("ノーマル", "ふつう"), "neutral": ("ノーマル", "ふつう")}
EDGE = {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"}
EDGE_PROSODY = {"angry": ("+8%", "+3Hz"), "shout": ("+12%", "+5Hz"), "sad": ("-12%", "-6Hz"), "whisper": ("-10%", "-4Hz"),
                "fear": ("+6%", "+4Hz"), "happy": ("+6%", "+3Hz"), "cold": ("-4%", "-3Hz")}


class VoiceError(RuntimeError):
    pass


# ------------------------------------------------------------------ provider discovery
def _vv_ok():
    try:
        return requests.get(cfg.voicevox_url + "/version", timeout=3).ok
    except Exception:  # noqa: BLE001
        return False


def _edge_ok():
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def available_providers():
    return {"voicevox": _vv_ok(), "edge": _edge_ok(), "human": any(f.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".m4a")
                                                                  for f in (ASSETS / "voices").glob("*")), "silent": True}


# ------------------------------------------------------------------ audio post
def finish_clip(x, out_path):
    """trim silence, normalise to about -20 dBFS RMS with a peak limit, write wav, return duration."""
    x = np.asarray(x, np.float32)
    w = int(0.02 * SR)
    if len(x) > 4 * w:
        rms = np.sqrt(np.convolve(x * x, np.ones(w) / w, mode="same"))
        idx = np.where(rms > 0.004)[0]
        if len(idx):
            a, b = max(0, idx[0] - int(0.04 * SR)), min(len(x), idx[-1] + int(0.08 * SR))
            x = x[a:b]
    r = float(np.sqrt(np.mean(x * x)) + 1e-9)
    x = x * (0.1 / r)
    pk = float(np.abs(x).max())
    if pk > 0.95:
        x = x * (0.95 / pk)
    write_wav(out_path, x)
    return len(x) / SR


# ------------------------------------------------------------------ VOICEVOX
class VoiceVox:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.speakers = requests.get(self.url + "/speakers", timeout=20).json()
        self.assigned = {}
        self.used = set()

    def _styles(self, sp):
        return {st["name"]: st["id"] for st in sp.get("styles", []) if st.get("type", "talk") == "talk"}

    def pick(self, char):
        key = char["id"]
        if key in self.assigned:
            return self.assigned[key]
        pool = (FEMALE if char.get("gender") == "female" else MALE)
        prefs = pool.get(char.get("voice"), pool["narrator"]) + pool["narrator"]
        choice = None
        for pref in prefs:
            for sp in self.speakers:
                if pref in sp["name"] and sp["name"] not in self.used and self._styles(sp):
                    choice = sp
                    break
            if choice:
                break
        if not choice:
            choice = next((sp for sp in self.speakers if sp["name"] not in self.used and self._styles(sp)), self.speakers[0])
        self.used.add(choice["name"])
        self.assigned[key] = choice
        return choice

    def style_id(self, sp, emotion):
        st = self._styles(sp)
        for key in STYLE_KEYS.get(emotion, ()):
            for name, sid_ in st.items():
                if key in name:
                    return sid_
        for name, sid_ in st.items():
            if "ノーマル" in name or "ふつう" in name:
                return sid_
        return next(iter(st.values()))

    def synth(self, text, style_id, emotion, intensity):
        q = requests.post(f"{self.url}/audio_query", params={"text": text, "speaker": style_id}, timeout=60)
        q.raise_for_status()
        q = q.json()
        inten = max(1, min(5, int(intensity or 2)))
        q["speedScale"] = {"whisper": 0.92, "sad": 0.93, "shout": 1.08, "angry": 1.04}.get(emotion, 1.0)
        q["pitchScale"] = {"sad": -0.03, "shout": 0.04, "angry": 0.02, "fear": 0.04, "happy": 0.03}.get(emotion, 0.0)
        q["intonationScale"] = 1.0 + 0.1 * inten
        q["volumeScale"] = 1.0
        q["prePhonemeLength"], q["postPhonemeLength"] = 0.05, 0.12
        r = requests.post(f"{self.url}/synthesis", params={"speaker": style_id}, json=q, timeout=120)
        r.raise_for_status()
        return r.content


def _bytes_to_array(b, tmp):
    p = pathlib.Path(tmp)
    p.write_bytes(b)
    return decode_audio(p, channels=1)


# ------------------------------------------------------------------ edge-tts
def _edge_synth(text, voice, emotion, out_mp3):
    import edge_tts
    rate, pitch = EDGE_PROSODY.get(emotion, ("+0%", "+0Hz"))

    async def go():
        await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(out_mp3))
    asyncio.run(go())


# ------------------------------------------------------------------ main entry
def _human(shot_id):
    d = ASSETS / "voices"
    for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
        p = d / f"{shot_id}{ext}"
        if p.exists():
            return p
    return None


def _silence_len(text):
    return 0.5 + 0.11 * len(re.sub(r"\s", "", text))


def synthesize(script, work, *, provider=None):
    out = {}
    wd = pathlib.Path(work) / "voice"
    wd.mkdir(parents=True, exist_ok=True)
    chars = {c["id"]: c for c in script.get("characters", [])}
    prov = (provider or cfg.voice_provider or "auto").lower()
    vv = None
    if prov in ("auto", "voicevox") and _vv_ok():
        try:
            vv = VoiceVox(cfg.voicevox_url)
            log.info("voice: VOICEVOX engine with %d speakers", len(vv.speakers))
        except Exception as e:  # noqa: BLE001
            log.warning("VOICEVOX unusable: %s", e)
    elif prov == "voicevox":
        raise VoiceError(f"VOICE_PROVIDER=voicevox but no engine answers at {cfg.voicevox_url} "
                         "(run the VOICEVOX engine: docker run -p 50021:50021 voicevox/voicevox_engine:cpu-latest)")
    edge = prov in ("auto", "edge") and _edge_ok()
    silent_used = 0
    for shot in script["shots"]:
        ln = shot.get("line")
        if not ln:
            continue
        sid_ = shot["id"]
        text = ln["ja"]
        ch = chars.get(ln["speaker"], {"id": ln["speaker"], "gender": "male", "voice": "narrator"})
        path = wd / f"line_{sid_:04d}.wav"
        entry = None
        h = _human(sid_)
        try:
            if h:
                dur = finish_clip(decode_audio(h, channels=1), path)
                entry = {"provider": "human", "credit": ""}
            elif vv:
                sp = vv.pick(ch)
                wav = vv.synth(text, vv.style_id(sp, ln["emotion"]), ln["emotion"], ln["intensity"])
                dur = finish_clip(_bytes_to_array(wav, wd / "_tmp.wav"), path)
                entry = {"provider": "voicevox", "credit": f"VOICEVOX:{sp['name']}"}
            elif edge:
                mp3 = wd / f"_tmp_{sid_}.mp3"
                _edge_synth(text, EDGE["female" if ch.get("gender") == "female" else "male"], ln["emotion"], mp3)
                dur = finish_clip(decode_audio(mp3, channels=1), path)
                entry = {"provider": "edge", "credit": ""}
        except Exception as e:  # noqa: BLE001
            log.warning("TTS failed for shot %s (%s) - falling back", sid_, e)
        if entry is None:
            silent_used += 1
            dur = _silence_len(text)
            write_wav(path, np.zeros(int(dur * SR), np.float32))
            entry = {"provider": "silent", "credit": ""}
        out[sid_] = {"path": str(path), "dur": float(dur), "speaker": ln["speaker"], **entry}
    if silent_used:
        log.error("VOICE: %d line(s) are SILENT - no TTS engine was usable. Start VOICEVOX or `pip install edge-tts`.",
                  silent_used)
    return out
