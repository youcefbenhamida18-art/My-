"""`python -m mangaforge demo` - an offline end-to-end run (no API keys, no network): canned script, LABELLED placeholder art,
local TTS if present (silence otherwise), procedural score, the real director / renderer / post / subtitles / assembly.
It proves your installation (ffmpeg, fonts, Blender or the cv engine) before you spend any API quota."""
from __future__ import annotations

import json

from . import art as art_pkg
from . import assemble, director, render, schema, voice
from .config import cfg, log, work_dir

CHARS = [
    {"id": "ren", "name_ja": "蓮", "name_ar": "رين", "gender": "male", "voice": "hero",
     "look": "1boy, spiky black hair, red eyes, scar on cheek, black haori with white flame pattern, katana"},
    {"id": "sora", "name_ja": "空", "name_ar": "سورا", "gender": "female", "voice": "girl",
     "look": "1girl, long white hair, blue eyes, white kimono with blue sash, hair ribbon"},
]

RAW = {
    "meta": {"title_ja": "月下の誓い", "title_ar": "عهدٌ تحت القمر", "description_ar": "مشهد تجريبي لاختبار خط الإنتاج.",
             "summary": "Ren meets a spirit at the shrine and vows to protect Sora.", "tags": ["انمي"], "hashtags": ["#انمي"], "part": 1},
    "images": [
        {"id": "i01", "scene": "s1", "prompt": "ruined shrine on a hill at dusk, torii gate, drifting petals, huge moon", "characters": [],
         "framing": "wide", "palette": "orange dusk", "no_humans": True},
        {"id": "i02", "scene": "s1", "prompt": "swordsman standing in a field of grass, wind, moonlight", "characters": ["ren"],
         "framing": "medium", "palette": "cold blue moonlight", "no_humans": False},
        {"id": "i03", "scene": "s2", "prompt": "a spirit girl standing before the shrine, fireflies", "characters": ["sora"],
         "framing": "medium", "palette": "mystic teal", "no_humans": False},
        {"id": "i04", "scene": "s3", "prompt": "dramatic sword slash, shockwave, dark forest ground with stones", "characters": ["ren"],
         "framing": "wide", "palette": "red and black", "no_humans": False},
    ],
    "shots": [
        {"image": "i01", "type": "establishing", "dur": 3.2, "camera": {"move": "pull_out", "strength": 0.6}, "fx": ["petals", "mist", "light_rays"],
         "grade": "dawn", "line": {"speaker": "narrator", "ja": "この世界には、夜に目覚めるものがいる。", "ar": "في هذا العالم، كائناتٌ تستيقظ مع الليل.",
                                   "emotion": "calm", "intensity": 2}, "music": "mystic", "trans_in": "fade"},
        {"image": "i02", "type": "medium", "dur": 3.0, "camera": {"move": "push_in"}, "fx": ["wind_sway", "hair_sway", "dust"], "grade": "night",
         "line": {"speaker": "ren", "ja": "……来る。", "ar": "…إنهم قادمون.", "emotion": "whisper", "intensity": 3}},
        {"image": "i02", "type": "closeup", "dur": 2.4, "camera": {"move": "push_in"}, "fx": ["fireflies"], "line": None},
        {"image": "i03", "type": "medium", "dur": 3.4, "camera": {"move": "orbit_r"}, "fx": ["fireflies", "mist"], "grade": "mystic",
         "line": {"speaker": "sora", "ja": "お願い、行かないで。", "ar": "أرجوك، لا تذهب.", "emotion": "sad", "intensity": 3}},
        {"image": "i03", "type": "closeup", "dur": 2.4, "camera": {"move": "drift"}, "fx": ["fireflies"], "line": None},
        {"image": "i02", "type": "reaction", "dur": 2.2, "camera": {"move": "handheld"}, "fx": ["dust"],
         "line": {"speaker": "ren", "ja": "誓うよ。誰も奪わせない。", "ar": "أُقسم. لن أدع أحدًا يأخذكِ.", "emotion": "angry", "intensity": 4}},
        {"image": "i04", "type": "action", "dur": 1.6, "camera": {"move": "whip_r"}, "fx": ["speed_lines", "sparks"], "trans_in": "whip", "sfx": ["whoosh"], "line": None},
        {"image": "i04", "type": "impact", "dur": 1.4, "camera": {"move": "punch_in"}, "fx": ["slash", "shockwave", "debris"], "sfx": ["slash", "boom"],
         "line": {"speaker": "ren", "ja": "はあああっ！", "ar": "هاااا!", "emotion": "shout", "intensity": 5}},
        {"image": "i04", "type": "impact", "dur": 1.2, "camera": {"move": "handheld"}, "fx": ["lightning", "sparks"], "line": None},
        {"image": "i03", "type": "closeup", "dur": 2.6, "camera": {"move": "push_in"}, "fx": ["petals", "glow_pulse"], "grade": "warm",
         "line": {"speaker": "sora", "ja": "……ありがとう。", "ar": "…شكرًا لك.", "emotion": "tender", "intensity": 2}, "music": "warm"},
        {"image": "i01", "type": "wide", "dur": 3.0, "camera": {"move": "crane_up"}, "fx": ["petals", "light_rays", "mist"], "grade": "dawn", "line": None},
    ],
}


def build_script():
    return schema.normalize_script(json.loads(json.dumps(RAW)), characters=CHARS)


def run(engine=None, seconds=30.0):
    job = "demo"
    cfg.allow_placeholder_art = True
    cfg.image_provider = "none"
    cfg.art_target_w = min(cfg.art_target_w, 2560)
    W = work_dir(job)
    script = build_script()
    (W / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=1), "utf-8")
    voices = voice.synthesize(script, W)
    (W / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=1), "utf-8")
    arts = art_pkg.generate_art(script, W)
    shots = director.plan(script, voices, arts, job, target_seconds=seconds)
    res = render.render_shots(shots, engine=engine)
    out = assemble.assemble_short(shots, script, job, voices=voices)
    out["engines"] = sorted({r["engine"] for r in res.values()})
    log.info("DEMO READY: %s (%.1fs, engines: %s)", out["video"], out["duration"], out["engines"])
    return out
