"""Gemini story room: an ORIGINAL dark-fantasy series, written act by act for retention."""
import functools
import json
import re
import time

import requests

from .config import ACTS, EPISODE_MIN, env, gemini_keys, log

URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MOODS = ("embers", "shadow", "rain", "aura", "dust", "snow")
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]          # used only if discovery fails
NOT_TEXT = ("tts", "image", "live", "audio", "embedding", "native", "omni", "cyber", "robotics",
            "computer", "translate", "transcribe", "imagen", "veo")


def _rank(name):
    v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return ("preview" in name or "exp" in name, -(float(v.group(1)) if v else 0), "lite" in name, name)   # stable first, then newest


@functools.lru_cache(1)
def discover_models():
    """Self-healing: ask the API which Flash models THIS key can use today, newest stable first.
    Model names get retired (gemini-2.0/2.5-flash already 404), so nothing is hard-coded except a last resort.
    Override with the GEMINI_MODELS env var (comma separated) if you want to force one."""
    forced = [m.strip() for m in env("GEMINI_MODELS", "").split(",") if m.strip()]
    found = []
    for key in gemini_keys():
        try:
            r = requests.get(LIST_URL, headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
            r.raise_for_status()
            for m in r.json().get("models", []):
                n = m["name"].split("/")[-1]
                if ("generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini-")
                        and "flash" in n and not any(b in n for b in NOT_TEXT)):
                    found.append(n)
            if found:
                break
        except Exception as e:  # noqa: BLE001
            log.warning("ListModels failed: %s", e)
    ordered = forced + [n for n in sorted(set(found), key=_rank) if n not in forced][:5]
    ordered = ordered or FALLBACK_MODELS
    log.info("Gemini models to try: %s", ordered)
    return ordered


def ask(prompt, temperature=1.0):
    """Walk the discovered models and API keys until one answers with valid JSON."""
    last = None
    for model in discover_models():
        for key in gemini_keys():
            try:
                r = requests.post(
                    URL.format(model), headers={"x-goog-api-key": key}, timeout=300,
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": temperature,
                                               "responseMimeType": "application/json",
                                               "maxOutputTokens": 30000}})
                if r.status_code == 404:                       # model retired/unavailable: next model
                    last = f"{model}: 404 {r.text[:160]}"
                    log.warning("model %s not available (404) - trying the next one", model)
                    break
                if r.status_code in (429, 500, 503):
                    last = f"{model}: {r.status_code}"
                    time.sleep(4)
                    continue
                if r.status_code >= 400:
                    last = f"{model}: {r.status_code} {r.text[:200]}"
                    log.warning("Gemini %s -> %s", model, last)
                    continue
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
                return json.loads(txt)
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("Gemini %s failed: %s", model, e)
    raise RuntimeError(f"all Gemini keys/models failed: {last}")


def make_bible():
    prompt = """You are the head writer of a hit Dark Fantasy / Progression anime.
Create a 100% ORIGINAL series. Do NOT reuse names, characters, power systems, plots or settings from any existing
manga, anime, novel or game. Build a hook-driven premise that fans of the genre love: hidden potential, power growth,
dungeons or ruined worlds, betrayal, dark gods, a hero who pays a price for every level.
Return ONLY JSON:
{"title_ja":"","title_ar":"","premise":"","world":"",
 "art_style":"one English line for image prompts: dark fantasy anime key visual, high contrast ink and glow, ...",
 "characters":[{"id":"snake_case","name_ja":"","name_ar":"","gender":"male|female",
   "voice":"hero|villain|elder|young|narrator",
   "look":"FIXED English visual description: hair, eyes, outfit, scars, signature glow colour (max 40 words)"}],
 "arc":["10 one-line episode beats, each ending on a hook"],
 "hashtags":["#..."]}
Use 5 to 6 characters. Stylised fantasy violence only (YouTube ad-friendly: no gore, no sexual content)."""
    return ask(prompt)


SCHEMA = ('{"summary":"2 sentences","scenes":[{"setting":"English visual description of ONE striking image: '
          'location, action, camera angle, lighting. No text or letters in the image",'
          '"mood":"embers|shadow|rain|aura|dust|snow","present":["character ids visible"],"impact":false,'
          '"lines":[{"speaker":"character id or narrator","ja":"Japanese line","ar":"Arabic subtitle",'
          '"intensity":1}]}]}')


def _act(bible, prev_eps, ep, beat, act, carry, target_lines):
    first = act == 1
    last = act == ACTS
    rules = [
        "Short punchy lines (max ~28 Japanese characters). Scenes of 2-6 lines. A new image every scene.",
        "No filler, no recap, no exposition dumps: reveal the world through action and conflict.",
        "Every scene must raise the stakes or reveal something. Keep the viewer guessing until the last second.",
        "Intensity 1-5 per line (5 = the biggest moment, use sparingly: 1-2 per act). Set impact=true on explosive scenes.",
        "Arabic subtitles must be natural, emotional Modern Standard Arabic, not literal word-for-word.",
    ]
    if first:
        rules.append("COLD OPEN: the very first line (within 3 seconds) must be a shocking image, threat or question. "
                     "Start in the middle of danger. Introduce the title only after the hook.")
    rules.append("End this act on a strong cliffhanger." if not last else
                 "End the episode on a twist and a promise that makes the next episode unmissable.")
    prompt = f"""You are writing episode {ep} of this anime, act {act} of {ACTS}.
SERIES BIBLE: {json.dumps(bible, ensure_ascii=False)}
PREVIOUS EPISODES: {json.dumps(prev_eps, ensure_ascii=False)}
THIS EPISODE'S BEAT: {beat}
WHAT HAPPENED EARLIER IN THIS EPISODE: {carry or 'nothing yet - this is the opening'}
Write about {target_lines} spoken lines in total for this act (narrator lines allowed, keep them rare).
Rules:
- """ + "\n- ".join(rules) + f"""
Return ONLY JSON in this shape: {SCHEMA}"""
    return ask(prompt)


def _clean(scenes, ids, act):
    out = []
    for sc in scenes:
        lines = []
        for ln in sc.get("lines", []):
            ja, ar = str(ln.get("ja", "")).strip(), str(ln.get("ar", "")).strip()
            if not ja or not ar:
                continue
            sp = ln.get("speaker", "narrator")
            lines.append({"speaker": sp if sp in ids else "narrator", "ja": ja, "ar": ar,
                          "intensity": max(1, min(5, int(ln.get("intensity", 2) or 2)))})
        if lines and sc.get("setting"):
            out.append({"setting": sc["setting"], "mood": sc.get("mood") if sc.get("mood") in MOODS else "dust",
                        "present": [p for p in sc.get("present", []) if p in ids],
                        "impact": bool(sc.get("impact")), "lines": lines, "act": act})
    return out


def write_episode(bible, state, ep):
    ids = {c["id"] for c in bible["characters"]}
    prev = [{"n": e["n"], "summary": e["summary"]} for e in state["episodes"][-3:]]
    beat = bible["arc"][min(ep - 1, len(bible["arc"]) - 1)]
    target = round(EPISODE_MIN * 60 / ACTS / 4.3)
    scenes, summaries = [], []
    for act in range(1, ACTS + 1):
        for attempt in range(3):
            data = _act(bible, prev, ep, beat, act, " ".join(summaries), target)
            got = _clean(data.get("scenes", []), ids, act)
            n = sum(len(s["lines"]) for s in got)
            if n >= target * 0.6:
                break
            log.warning("act %d too short (%d/%d lines) - rewriting", act, n, target)
        scenes += got
        summaries.append(data.get("summary", ""))
    meta = ask(f"""Episode {ep} of '{bible['title_ja']}' is done. Act summaries: {json.dumps(summaries, ensure_ascii=False)}
Return ONLY JSON: {{"title_ja":"","title_ar":"gripping Arabic title (max 60 chars, curiosity-driven)",
"description_ar":"3-4 line Arabic description with a hook, no spoilers of the twist",
"tags":["10 short tags in Arabic/Japanese/English"]}}""")
    return {"scenes": scenes, "summary": " ".join(summaries), "meta": meta}
