#!/usr/bin/env python3
"""
Channel Factory v2  -  one long documentary (~13 min) + Shorts every day, run like a real channel manager.

Pipeline (long):  trends -> idea -> outline -> chapter-by-chapter script -> per-scene voice (edge-tts)
                  -> stock footage + AI images -> karaoke captions + chapter cards -> music ducking -> FFmpeg
                  -> pro thumbnail (Pillow) -> upload + schedule + playlist + chapters in description
Pipeline (short): ideas -> script -> voice -> footage -> captions -> render -> upload
Manager tasks:    reply to comments, post the first comment, re-title weak videos, smart publish hours, report.

TASK : produce (long + shorts, default) | long | shorts | manage | optimize
MODE : dry (make only) | private (upload private) | now (public now) | publish (private + scheduled public)
All settings come from environment variables (see .github/workflows/daily.yml).
AI writers rotate automatically: Gemini (many keys/models) -> Groq -> Mistral -> OpenRouter -> Cloudflare.
"""
import asyncio
import base64
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests


def env(name, default=""):
    return (os.environ.get(name) or default).strip()


def inum(name, default):
    try:
        return int(float(env(name)))
    except ValueError:
        return default


def fnum(name, default):
    try:
        return float(env(name))
    except ValueError:
        return default


NICHE = env("NICHE", "mind-blowing facts about space, the ocean, animals and nature")
VIDEO_LANG = env("VIDEO_LANG", "en")  # "en" or "ar"
TASK = env("TASK", "produce")
MODE = env("MODE", "publish")
SHORTS_COUNT = inum("SHORTS_PER_DAY", inum("VIDEOS_PER_DAY", 3))
LONG_MINUTES = fnum("LONG_MINUTES", 13.0)
PUBLISH_HOURS = [int(x) for x in env("PUBLISH_HOURS_UTC", "9,12,15,18,21").split(",") if x.strip()]
LONG_HOURS = [int(x) for x in env("LONG_PUBLISH_HOURS_UTC", "15,17,13").split(",") if x.strip()]
CATEGORY = env("CATEGORY_ID", "27")  # 27 Education, 28 Science & Tech, 24 Entertainment
DISCLOSE_AI = env("DISCLOSE_AI", "1") == "1"  # marks the video as containing synthetic media
AI_IMAGES = env("AI_IMAGES", "1") == "1"
MAX_REPLIES = inum("MAX_REPLIES", 8)
PLAYLIST_NAME = env("PLAYLIST_NAME", "Full Documentaries" if VIDEO_LANG == "en" else "الأفلام الوثائقية الكاملة")

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUT = ROOT / "out"
ASSETS = ROOT / "assets"
HISTORY = ROOT / "history.json"
STATE = ROOT / "state.json"
MUSIC_DIR = ROOT / "music"
FPS = 30

PROFILE = {
    "short": {"w": 1080, "h": 1920},
    "long": {"w": 1920, "h": 1080},
}

LANGS = {
    "en": {"name": "English", "voices": ["en-US-AndrewNeural", "en-US-GuyNeural", "en-US-ChristopherNeural"],
           "upper": True, "wpm": 150},
    "ar": {"name": "Modern Standard Arabic (simple, clear, easy to understand)",
           "voices": ["ar-SA-HamedNeural", "ar-EG-ShakirNeural"], "upper": False, "wpm": 120},
}

FORMATS = [
    "Surprising fact with a twist ending",
    "Top 3 mind-blowing facts",
    "Myth vs reality",
    "What would happen if...",
    "Mini story with an unexpected ending",
    "Scale comparison (small things that are actually enormous)",
    "The most extreme example of something",
]

LONG_FORMATS = [
    "Documentary deep-dive that answers one huge question",
    "Top 10 countdown of the most extreme examples",
    "Mystery investigation: what we know, what we don't, what we just discovered",
    "What would happen if... simulated step by step",
    "The untold story behind something everybody thinks they understand",
    "Myths destroyed one by one, each worse than the last",
]


class QuotaStop(Exception):
    pass


# ----------------------------------------------------------------- helpers
def run(cmd, cwd=None):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(map(str, cmd))[:200] + "\n" + p.stderr[-1500:])
    return p.stdout


def _ffmpeg_info(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return p.stderr


def duration(path):
    try:
        out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path])
        return float(out.strip())
    except FileNotFoundError:
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", _ffmpeg_info(path))
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def probe_size(path):
    try:
        out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                   "-of", "csv=p=0:s=x", path]).strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except FileNotFoundError:
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", _ffmpeg_info(path))
        return int(m.group(1)), int(m.group(2))


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt_ts(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# ------------------------------------------------- AI writers (auto-rotating)
GEMINI = "https://generativelanguage.googleapis.com/v1beta"
_gm = {"list": None}
_dead = {}  # (provider/model/key) -> time until which we skip it


def gemini_keys():
    ks = [env("GEMINI_API_KEY")] + [env(f"GEMINI_API_KEY_{i}") for i in range(2, 7)]
    return [k for k in ks if k]


def gemini_models():
    """Model names change often, so we ask the API which ones exist."""
    if env("GEMINI_MODEL"):
        return [m.strip() for m in env("GEMINI_MODEL").split(",") if m.strip()]
    if _gm["list"] is None:
        names = []
        for key in gemini_keys():
            try:
                r = requests.get(f"{GEMINI}/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=30)
                r.raise_for_status()
                for m in r.json().get("models", []):
                    n = m.get("name", "").split("/")[-1]
                    if "generateContent" not in m.get("supportedGenerationMethods", []) or "flash" not in n:
                        continue
                    if any(x in n for x in ("image", "tts", "live", "audio", "embed", "exp", "robotics", "computer")):
                        continue
                    names.append(n)
                break
            except Exception as e:
                print("  gemini model list failed:", repr(e)[:120])

        def rank(n):
            v = re.search(r"(\d+(?:\.\d+)?)", n)
            return ("lite" in n, "preview" in n, -(float(v.group(1)) if v else 0.0))

        _gm["list"] = sorted(set(names), key=rank) or ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
        print("  gemini models:", _gm["list"][:5])
    return _gm["list"]


def llm_gemini(prompt):
    keys = gemini_keys()
    last = "no model"
    for model in gemini_models()[:4]:
        for key in keys:  # every extra key = a separate free quota
            tag = ("gemini", model, key[-6:])
            if _dead.get(tag, 0) > time.time():
                continue
            try:
                r = requests.post(
                    f"{GEMINI}/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 1.0, "responseMimeType": "application/json",
                                               "maxOutputTokens": 16384}},
                    timeout=180)
                if r.status_code == 429:
                    daily = "PerDay" in r.text or "limit: 0" in r.text
                    _dead[tag] = time.time() + (6 * 3600 if daily else 65)
                    last = f"{model}: rate limited ({'daily' if daily else 'minute'})"
                    continue
                if r.status_code >= 400:
                    last = f"{model}: HTTP {r.status_code} {r.text[:120]}"
                    if r.status_code in (400, 403, 404):
                        _dead[tag] = time.time() + 3600
                    continue
                parts = r.json()["candidates"][0]["content"]["parts"]
                obj = parse_json("".join(p.get("text", "") for p in parts))
                time.sleep(3)  # stay under free-tier requests/minute
                return obj
            except Exception as e:
                last = f"{model}: {e!r}"[:160]
    raise RuntimeError("Gemini: " + last)


def _oai(label, url, key, models, prompt, headers=None, json_mode=True):
    """Any OpenAI-compatible chat endpoint (Groq, Mistral, OpenRouter...)."""
    last = "no model"
    for m in models:
        tag = (label, m)
        if _dead.get(tag, 0) > time.time():
            continue
        try:
            body = {"model": m, "temperature": 0.9, "max_tokens": 4096,
                    "messages": [{"role": "system", "content": "Reply with ONLY one valid JSON object. No markdown, no commentary."},
                                 {"role": "user", "content": prompt}]}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(url, headers={"Authorization": "Bearer " + key, **(headers or {})}, json=body, timeout=240)
            if r.status_code == 429:
                _dead[tag] = time.time() + (3 * 3600 if "day" in r.text.lower() else 60)
                last = f"{m}: rate limited"
                continue
            if r.status_code >= 400:
                last = f"{m}: HTTP {r.status_code} {r.text[:120]}"
                if r.status_code in (401, 403, 404):
                    _dead[tag] = time.time() + 1800
                continue
            return parse_json(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            last = f"{m}: {e!r}"[:160]
    raise RuntimeError(f"{label}: {last}")


def llm_groq(prompt):
    models = env("GROQ_MODEL", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",")
    return _oai("groq", "https://api.groq.com/openai/v1/chat/completions", env("GROQ_API_KEY"), models, prompt)


def llm_mistral(prompt):
    models = env("MISTRAL_MODEL", "mistral-small-latest").split(",")
    return _oai("mistral", "https://api.mistral.ai/v1/chat/completions", env("MISTRAL_API_KEY"), models, prompt)


def llm_openrouter(prompt):
    models = env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free,google/gemma-3-27b-it:free,"
                                     "deepseek/deepseek-chat-v3-0324:free").split(",")
    return _oai("openrouter", "https://openrouter.ai/api/v1/chat/completions", env("OPENROUTER_API_KEY"), models,
                prompt, json_mode=False)


def llm_cloudflare(prompt):
    base = f"https://api.cloudflare.com/client/v4/accounts/{env('CF_ACCOUNT_ID')}/ai"
    hdr = {"Authorization": f"Bearer {env('CF_API_TOKEN')}"}
    last = "no model answered"
    for m in ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-3.1-8b-instruct"]:
        try:
            r = requests.post(f"{base}/run/{m}", headers=hdr, timeout=240, json={
                "messages": [{"role": "system", "content": "Reply with ONLY one valid JSON object. No markdown."},
                             {"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": 0.9})
            if r.status_code >= 400:
                last = f"{m}: HTTP {r.status_code} {r.text[:120]}"
                continue
            res = r.json().get("result", {})
            text = res.get("response") if isinstance(res, dict) else None
            if text is None and isinstance(res, dict) and res.get("choices"):
                text = res["choices"][0]["message"]["content"]
            return text if isinstance(text, dict) else parse_json(text or "")
        except Exception as e:
            last = f"{m}: {e!r}"[:160]
    raise RuntimeError("cloudflare: " + last)


PROVIDERS = {
    "gemini": (llm_gemini, lambda: bool(gemini_keys())),
    "groq": (llm_groq, lambda: bool(env("GROQ_API_KEY"))),
    "mistral": (llm_mistral, lambda: bool(env("MISTRAL_API_KEY"))),
    "openrouter": (llm_openrouter, lambda: bool(env("OPENROUTER_API_KEY"))),
    "cloudflare": (llm_cloudflare, lambda: bool(env("CF_ACCOUNT_ID") and env("CF_API_TOKEN"))),
}


def llm_json(prompt):
    """Tries every configured AI in turn; if all are busy it waits and tries again (no more random failures)."""
    order = [p.strip() for p in env("LLM_ORDER", "gemini,groq,mistral,openrouter,cloudflare").split(",")]
    avail = [p for p in order if p in PROVIDERS and PROVIDERS[p][1]()]
    if not avail:
        raise RuntimeError("No AI key configured: add GEMINI_API_KEY (and ideally GROQ_API_KEY too)")
    errs = []
    for rnd in range(3):
        for p in avail:
            try:
                obj = PROVIDERS[p][0](prompt)
                if isinstance(obj, dict):
                    return obj
                errs.append(f"{p}: not an object")
            except Exception as e:
                errs.append(str(e)[:150])
                print(f"  [{p}] {str(e)[:150]}")
        if rnd < 2:
            wait = 45 * (rnd + 1)
            print(f"  every AI is busy, waiting {wait}s then retrying...")
            time.sleep(wait)
    raise RuntimeError("all AI providers failed: " + " | ".join(errs[-4:]))


# ----------------------------------------------------------- trend research
def youtube_suggest(seed):
    try:
        r = requests.get("https://suggestqueries.google.com/complete/search",
                         params={"client": "firefox", "ds": "yt", "q": seed, "hl": VIDEO_LANG}, timeout=15)
        return [s for s in r.json()[1] if isinstance(s, str)]
    except Exception:
        return []


def trending_terms():
    """Real YouTube autocomplete = what people are typing right now."""
    parts = [p.strip() for p in re.split(r",| and |،| و", NICHE) if p.strip()]
    pre = ["why", "how", "what if"] if VIDEO_LANG == "en" else ["لماذا", "كيف", "ماذا لو"]
    out = []
    for part in random.sample(parts, min(3, len(parts))):
        key = part.split()[-1]
        for p in pre:
            out += youtube_suggest(f"{p} {key}")[:6]
            time.sleep(0.3)
    return list(dict.fromkeys(out))[:40]


# --------------------------------------------------------- ideas and scripts
def get_ideas(n, hist, hint, trends, teaser):
    recent = [h["topic"] for h in hist[-60:] if h.get("kind", "short") == "short"]
    extra = f"\n- One idea MUST be a teaser about today's long video topic: {teaser}" if teaser else ""
    prompt = f"""You run a successful YouTube Shorts channel. Niche: {NICHE}.
Language of the videos: {LANGS[VIDEO_LANG]['name']}.
Propose {n} different ideas for today's Shorts. Every idea must use a DIFFERENT format from this list: {json.dumps(FORMATS)}.
Rules:
- Only true, well-established, surprising facts (a fact-checker must agree with every claim).
- No medical, financial, political or religious claims. No violence or gore. Nothing about private people.
- Every idea needs a curiosity hook that makes a stranger stop scrolling within 2 seconds.
- Never repeat or closely resemble these recent topics: {json.dumps(recent, ensure_ascii=False)}
- Real things people search for on YouTube right now (use as inspiration): {json.dumps(trends[:25], ensure_ascii=False)}{extra}
{hint}
Return ONLY JSON: {{"ideas":[{{"topic":"...","format":"...","angle":"the surprising core of the video in one sentence"}}]}}"""
    ideas = llm_json(prompt).get("ideas", [])
    return [i for i in ideas if isinstance(i, dict) and i.get("topic")][:n]


def valid_script(m):
    try:
        sc = m["scenes"]
        if not (5 <= len(sc) <= 9):
            return False
        for s in sc:
            if not s["narration"].strip() or not s["search"].strip():
                return False
        words = len(" ".join(s["narration"] for s in sc).split())
        m["score"] = float(m.get("score", 5))
        return 40 <= words <= 170 and bool(m["title"].strip())
    except Exception:
        return False


def write_script(idea):
    words = "85-105 words" if VIDEO_LANG == "en" else "75-95 words"
    prompt = f"""Write the script of ONE YouTube Short (about 35-45 seconds when read aloud, {words} in total).
Topic: {idea['topic']}
Format: {idea.get('format', '')}
Angle: {idea.get('angle', '')}
Language of narration, title and description: {LANGS[VIDEO_LANG]['name']}.
Rules:
- Scene 1 is a hook that creates curiosity within 3 seconds (a question or a shocking claim). No greetings, no filler.
- Middle scenes: escalating, concrete, visual facts. Short sentences. No fluff.
- Last scene: a punchy payoff that connects back to the hook so people replay the video. Never beg for subscribes.
- Every fact must be true and well established. If you are not sure about a number, leave it out.
- 6 to 8 scenes. Each scene has 1-2 short sentences of narration and "search": 2-3 ENGLISH words for a stock-footage search (something concrete that can be filmed, e.g. "ocean waves", "night sky stars", "close up eye").
Return ONLY JSON:
{{"title":"max 70 chars, curiosity-driven, honest","hook_text":"max 6 words, big on-screen text","scenes":[{{"narration":"...","search":"..."}}],"description":"2 short sentences ending with a question for comments","hashtags":["#shorts","#tag2","#tag3"],"tags":["8 to 12 keywords"],"score":7}}
"score" = your honest rating from 1 to 10 of hook strength AND factual reliability."""
    best = None
    for _ in range(2):
        meta = llm_json(prompt)
        if valid_script(meta):
            if best is None or meta["score"] > best["score"]:
                best = meta
            if meta["score"] >= 7:
                break
    if best is None:
        raise RuntimeError("AI returned no valid script")
    return best


# ---- long video: idea -> outline -> chapters
def long_idea(hist, hint, trends):
    recent = [h["topic"] for h in hist if h.get("kind") == "long"][-40:]
    fmt = random.choice(LONG_FORMATS)
    prompt = f"""You are the head of content strategy of a YouTube channel that must beat the best professional creators in this niche: {NICHE}.
Language: {LANGS[VIDEO_LANG]['name']}.
Real YouTube autocomplete (what people type right now): {json.dumps(trends, ensure_ascii=False)}
{hint}
Never repeat or resemble these recent long videos: {json.dumps(recent, ensure_ascii=False)}
Silently brainstorm 8 ideas for ONE long video of {LONG_MINUTES:.0f} minutes, then return only the single strongest.
The winner needs: a specific curiosity gap, a concrete shocking claim, emotional or survival stakes, proven search demand, and enough real substance to fill {LONG_MINUTES:.0f} minutes with zero padding.
Suggested format: {fmt}
Rules: only true, well-established facts. No medical, financial, political or religious claims. No gore.
Return ONLY JSON: {{"topic":"...","format":"...","angle":"the surprising core in one sentence","viewer_promise":"what the viewer will know or feel at the end"}}"""
    idea = llm_json(prompt)
    if not idea.get("topic"):
        raise RuntimeError("no long idea")
    return idea


def chapter_count():
    return max(6, round(LONG_MINUTES * 0.8))


def total_words():
    return int((LONG_MINUTES * 60 - 25) * LANGS[VIDEO_LANG]["wpm"] / 60 * 1.03)


def plan_long(idea):
    n = chapter_count()
    prompt = f"""Design a {LONG_MINUTES:.0f}-minute YouTube video that beats professional creators on audience retention and click-through.
Topic: {idea['topic']}
Format: {idea.get('format', '')}
Angle: {idea.get('angle', '')}
Viewer promise: {idea.get('viewer_promise', '')}
Language of every text field: {LANGS[VIDEO_LANG]['name']} (image prompts and search words stay in English).
Structure: exactly {n} chapters. Chapter 1 is a COLD OPEN (about 45 seconds): it starts in the middle of the most shocking moment, makes a specific promise and opens a loop that is only closed in the last chapter. The last chapter is the PAYOFF: it closes the loop with a final twist.
Retention rules: no greetings, no "in this video", no channel intro. Each chapter ends with a mini cliffhanger. A pattern interrupt (surprising number, direct question, comparison, reversal) every ~40 seconds. Escalate: every chapter must be more surprising than the previous one. Concrete, visual, specific. Only true, well-established facts.
Title: curiosity-driven, honest, max 70 chars, strong emotion or number, no clickbait lies.
Thumbnail: "thumb_text" = max 4 words in big letters that add a NEW curiosity (never repeat the title). "thumb_prompt" = English prompt for an image: ONE dramatic subject, extreme close-up or epic scale, high contrast, vivid saturated colours, empty space on the left for text, no text, no people's faces.
Return ONLY JSON:
{{"title":"...","title_alts":["...","..."],"thumb_text":"...","thumb_prompt":"...","hook_text":"max 6 words shown on screen in the first seconds","chapters":[{{"title":"max 5 words","goal":"what happens and what the viewer feels","key_points":["concrete fact","concrete fact","concrete fact"]}}],"description":"3 short sentences, curiosity first, no spoilers, ends with a question for the comments","hashtags":["#tag1","#tag2","#tag3"],"tags":["10 to 15 keywords"],"first_comment":"a debate-starting question for the first comment","score":8}}"""
    for _ in range(2):
        plan = llm_json(prompt)
        chs = [c for c in plan.get("chapters", []) if isinstance(c, dict) and c.get("title")]
        if len(chs) >= 5 and plan.get("title"):
            plan["chapters"] = chs
            return plan
    raise RuntimeError("outline failed")


def clean_scenes(raw):
    out = []
    for s in raw if isinstance(raw, list) else []:
        if isinstance(s, dict) and str(s.get("narration", "")).strip() and str(s.get("search", "")).strip():
            out.append({"narration": str(s["narration"]).strip(), "search": str(s["search"]).strip(),
                        "visual": "ai" if s.get("visual") == "ai" and s.get("image_prompt") else "stock",
                        "image_prompt": str(s.get("image_prompt", "")).strip()})
    return out


def write_chapter(plan, i, words, prev_lines, used_search):
    chs = plan["chapters"]
    n = len(chs)
    role = ("COLD OPEN: start in the middle of the most shocking moment of the whole video, state the promise, open a loop. "
            "No greeting, no 'in this video'." if i == 0 else
            "FINALE: close the opening loop, deliver the biggest twist, end on a line that makes the viewer want another video "
            "of ours about a related mystery. Never say subscribe." if i == n - 1 else
            "MIDDLE: more surprising than the previous chapter, end with a mini cliffhanger that pulls into the next chapter.")
    outline = "\n".join(f"{k + 1}. {c['title']} - {c.get('goal', '')}" for k, c in enumerate(chs))
    prompt = f"""You are writing chapter {i + 1} of {n} of a {LONG_MINUTES:.0f}-minute YouTube video titled: {plan['title']}
Language of narration: {LANGS[VIDEO_LANG]['name']}.
Full outline:
{outline}

THIS chapter: "{chs[i]['title']}" - {chs[i].get('goal', '')}
Facts to cover: {json.dumps(chs[i].get('key_points', []), ensure_ascii=False)}
Role: {role}
Last lines of the previous chapter (continue naturally, do not repeat): {json.dumps(prev_lines, ensure_ascii=False)}
Length: about {words} words of narration in total (this matters: not much shorter, not much longer).
Style: the voice of an elite storyteller. Conversational, vivid, concrete numbers, short punchy sentences mixed with longer ones, direct "you" address, one pattern interrupt. Zero filler. Every fact true and well established; if unsure about a number, leave it out.
Scenes: every scene = 1-2 sentences, max 24 words, one new visual idea. For each scene give:
 "search": 2-3 ENGLISH words for a stock-footage search, something concrete that can be filmed. Do NOT reuse these recent searches: {json.dumps(used_search[-30:])}
 "visual": "stock" or "ai". Use "ai" for about 1 scene in 4, only when stock footage cannot show it (inside a black hole, extinct animal, a microscopic scene, a historical moment). For "ai" add "image_prompt": a detailed English cinematic image prompt, no text, no real people's faces.
Return ONLY JSON: {{"scenes":[{{"narration":"...","search":"...","visual":"stock","image_prompt":""}}]}}"""
    best = None
    for _ in range(2):
        sc = clean_scenes(llm_json(prompt).get("scenes"))
        w = sum(len(s["narration"].split()) for s in sc)
        if len(sc) >= 3 and (best is None or abs(w - words) < abs(best[1] - words)):
            best = (sc, w)
        if best and 0.8 * words <= best[1] <= 1.3 * words:
            break
    if not best:
        raise RuntimeError(f"chapter {i + 1} failed")
    return best[0]


def write_long(idea):
    plan = plan_long(idea)
    n = len(plan["chapters"])
    weights = [0.6] + [1.0] * (n - 2) + [0.9]
    tw = total_words()
    scenes, prev, used_search = [], [], []
    for i in range(n):
        words = int(tw * weights[i] / sum(weights))
        print(f"  writing chapter {i + 1}/{n}: {plan['chapters'][i]['title']} (~{words} words)")
        for s in write_chapter(plan, i, words, prev, used_search):
            s["chapter"] = i
            scenes.append(s)
            used_search.append(s["search"])
        prev = [s["narration"] for s in scenes[-2:]]
    plan["scenes"] = scenes
    print(f"  script ready: {len(scenes)} scenes, {sum(len(s['narration'].split()) for s in scenes)} words")
    return plan


# ------------------------------------------------------------------- voice
async def _edge_one(text, voice, rate, out_mp3, sem):
    import edge_tts
    async with sem:
        for attempt in range(3):
            try:
                try:
                    com = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
                except TypeError:  # older edge-tts without the boundary option
                    com = edge_tts.Communicate(text, voice, rate=rate)
                words, sents = [], []
                with open(out_mp3, "wb") as f:
                    async for ch in com.stream():
                        if ch["type"] == "audio":
                            f.write(ch["data"])
                        elif ch["type"] in ("WordBoundary", "SentenceBoundary"):
                            s = ch["offset"] / 1e7
                            item = (s, s + ch["duration"] / 1e7, ch["text"])
                            (words if ch["type"] == "WordBoundary" else sents).append(item)
                if Path(out_mp3).stat().st_size > 1500:
                    return words or sents
            except Exception as e:
                print("  edge-tts retry:", repr(e)[:100])
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


async def _edge_all(texts, voice, rate, outs):
    sem = asyncio.Semaphore(4)
    return await asyncio.gather(*[_edge_one(t, voice, rate, o, sem) for t, o in zip(texts, outs)])


def synth_scenes(texts, wd, rate_pct):
    outs = [wd / f"n{i}.mp3" for i in range(len(texts))]
    voices = ([env("VOICE")] if env("VOICE") else []) + LANGS[VIDEO_LANG]["voices"]
    res = []
    for voice in voices:
        res = asyncio.run(_edge_all(texts, voice, f"{rate_pct:+d}%", outs))
        if sum(r is None for r in res) < len(res) * 0.5:
            break
    bounds = []
    for i, r in enumerate(res):
        if r is None:  # fallback: no word timings, we estimate them
            from gtts import gTTS
            gTTS(texts[i], lang=VIDEO_LANG).save(str(outs[i]))
            r = []
        bounds.append(r)
    return outs, bounds


def weight(tok):
    core = re.sub(r"[^\w]", "", tok, flags=re.U)
    w = max(len(core), 1) + 1.5
    if tok[-1:] in ".!?\u061f\u2026":
        w += 5
    elif tok[-1:] in ",;:\u060c\u061b":
        w += 2.5
    return w


def token_times(text, bounds, total):
    """Returns [(start, end, token)] for one scene (relative to the scene start)."""
    out = []
    if bounds:  # real timings from the TTS engine (word or sentence level)
        for s, e, t in bounds:
            toks = t.split()
            if not toks:
                continue
            ws = [weight(x) for x in toks]
            cur = s
            for x, w in zip(toks, ws):
                d = (e - s) * w / sum(ws)
                out.append((cur, cur + d, x))
                cur += d
        return out
    toks = text.split()
    ws = [weight(x) for x in toks]
    cur, span = 0.05, max(total - 0.25, 0.5)
    for x, w in zip(toks, ws):
        d = span * w / sum(ws)
        out.append((cur, cur + d, x))
        cur += d
    return out


# ---------------------------------------------------------------- captions
def ass_head(kind):
    w, h = PROFILE[kind]["w"], PROFILE[kind]["h"]
    if kind == "short":
        cap = "Style: Cap,DejaVu Sans,88,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,3,2,70,70,560,1"
        hook = "Style: Hook,DejaVu Sans,96,&H0000E5FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,10,4,8,70,70,300,1"
    else:
        cap = "Style: Cap,DejaVu Sans,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,160,160,70,1"
        hook = "Style: Hook,DejaVu Sans,84,&H0000E5FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,3,8,120,120,90,1"
    chap = "Style: Chap,DejaVu Sans,54,&H0000E5FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,7,90,90,70,1"
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{cap}
{hook}
{chap}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t):
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean(t):
    return t.replace("{", "(").replace("}", ")").replace("\\", "")


def build_ass(tokens, hook, chapter_starts, path, kind):
    """Word-by-word highlighted captions (the style that keeps people watching) + chapter title cards."""
    up = LANGS[VIDEO_LANG]["upper"]
    per = 3 if kind == "short" else 4
    chunks, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if len(cur) >= per or tok[2][-1:] in ".!?\u061f,\u060c;:":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    pop = "{\\fscx86\\fscy86\\t(0,110,\\fscx100\\fscy100)}"
    on, off = "{\\1c&H0000E5FF&}", "{\\1c&H00FFFFFF&}"
    lines = [ass_head(kind)]
    h = clean(hook)
    lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(3.2 if kind == 'long' else 2.6)},Hook,,0,0,0,,{{\\fad(0,300)}}{h.upper() if up else h}\n")
    for i, (st, title) in enumerate(chapter_starts):
        if i == 0:
            continue
        t = clean(title)
        lines.append(f"Dialogue: 1,{ass_time(st + 0.2)},{ass_time(st + 3.6)},Chap,,0,0,0,,{{\\fad(350,450)}}{i + 1}  |  {t.upper() if up else t}\n")
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0][0] if ci + 1 < len(chunks) else ch[-1][1] + 0.5
        for k, (s, e, _) in enumerate(ch):
            en = ch[k + 1][0] if k + 1 < len(ch) else min(nxt, ch[-1][1] + 0.5)
            en = max(en, s + 0.08)
            parts = []
            for j, (_, _, w) in enumerate(ch):
                t = clean(w)
                t = t.upper() if up else t
                parts.append(on + t + off if j == k else t)
            lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(en)},Cap,,0,0,0,,{pop if k == 0 else ''}{' '.join(parts)}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ footage
_use_lock = threading.Lock()
_hits_lock = threading.Lock()
_hits = {}


def download(url, dest):
    with requests.get(url, stream=True, timeout=180) as dl:
        dl.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in dl.iter_content(1 << 20):
                fh.write(chunk)


def pixabay_hits(q):
    with _hits_lock:
        if q in _hits:
            return _hits[q]
    hits = []
    for _ in range(3):
        r = requests.get("https://pixabay.com/api/videos/",
                         params={"key": env("PIXABAY_API_KEY"), "q": q[:100], "per_page": 40,
                                 "video_type": "film", "safesearch": "true"}, timeout=30)
        if r.status_code == 429:
            time.sleep(20)
            continue
        if r.status_code == 200:
            hits = r.json().get("hits", [])
        else:
            print(f"  pixabay HTTP {r.status_code}: {r.text[:100]}")
        break
    with _hits_lock:
        _hits[q] = hits
    return hits


def pixabay_variant(h, kind):
    opts = [v for v in (h.get("videos") or {}).values()
            if isinstance(v, dict) and v.get("url") and v.get("width") and v.get("height")]
    wide = [v for v in opts if v["height"] < v["width"] and v["height"] >= 720]
    if kind == "long":
        small = [v for v in wide if (v.get("size") or 0) <= 90e6] or wide
        return min(small, key=lambda v: abs(v["height"] - 1080)) if small else None
    portrait = [v for v in opts if v["height"] >= v["width"] and v["height"] >= 1000]
    if portrait:
        return min(portrait, key=lambda v: abs(v["height"] - 1920))
    if wide:
        small = [v for v in wide if (v.get("size") or 0) <= 110e6] or wide
        return max(small, key=lambda v: v["height"])
    return None


def clip_pixabay(q, dest, used, kind):
    """Free key: pixabay.com/api/docs (log in to see it)."""
    hits = [h for h in pixabay_hits(q) if h.get("duration", 0) >= 4]
    random.shuffle(hits)
    choice = None
    with _use_lock:
        for h in hits:
            if f"pixabay:{h['id']}" in used:
                continue
            v = pixabay_variant(h, kind)
            if v:
                choice = (h, v)
                used.add(f"pixabay:{h['id']}")
                break
    if not choice:
        return None
    download(choice[1]["url"], dest)
    return f"pixabay:{choice[0]['id']}"


def clip_pexels(q, dest, used, kind):
    """Optional: only used if you already own a Pexels key."""
    land = kind == "long"
    for size in ("medium", "small"):
        r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": env("PEXELS_API_KEY")},
                         params={"query": q, "orientation": "landscape" if land else "portrait", "size": size,
                                 "per_page": 20}, timeout=30)
        if r.status_code != 200:
            continue
        vids = [v for v in r.json().get("videos", []) if v.get("duration", 0) >= 4]
        random.shuffle(vids)
        for v in vids:
            with _use_lock:
                if f"pexels:{v['id']}" in used:
                    continue
                files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"
                         and (f.get("height") or 0) >= (720 if land else 1000)
                         and ((f.get("width") or 0) >= (f.get("height") or 1)) == land]
                if not files:
                    continue
                used.add(f"pexels:{v['id']}")
            files.sort(key=lambda f: abs(f["height"] - (1080 if land else 1920)))
            download(files[0]["link"], dest)
            return f"pexels:{v['id']}"
    return None


def abstract_clip(dest, kind):
    """Last resort so a video is still produced when no footage is found."""
    w, h = PROFILE[kind]["w"], PROFILE[kind]["h"]
    c = random.choice([("0x0b132b", "0x3a506b", "0x5bc0be"), ("0x1a1a2e", "0x16213e", "0x0f3460"),
                       ("0x2d1b69", "0x6a4c93", "0xc77dff")])
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"gradients=s={w}x{h}:r={FPS}:d=12:c0={c[0]}:c1={c[1]}:c2={c[2]}:nb_colors=3:speed=0.05",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", dest])
    return "abstract"


def fetch_clip(query, dest, used, kind):
    providers = []
    if env("PIXABAY_API_KEY"):
        providers.append(clip_pixabay)
    if env("PEXELS_API_KEY"):
        providers.append(clip_pexels)
    if not providers:
        raise RuntimeError("No footage source: add the PIXABAY_API_KEY secret")
    words = query.split()
    queries = [query, " ".join(words[:2]), words[0] if words else "",
               random.choice(["nature landscape", "city lights night", "abstract light", "clouds sky timelapse"])]
    for q in dict.fromkeys(x for x in queries if x):
        for prov in providers:
            try:
                cid = prov(q, dest, used, kind)
            except Exception as e:
                print(f"  {prov.__name__} failed for {q!r}: {e!r}"[:200])
                cid = None
            if cid:
                return cid
    print(f"  no footage found for {query!r}, using an abstract background")
    return abstract_clip(dest, kind)


# ---- free AI images (Pollinations first, Cloudflare FLUX as backup)
_img_lock = threading.Lock()
_img = {"fails": 0}
STYLE = ", cinematic photo, ultra detailed, dramatic lighting, rich colours, sharp focus, no text, no watermark, no letters"


def _img_pollinations(prompt, dest, w, h):
    r = requests.get(f"https://image.pollinations.ai/prompt/{quote(prompt + STYLE)}",
                     params={"width": w, "height": h, "nologo": "true", "model": "flux",
                             "seed": random.randint(1, 10**6)}, timeout=120)
    if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 15000:
        Path(dest).write_bytes(r.content)
        return True
    return False


def _img_cloudflare(prompt, dest, w, h):
    if not (env("CF_ACCOUNT_ID") and env("CF_API_TOKEN")):
        return False
    r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{env('CF_ACCOUNT_ID')}/ai/run/@cf/black-forest-labs/flux-1-schnell",
                      headers={"Authorization": f"Bearer {env('CF_API_TOKEN')}"},
                      json={"prompt": prompt + STYLE, "steps": 6}, timeout=120)
    if r.status_code == 200:
        b64 = (r.json().get("result") or {}).get("image")
        if b64:
            Path(dest).write_bytes(base64.b64decode(b64))
            return True
    return False


def ai_image(prompt, dest, w, h):
    if not AI_IMAGES or _img["fails"] >= 4 or not prompt:
        return False
    with _img_lock:  # free services prefer one request at a time
        for fn in (_img_pollinations, _img_cloudflare):
            try:
                if fn(prompt, dest, w, h):
                    _img["fails"] = 0
                    return True
            except Exception as e:
                print(f"  {fn.__name__} failed: {repr(e)[:100]}")
        _img["fails"] += 1
    return False


# ------------------------------------------------------------------- render
def render_scene(src, nframes, out, kind, zoom):
    w, h = PROFILE[kind]["w"], PROFILE[kind]["h"]
    dur = nframes / FPS
    sd = duration(src)
    sw, sh = probe_size(src)
    vf = (f"crop=trunc(min(iw\\,ih*{w}/{h})/2)*2:trunc(min(ih\\,iw*{h}/{w})/2)*2,"
          f"scale={w}:{h}:flags=lanczos")
    if zoom > 1.001:  # alternating punch-in keeps the eye moving
        vf += f",crop=trunc({w}/{zoom:.3f}/2)*2:trunc({h}/{zoom:.3f}/2)*2,scale={w}:{h}:flags=bicubic"
    vf += f",setsar=1,fps={FPS},eq=saturation=1.14:contrast=1.05"
    if sh < (h if kind == "long" else 1500) * 0.9:
        vf += ",unsharp=5:5:0.7"
    if kind == "long":
        vf += ",vignette=PI/7"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if sd < dur + 0.5:
        cmd += ["-stream_loop", "-1"]
    else:
        cmd += ["-ss", f"{random.uniform(0, max(sd - dur - 0.3, 0)):.2f}"]
    cmd += ["-i", src, "-frames:v", str(nframes), "-an", "-vf", vf, "-threads", "2",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "18", "-pix_fmt", "yuv420p", out]
    run(cmd)


def render_image_scene(img, nframes, out, kind):
    """Ken Burns slow zoom on a still image."""
    w, h = PROFILE[kind]["w"], PROFILE[kind]["h"]
    zin = random.random() < 0.6
    z = f"1+0.14*on/{nframes}" if zin else f"1.14-0.14*on/{nframes}"
    px, py = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
    vf = (f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
          f"zoompan=z='{z}':x='(iw-iw/zoom)*{px:.2f}':y='(ih-ih/zoom)*{py:.2f}':d={nframes}:s={w}x{h}:fps={FPS},"
          "setsar=1,eq=saturation=1.1:contrast=1.04")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", img, "-frames:v", str(nframes), "-an", "-vf", vf, "-threads", "2",
         "-c:v", "libx264", "-preset", "superfast", "-crf", "18", "-pix_fmt", "yuv420p", out])


def make_scene(i, s, nframes, wd, used, kind):
    out = wd / f"scene{i}.mp4"
    w, h = PROFILE[kind]["w"], PROFILE[kind]["h"]
    zoom = 1.08 if i % 2 else 1.0
    if s.get("visual") == "ai":
        img = wd / f"img{i}.jpg"
        if ai_image(s.get("image_prompt", ""), img, 1344 if kind == "long" else 768, 768 if kind == "long" else 1344):
            try:
                render_image_scene(str(img), nframes, out, kind)
                img.unlink(missing_ok=True)
                return out, "ai"
            except Exception as e:
                print("  image scene failed:", repr(e)[:150])
    src = wd / f"src{i}.mp4"
    cid = fetch_clip(s["search"], src, used, kind)
    try:
        render_scene(str(src), nframes, out, kind, zoom)
    except Exception as e:
        print("  render failed, using abstract:", repr(e)[:150])
        abstract_clip(src, kind)
        render_scene(str(src), nframes, out, kind, 1.0)
        cid = "abstract"
    src.unlink(missing_ok=True)
    return out, cid


def ambient_track(dest, secs):
    """No music folder? Make a soft cinematic drone so the video is never silent."""
    secs = int(secs) + 3
    ins = []
    for f in (55, 82.41, 110, 164.81):
        ins += ["-f", "lavfi", "-t", str(secs), "-i", f"sine=frequency={f}:sample_rate=22050"]
    run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex",
         "amix=inputs=4:normalize=0,lowpass=f=700,tremolo=f=0.12:d=0.6,volume=0.8", dest])


def compose(scene_files, voice, ass, final, wd, total):
    (wd / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in scene_files))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
         "silent.mp4"], cwd=wd)
    music = []
    if MUSIC_DIR.exists():
        for ext in ("mp3", "wav", "m4a", "ogg"):
            music += sorted(MUSIC_DIR.glob(f"*.{ext}"))
    if music:
        mfile, mvol = random.choice(music).resolve(), 0.30
    else:
        mfile, mvol = wd / "ambient.wav", 0.35
        ambient_track(mfile, total)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", "silent.mp4", "-i", voice.name,
           "-stream_loop", "-1", "-i", str(mfile)]
    audio = (f"[1:a]apad=whole_dur={total:.3f},aresample=44100,asplit=2[v][sc];"
             f"[2:a]aresample=44100,volume={mvol}[m0];"
             "[m0][sc]sidechaincompress=threshold=0.03:ratio=9:attack=20:release=600[m];"  # music ducks under the voice
             "[v][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
             "loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    cmd += ["-filter_complex", f"[0:v]subtitles={ass.name}[vout];{audio}", "-map", "[vout]", "-map", "[a]",
            "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final.resolve()]
    run(cmd, cwd=wd)


# ---------------------------------------------------------------- thumbnail
FONT_URLS = {"en": ("Anton-Regular.ttf", "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"),
             "ar": ("Tajawal-ExtraBold.ttf", "https://github.com/google/fonts/raw/main/ofl/tajawal/Tajawal-ExtraBold.ttf")}
ACCENTS = [(255, 230, 0), (0, 229, 255), (255, 140, 0), (120, 255, 60), (255, 60, 90)]


def get_font(size):
    from PIL import ImageFont
    name, url = FONT_URLS[VIDEO_LANG]
    p = ASSETS / name
    if not p.exists():
        try:
            ASSETS.mkdir(exist_ok=True)
            download(url, p)
        except Exception:
            p.unlink(missing_ok=True)
    for cand in (p, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(str(cand), size)
        except Exception:
            continue
    return ImageFont.load_default()


def shape(text):
    if VIDEO_LANG != "ar":
        return text.upper()
    import arabic_reshaper
    from bidi.algorithm import get_display
    shaped = get_display(arabic_reshaper.reshape(text))
    # Google fonts lack some "isolated form" glyphs (alef, noon...): the plain letter looks identical
    missing = {"\ufe8d": "\u0627", "\ufe83": "\u0623", "\ufe87": "\u0625", "\ufe81": "\u0622", "\ufee5": "\u0646"}
    return "".join(missing.get(c, c) for c in shaped)


def make_thumbnail(text, bg, out):
    """1280x720: punchy colours, dark text side, vignette, huge outlined text with a highlighted last line."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    tw, th = 1280, 720
    img = Image.open(bg).convert("RGB")
    s = max(tw / img.width, th / img.height)
    img = img.resize((int(img.width * s) + 1, int(img.height * s) + 1), Image.LANCZOS)
    l, t = (img.width - tw) // 2, (img.height - th) // 2
    img = img.crop((l, t, l + tw, t + th))
    img = ImageEnhance.Color(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    rtl = VIDEO_LANG == "ar"
    grad = Image.new("L", (tw, th))
    gd = ImageDraw.Draw(grad)
    for x in range(tw):
        d = (tw - x) if rtl else x
        gd.line([(x, 0), (x, th)], fill=int(215 * max(0.0, 1 - d / (tw * 0.65))))
    img = Image.composite(Image.new("RGB", (tw, th), (0, 0, 0)), img, grad)
    vig = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(vig).ellipse((-tw * 0.2, -th * 0.3, tw * 1.2, th * 1.3), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(110))
    img = Image.composite(img, ImageEnhance.Brightness(img).enhance(0.4), vig)
    words = text.split()
    n = len(words)
    k = 1 if n == 1 else 2 if n <= 4 else 3
    per = math.ceil(n / k)
    lines = [" ".join(words[i:i + per]) for i in range(0, n, per)]
    draw = ImageDraw.Draw(img)
    size = 230
    while size > 60:
        font = get_font(size)
        boxes = [draw.textbbox((0, 0), shape(x), font=font, stroke_width=12) for x in lines]
        if max(b[2] - b[0] for b in boxes) <= tw * 0.58 and sum(b[3] - b[1] + 18 for b in boxes) <= th * 0.78:
            break
        size -= 8
    accent = random.choice(ACCENTS)
    total_h = sum(b[3] - b[1] + 18 for b in boxes)
    y = (th - total_h) // 2
    for idx, (line, b) in enumerate(zip(lines, boxes)):
        w_, h_ = b[2] - b[0], b[3] - b[1]
        x = (tw - 60 - w_) if rtl else 60
        fill = accent if idx == len(lines) - 1 else (255, 255, 255)
        draw.text((x - b[0] + 7, y - b[1] + 9), shape(line), font=font, fill=(0, 0, 0), stroke_width=12, stroke_fill=(0, 0, 0))
        draw.text((x - b[0], y - b[1]), shape(line), font=font, fill=fill, stroke_width=12, stroke_fill=(0, 0, 0))
        y += h_ + 18
    draw.rectangle((0, 0, tw - 1, th - 1), outline=accent, width=10)
    q = 92
    while True:
        img.save(out, "JPEG", quality=q, optimize=True)
        if Path(out).stat().st_size < 1_900_000 or q < 60:
            break
        q -= 8
    return out


def thumbnail_for(plan, wd, fallback_scene, name="thumb"):
    bg = wd / f"{name}_bg.jpg"
    ok = ai_image(plan.get("thumb_prompt", ""), bg, 1280, 720)
    if not ok:
        try:
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1.5", "-i", fallback_scene, "-frames:v", "1", "-q:v", "2", bg])
            ok = bg.exists()
        except Exception as e:
            print("  no thumbnail background:", repr(e)[:100])
    if not ok:
        return None
    return make_thumbnail(plan.get("thumb_text") or plan["title"][:30], bg, wd / f"{name}.jpg")


# -------------------------------------------------------------- build video
def build_video(meta, kind, idx, used):
    wd = WORK / f"{kind}{idx}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    scs = meta["scenes"]
    texts = [s["narration"] for s in scs]
    pad = 0.22 if kind == "long" else 0.1
    target = LONG_MINUTES * 60 if kind == "long" else 0
    rate = 0
    for attempt in range(2):
        mp3s, bounds = synth_scenes(texts, wd, rate)
        raw = [duration(p) for p in mp3s]
        frames = [math.ceil((d + pad) * FPS) for d in raw]
        frames[-1] += int(0.6 * FPS)
        total = sum(frames) / FPS
        print(f"  voice ready: {fmt_ts(total)} (rate {rate:+d}%)")
        if not target or attempt:
            break
        ratio = total / target
        new = max(-5, min(12, round(((1 + rate / 100) * ratio - 1) * 100)))
        if abs(ratio - 1) < 0.05 or new == rate:
            break
        rate = new
    wavs = []
    for i, (mp3, nf) in enumerate(zip(mp3s, frames)):  # every scene = a whole number of frames -> no A/V drift
        wav = wd / f"n{i}.wav"
        d = nf / FPS
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-af", f"aresample=44100,apad=whole_dur={d:.6f}",
             "-t", f"{d:.6f}", "-ac", "1", wav])
        wavs.append(wav)
    (wd / "voice.txt").write_text("".join(f"file '{p.name}'\n" for p in wavs))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "voice.txt", "-c", "copy",
         "voice.wav"], cwd=wd)
    workers = 3 if kind == "long" else 2
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(make_scene, i, s, nf, wd, used, kind) for i, (s, nf) in enumerate(zip(scs, frames))]
        results = []
        for i, f in enumerate(futs):
            results.append(f.result())
            if (i + 1) % 20 == 0:
                print(f"  rendered {i + 1}/{len(futs)} scenes")
    scene_files = [r[0] for r in results]
    clip_ids = [r[1] for r in results]
    meta["credits"] = sorted({c.split(":")[0] for c in clip_ids if ":" in c})
    tokens, start, chap_start = [], 0.0, {}
    for i, (s, nf) in enumerate(zip(scs, frames)):
        d = nf / FPS
        if s.get("chapter") is not None and s["chapter"] not in chap_start:
            chap_start[s["chapter"]] = start
        tokens += [(a + start, b + start, t) for a, b, t in token_times(s["narration"], bounds[i], d)]
        start += d
    chapters = [(chap_start[c], meta["chapters"][c]["title"]) for c in sorted(chap_start)] if kind == "long" else []
    ass = wd / "captions.ass"
    build_ass(tokens, meta.get("hook_text") or meta["title"], chapters, ass, kind)
    thumb = None
    if kind == "long":
        try:
            thumb = thumbnail_for(meta, wd, str(scene_files[min(3, len(scene_files) - 1)]))
            if thumb:
                OUT.mkdir(exist_ok=True)
                keep = OUT / f"{dt.date.today():%Y%m%d}_thumb.jpg"
                shutil.copy(thumb, keep)
                thumb = keep
        except Exception:
            traceback.print_exc()
    OUT.mkdir(exist_ok=True)
    final = OUT / f"{dt.date.today():%Y%m%d}_{kind}{idx + 1}.mp4"
    compose(scene_files, wd / "voice.wav", ass, final, wd, total)
    shutil.rmtree(wd, ignore_errors=True)
    return {"path": final, "clips": clip_ids, "thumb": thumb, "chapters": chapters, "seconds": total}


# ------------------------------------------------------------------ YouTube
def youtube():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(None, refresh_token=env("YT_REFRESH_TOKEN"), token_uri="https://oauth2.googleapis.com/token",
                        client_id=env("YT_CLIENT_ID"), client_secret=env("YT_CLIENT_SECRET"))
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def spread(hours, n):
    hours = hours or [12]
    return [hours[int(i * len(hours) / max(n, 1)) % len(hours)] for i in range(n)]


def pick_hours(hist, stats, kind, n, defaults):
    """Learns which publish hours earn the most views per day; keeps 30% exploration."""
    by = {}
    for h in hist:
        v = stats.get(h.get("video_id"))
        if not v or h.get("kind", "short") != kind or not h.get("publish_at"):
            continue
        age = max((dt.date.today() - dt.date.fromisoformat(h["date"])).days, 1)
        by.setdefault(int(h["publish_at"][11:13]), []).append(v["views"] / age)
    ranked = [hr for hr, v in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])) if len(v) >= 2]
    base = spread(defaults, n)
    if not ranked:
        return base
    out = list(dict.fromkeys(ranked[:n]))
    for hr in base:
        if len(out) < n and hr not in out:
            out.append(hr)
    fresh = [d for d in defaults if d not in out[:n]]
    if n > 1 and fresh and random.random() < 0.3:
        out[random.randrange(n)] = random.choice(fresh)
    print(f"  smart publish hours ({kind}): {out[:n]} (learned from {sum(len(v) for v in by.values())} videos)")
    return out[:n]


def publish_slots(hours):
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for i, hr in enumerate(hours):
        t = now.replace(hour=hr, minute=random.choice([0, 6, 11, 17, 24, 33, 41]), second=0, microsecond=0)
        out.append(max(t, now + dt.timedelta(minutes=20 + 12 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return sorted(out)


def build_description(meta, kind, chapters, long_link):
    tags_ = [h if h.startswith("#") else "#" + h for h in meta.get("hashtags", []) if isinstance(h, str)][:3]
    if kind == "short" and not any(h.lower() == "#shorts" for h in tags_):
        tags_.insert(0, "#shorts")
    desc = meta.get("description", "").strip()
    if long_link:
        desc += ("\n\n▶ Watch the full documentary: " if VIDEO_LANG == "en" else "\n\n▶ شاهد الفيلم الوثائقي كاملاً: ") + long_link
    if chapters:
        desc += "\n\n" + ("Chapters" if VIDEO_LANG == "en" else "الفصول") + "\n" + "\n".join(f"{fmt_ts(s)} {t}" for s, t in chapters)
    desc += f"\n\n{' '.join(tags_)}"
    names = {"pixabay": "Pixabay (pixabay.com)", "pexels": "Pexels (pexels.com)"}
    credit = ", ".join(names[c] for c in meta.get("credits", []) if c in names)
    if credit:
        desc += f"\n\nStock footage: {credit}"
    desc += ("\nNarration and some visuals are AI-generated." if VIDEO_LANG == "en"
             else "\nالتعليق الصوتي وبعض الصور مُنشأة بالذكاء الاصطناعي.")
    return desc[:4900]


def upload(yt, path, meta, kind, publish_at, desc):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    tags = [t.strip("# ") for t in meta.get("tags", []) if isinstance(t, str)][:15]
    status = {"privacyStatus": "public" if MODE == "now" else "private", "selfDeclaredMadeForKids": False,
              "embeddable": True}
    if DISCLOSE_AI:
        status["containsSyntheticMedia"] = True  # the "altered or synthetic content" = Yes box
    if MODE == "publish":
        status["publishAt"] = publish_at
    body = {"snippet": {"title": meta["title"].strip()[:99], "description": desc, "tags": tags, "categoryId": CATEGORY,
                        "defaultLanguage": VIDEO_LANG, "defaultAudioLanguage": VIDEO_LANG}, "status": status}
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=32 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp, tries = None, 0
    while resp is None:
        try:
            _, resp = req.next_chunk()
        except HttpError as e:
            text = str(e)
            if "uploadLimitExceeded" in text or "quotaExceeded" in text:
                raise QuotaStop(text[:200])
            if e.resp.status in (500, 502, 503, 504) and tries < 6:
                tries += 1
                time.sleep(2 ** tries)
                continue
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            if tries >= 6:
                raise
            tries += 1
            print("  upload hiccup, retrying:", repr(e)[:100])
            time.sleep(2 ** tries)
    return resp["id"]


def set_thumbnail(yt, vid, path):
    from googleapiclient.http import MediaFileUpload
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(path), mimetype="image/jpeg")).execute()
        return True
    except Exception as e:
        print("  thumbnail not set (channel must be verified for custom thumbnails):", repr(e)[:160])
        return False


def ensure_playlist(yt, state, title):
    pid = state.get("playlist")
    if pid:
        return pid
    for it in yt.playlists().list(part="snippet", mine=True, maxResults=50).execute().get("items", []):
        if it["snippet"]["title"] == title:
            state["playlist"] = it["id"]
            return it["id"]
    r = yt.playlists().insert(part="snippet,status", body={
        "snippet": {"title": title, "description": f"{NICHE}"}, "status": {"privacyStatus": "public"}}).execute()
    state["playlist"] = r["id"]
    return r["id"]


def add_to_playlist(yt, state, vid):
    try:
        pid = ensure_playlist(yt, state, PLAYLIST_NAME)
        yt.playlistItems().insert(part="snippet", body={"snippet": {
            "playlistId": pid, "resourceId": {"kind": "youtube#video", "videoId": vid}}}).execute()
    except Exception as e:
        print("  playlist step skipped:", repr(e)[:140])


# ------------------------------------------------- analytics / the manager
def fetch_stats(yt, entries):
    out = {}
    ids = [h["video_id"] for h in entries if h.get("video_id")]
    for i in range(0, len(ids), 50):
        try:
            r = yt.videos().list(part="statistics,snippet,status", id=",".join(ids[i:i + 50])).execute()
        except Exception as e:
            print("  (stats unavailable:", repr(e)[:100], ")")
            return out
        for it in r.get("items", []):
            s = it.get("statistics", {})
            out[it["id"]] = {"views": int(s.get("viewCount", 0)), "likes": int(s.get("likeCount", 0)),
                             "comments": int(s.get("commentCount", 0)), "snippet": it["snippet"],
                             "privacy": it["status"].get("privacyStatus")}
    return out


def performance_hint(hist, stats, kind):
    cutoff = str(dt.date.today() - dt.timedelta(days=2))
    old = [h for h in hist[-120:] if h.get("kind", "short") == kind and h.get("video_id") in stats
           and h.get("date", "9") <= cutoff]
    if len(old) < (4 if kind == "long" else 8):
        return ""
    ranked = sorted(old, key=lambda h: -stats[h["video_id"]]["views"])
    fmt = lambda h: f'{h["title"]} ({stats[h["video_id"]]["views"]} views, {stats[h["video_id"]]["likes"]} likes)'
    return ("Our best videos so far: " + json.dumps([fmt(h) for h in ranked[:5]], ensure_ascii=False) +
            ". Our weakest: " + json.dumps([fmt(h) for h in ranked[-3:]], ensure_ascii=False) +
            ". Study the pattern (topic type, title style, emotion): make more like the best, fewer like the weakest.")


def my_channel_id(yt, state):
    if not state.get("channel_id"):
        state["channel_id"] = yt.channels().list(part="id", mine=True).execute()["items"][0]["id"]
    return state["channel_id"]


def post_first_comments(yt, hist, stats):
    n = 0
    for h in hist:
        v = stats.get(h.get("video_id"))
        if not h.get("first_comment") or h.get("comment_posted") or not v or v["privacy"] != "public":
            continue
        try:
            yt.commentThreads().insert(part="snippet", body={"snippet": {"videoId": h["video_id"], "topLevelComment": {
                "snippet": {"textOriginal": h["first_comment"][:500]}}}}).execute()
            h["comment_posted"] = True
            n += 1
            print("  first comment posted on", h["video_id"], "(pin it from YouTube Studio for best effect)")
        except Exception as e:
            print("  first comment failed:", repr(e)[:140])
            h["comment_posted"] = True
    return n


def reply_to_comments(yt, hist, state):
    cid = my_channel_id(yt, state)
    titles = {h.get("video_id"): h.get("title", "") for h in hist}
    items = yt.commentThreads().list(part="snippet", allThreadsRelatedToChannelId=cid, maxResults=50, order="time",
                                     textFormat="plainText").execute().get("items", [])
    todo = []
    for it in items:
        sn = it["snippet"]
        top = sn["topLevelComment"]["snippet"]
        if sn.get("totalReplyCount", 0) or top.get("authorChannelId", {}).get("value") == cid:
            continue
        if not str(top.get("textDisplay", "")).strip():
            continue
        todo.append({"id": it["id"], "video": titles.get(sn.get("videoId"), ""), "text": top["textDisplay"][:400]})
    todo = todo[:MAX_REPLIES]
    if not todo:
        return 0
    prompt = f"""You run the YouTube channel about: {NICHE}. Reply to viewer comments as the channel owner.
Rules: answer in the SAME language as each comment; 1-2 short, warm sentences specific to what the person wrote; when natural ask a friendly follow-up question; never promise anything, never post links, never argue. If a comment is spam, hateful, a link, or you cannot reply meaningfully, set "reply" to null.
Comments: {json.dumps(todo, ensure_ascii=False)}
Return ONLY JSON: {{"replies":[{{"id":"...","reply":"... or null"}}]}}"""
    done = 0
    for r in llm_json(prompt).get("replies", []):
        if not isinstance(r, dict) or not r.get("reply") or r.get("id") not in {t["id"] for t in todo}:
            continue
        try:
            yt.comments().insert(part="snippet", body={"snippet": {"parentId": r["id"], "textOriginal": str(r["reply"])[:500]}}).execute()
            done += 1
            time.sleep(random.uniform(2, 6))
        except Exception as e:
            print("  reply failed:", repr(e)[:140])
    return done


def optimize_titles(yt, hist, stats, limit=2):
    """Re-titles (and re-thumbnails) long videos that clearly under-perform after 5+ days. Once per video."""
    today = dt.date.today()
    pool = [h for h in hist if h.get("kind") == "long" and h.get("video_id") in stats and not h.get("optimized")
            and stats[h["video_id"]]["privacy"] == "public"
            and (today - dt.date.fromisoformat(h["date"])).days >= 5]
    if not pool:
        return 0
    allv = sorted(stats[h["video_id"]]["views"] for h in hist if h.get("video_id") in stats and h.get("kind") == "long")
    median = allv[len(allv) // 2] if allv else 0
    pool.sort(key=lambda h: stats[h["video_id"]]["views"])
    done = 0
    for h in pool[:limit]:
        v = stats[h["video_id"]]
        if len(allv) >= 3 and v["views"] > median * 0.5:
            continue
        prompt = f"""A YouTube video is under-performing. Language: {LANGS[VIDEO_LANG]['name']}.
Current title: {h['title']}   Topic: {h['topic']}   Views: {v['views']}
Write 1 new title (max 70 chars, honest, stronger curiosity gap, concrete number or emotion, different angle from the old title) and 1 new thumbnail text (max 4 words).
Return ONLY JSON: {{"title":"...","thumb_text":"..."}}"""
        try:
            new = llm_json(prompt)
            title = str(new["title"]).strip()[:99]
            sn = v["snippet"]
            sn["title"] = title
            yt.videos().update(part="snippet", body={"id": h["video_id"], "snippet": sn}).execute()
            print(f"  re-titled {h['video_id']}: {h['title']!r} -> {title!r}")
            h["old_title"], h["title"], h["optimized"] = h["title"], title, True
            if h.get("thumb_prompt"):
                wd = WORK / "opt"
                wd.mkdir(parents=True, exist_ok=True)
                th = thumbnail_for({"thumb_prompt": h["thumb_prompt"], "thumb_text": new.get("thumb_text") or title[:30],
                                    "title": title}, wd, "", name=h["video_id"])
                if th:
                    set_thumbnail(yt, h["video_id"], th)
            done += 1
        except Exception as e:
            print("  optimize failed:", repr(e)[:160])
            h["optimized"] = True
    return done


def write_report(hist, stats):
    path = env("GITHUB_STEP_SUMMARY")
    rows = [h for h in hist[-30:] if h.get("video_id") in stats][-15:]
    if not path or not rows:
        return
    lines = ["## Channel report\n", "| date | kind | title | views | likes | comments |", "|---|---|---|---|---|---|"]
    for h in reversed(rows):
        s = stats[h["video_id"]]
        lines.append(f"| {h['date']} | {h.get('kind', 'short')} | {h['title'][:60]} | {s['views']} | {s['likes']} | {s['comments']} |")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------- main
def save_all(hist, state):
    if MODE != "dry":
        HISTORY.write_text(json.dumps(hist[-500:], ensure_ascii=False, indent=1), encoding="utf-8")
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def make_long(yt, hist, state, stats, trends, used):
    print("\n##### LONG VIDEO")
    idea = long_idea(hist, performance_hint(hist, stats, "long"), trends)
    print("  idea:", idea["topic"])
    meta = write_long(idea)
    res = build_video(meta, "long", 0, used)
    vid, slot = None, None
    desc = build_description(meta, "long", res["chapters"], None)
    if yt:
        slot = publish_slots(pick_hours(hist, stats, "long", 1, LONG_HOURS))[0]
        vid = upload(yt, res["path"], meta, "long", slot, desc)
        print("  uploaded long:", vid, "| goes public at", slot if MODE == "publish" else MODE)
        if res["thumb"]:
            set_thumbnail(yt, vid, res["thumb"])
        add_to_playlist(yt, state, vid)
    hist.append({"date": str(dt.date.today()), "kind": "long", "topic": idea["topic"], "title": meta["title"],
                 "video_id": vid, "clips": res["clips"], "publish_at": slot, "thumb_prompt": meta.get("thumb_prompt"),
                 "thumb_text": meta.get("thumb_text"), "first_comment": meta.get("first_comment"),
                 "seconds": round(res["seconds"])})
    save_all(hist, state)
    return idea, vid


def make_shorts(yt, hist, state, stats, trends, used, teaser, long_vid):
    print("\n##### SHORTS")
    ideas = get_ideas(SHORTS_COUNT, hist, performance_hint(hist, stats, "short"), trends, teaser)
    print(f"{len(ideas)} short ideas ready, mode={MODE}, lang={VIDEO_LANG}")
    slots = publish_slots(pick_hours(hist, stats, "short", len(ideas), PUBLISH_HOURS))
    link = f"https://youtu.be/{long_vid}" if long_vid else None
    if not link:
        prev = [h for h in hist if h.get("kind") == "long" and h.get("video_id")]
        link = f"https://youtu.be/{prev[-1]['video_id']}" if prev else None
    done = 0
    for i, idea in enumerate(ideas):
        print(f"\n=== short {i + 1}/{len(ideas)}: {idea['topic']}")
        try:
            meta = write_script(idea)
            res = build_video(meta, "short", i, used)
            vid = None
            if yt:
                vid = upload(yt, res["path"], meta, "short", slots[i], build_description(meta, "short", [], link))
                print("  uploaded:", vid, "| goes public at", slots[i] if MODE == "publish" else MODE)
            hist.append({"date": str(dt.date.today()), "kind": "short", "topic": idea["topic"], "title": meta["title"],
                         "video_id": vid, "clips": res["clips"], "publish_at": slots[i] if yt else None})
            save_all(hist, state)
            done += 1
        except QuotaStop as e:
            print("  YouTube daily limit reached, stopping:", e)
            break
        except Exception:
            traceback.print_exc()
    return done


def manage(yt, hist, state):
    print("\n##### MANAGER")
    stats = fetch_stats(yt, hist[-150:])
    print("  first comments:", post_first_comments(yt, hist, stats))
    try:
        print("  replies sent:", reply_to_comments(yt, hist, state))
    except Exception as e:
        print("  replies skipped:", repr(e)[:160])
    print("  titles optimized:", optimize_titles(yt, hist, stats))
    write_report(hist, stats)
    save_all(hist, state)


def main():
    hist = load_json(HISTORY, [])
    state = load_json(STATE, {})
    need_yt = MODE != "dry" or TASK in ("manage", "optimize")
    yt = youtube() if need_yt else None
    if TASK in ("manage", "optimize"):
        if TASK == "optimize":
            stats = fetch_stats(yt, hist[-150:])
            print("titles optimized:", optimize_titles(yt, hist, stats, limit=3))
            save_all(hist, state)
        else:
            manage(yt, hist, state)
        return
    stats = fetch_stats(yt, hist[-150:]) if yt else {}
    trends = trending_terms()
    print(f"trend terms: {len(trends)}")
    used = {c for h in hist[-60:] for c in h.get("clips", [])}
    long_ok, long_vid, teaser, shorts_done = False, None, None, 0
    if TASK in ("produce", "long"):
        try:
            idea, long_vid = make_long(yt, hist, state, stats, trends, used)
            teaser, long_ok = idea["topic"], True
        except QuotaStop as e:
            print("YouTube daily limit reached:", e)
        except Exception:
            traceback.print_exc()
    if TASK in ("produce", "shorts") and SHORTS_COUNT > 0:
        shorts_done = make_shorts(yt, hist, state, stats, trends, used, teaser, long_vid)
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"\nfinished: long={'yes' if long_ok else 'no'}, shorts={shorts_done}")
    if not long_ok and shorts_done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
