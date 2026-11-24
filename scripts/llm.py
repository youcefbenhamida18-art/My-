"""Gemini (REST) client + the prompts that turn a lecture into a deep whiteboard lesson script."""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from common import CACHE, env, log, sha
from sources import Lesson, LessonFile, IMAGE_MIME, catalog, fingerprint

BASE = "https://generativelanguage.googleapis.com"
# Preference order (all have a free tier as of Nov 2026). Override with the GEMINI_MODEL env var.
PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
SAFETY = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]

COLORS = {"white", "yellow", "blue", "green", "pink", "orange"}
ITEM_TYPES = {"heading", "text", "bullet", "math", "box", "ar", "flow", "table", "divider", "page_image"}


# ============================================================================ prompts
SYSTEM = """You are an exceptional university tutor who records a PRIVATE whiteboard video lesson for ONE student:
a 2nd-year pharmacy student in Algeria. Arabic is the student's first language; the university course material is
usually in French and the exams are in French. Your goal: after the video the student UNDERSTANDS every idea from
zero AND can RECALL it in the exam.

TEACHING METHOD
1. Explain like to a smart 8-year-old first (an everyday analogy: kitchen, football, traffic, keys and locks, a
   post office...), THEN give the precise scientific version. Never assume prerequisites: define every term the
   first time it appears.
2. Always say WHY, not only WHAT: link each fact to a cause, a purpose or a consequence.
3. Go slowly, one idea per step. Repeat the key idea in different words. Use contrast ("not X, but Y").
4. Worked examples: solve every numerical example completely on the board: given data -> formula -> substitution
   with units -> calculation (no skipped arithmetic) -> final answer in a box -> quick sanity check.
5. If past exercises (TD / exams) are provided, use them as the worked examples: restate the exercise briefly,
   solve it step by step, say what the examiner tests and the typical traps. Never invent exercises that look
   like they come from the professor; if you create an extra practice example, label it "Extra practice".
6. Memory: for every section give one memory hook (acronym, rhyme, story, mental picture, chunking into 3) and a
   one-board recap. Use active recall: ask a question, pause a few seconds, then answer.
7. Be FAITHFUL to the lecture: use its definitions, numbers, names, classifications exactly. Never invent facts,
   values or drug data. Extra background is allowed only if it is correct and simple. If a page is unreadable or
   ambiguous, say so briefly instead of guessing.
8. Language: mostly English plus short Arabic clarifications for the hardest ideas (about 15-20% of the speech).
   On the board, write the lecture's original (French) term in brackets next to the English term, e.g.
   "Glycolysis (la glycolyse)", because the exam uses it.

SPOKEN STYLE (the "say" field)
- A warm, calm, encouraging teacher talking to one student. Short sentences. Natural words like "Okay," "Look,"
  "Now watch this." Refer to what you write ("see this arrow here").
- Plain text only: no markdown, no emojis, no LaTeX, no symbols. Say numbers and formulas in words:
  "H two O", "ten to the power minus three", "delta G", "mole per liter".
- Every "say" part is in ONE language: lang "en", or lang "ar" (Arabic script only; write technical words in
  Arabic letters, e.g. جلوكوز; simple, friendly Modern Standard Arabic).

OUTPUT: valid JSON only. Inside JSON strings every backslash must be doubled (LaTeX: "\\\\frac{a}{b}"), and never put
line breaks inside a string."""

BOARD_SPEC = """BOARD ITEMS (what you write on the whiteboard; short lines, like a real teacher):
{"type":"heading","text":"1. What is X?","color":"yellow"}                      big underlined title line
{"type":"text","text":"Glycolysis (la glycolyse) = breaking glucose","color":"white","indent":0}
{"type":"bullet","text":"occurs in the cytosol","color":"blue","indent":1}
{"type":"math","text":"\\\\Delta G = \\\\Delta H - T\\\\Delta S","color":"green"}          centered formula
{"type":"box","text":"KEY: ATP is the energy coin of the cell","color":"yellow"}  framed key point
{"type":"ar","text":"يعني ببساطة: الإنزيم مثل المفتاح","color":"green"}               short Arabic helper, right aligned
{"type":"flow","items":["Glucose","G6P","F6P"],"direction":"horizontal","color":"blue"}   boxes + arrows (each item <= 3 words)
{"type":"table","rows":["Drug | Target | Effect","Aspirin | COX | less pain"],"color":"white"}   first row is the header
{"type":"divider"}
{"type":"page_image","file":1,"page":12,"caption":"Figure from your lecture, p.12"}   show a page (or a region) of an input file; optional "crop":[ymin,xmin,ymax,xmax] in 0-1000. Use only when a figure/structure/graph is essential.
colors: white (default), yellow (titles / key words), blue (definitions), green (results / answers / Arabic), pink (warnings / exam traps), orange (examples).
Inline math is allowed in text lines with $...$ and uses matplotlib-mathtext: \\\\frac{a}{b}, x_{1}, x^{2}, \\\\alpha, \\\\Delta, \\\\times, \\\\cdot, \\\\pm, \\\\rightarrow, \\\\rightleftharpoons, \\\\leq, \\\\geq, \\\\approx, \\\\sqrt{x}, \\\\log, \\\\ln, \\\\mu. Write chemistry and units with \\\\mathrm{}, e.g. $\\\\mathrm{H_2O}$, $\\\\mathrm{Ca^{2+}}$, $\\\\mathrm{mol\\\\ L^{-1}}$. Do NOT use \\\\text, \\\\begin, \\\\cancel, \\\\overset.
Rules: a line of text <= 12 words; at most 3 board items per step; at most about 9 lines per board (the board is cleared automatically when full)."""

STEP_EXAMPLE = """{"steps":[
 {"new_board":true,"board_title":"What is glycolysis?",
  "say":[{"lang":"en","text":"Okay, imagine your body is a car and glucose is the fuel. Glycolysis is the very first thing the cell does to get energy out of that fuel."},
         {"lang":"ar","text":"ببساطة: الجليكوليز هو أول مرحلة تحرق بها الخلية السكر لتأخذ منه الطاقة."}],
  "board":[{"type":"heading","text":"Glycolysis (la glycolyse)","color":"yellow"},
           {"type":"text","text":"glucose  ->  2 pyruvate  +  a little ATP","color":"white"}],
  "pause":0},
 {"say":[{"lang":"en","text":"Quick check. Where in the cell do you think this happens? Pause the video and answer."}],
  "board":[{"type":"box","text":"Quick check: where does glycolysis happen?","color":"pink"}],"pause":6}
],
"recap":"Glycolysis turns one glucose into two pyruvate in the cytosol and makes a little ATP."}"""


def outline_prompt(lesson: Lesson, max_total: int, test: bool) -> str:
    return f"""Below are the student's files.
{catalog(lesson)}

Student's notes for this lesson: {lesson.notes or "(none)"}
Lesson folder name: {lesson.title_hint}

TASK: plan ONE deep teaching video (it may later be cut into parts) that teaches ALL of the LECTURE material from zero.
Use the PAST TD / EXAM files (if any) as the exercises to solve; assign each relevant exercise to the section where
the needed theory has just been taught.

Choose the total length from the amount of material: about 2-3 minutes of video per lecture page / slide (more for
calculation-heavy or dense pages), between 10 and {max_total} minutes in total.{" (QUICK TEST: plan only 3 short sections of 2 minutes.)" if test else ""}
Cover EVERYTHING in the lecture, in a logical order from the simplest prerequisite to the hardest idea.
First section = "Big picture: what is this and why does it matter" (hook, learning goals, roadmap).
Last section = "Final recap, memory tricks and self-test" (everything on one cheat-sheet board + quiz).
Each normal section = 3 to 6 minutes of video about ONE coherent topic.

Return JSON exactly in this shape:
{{"video_title":"<=90 chars, English, include course + topic",
 "course":"...", "topic":"...", "language_of_material":"French|English|Arabic|mixed",
 "summary":"2-3 sentences for the YouTube description",
 "sections":[{{"title":"...","goal":"what the student can do after it","covers":"pages / headings of the lecture",
   "key_facts":["exact definitions, numbers, names, formulas the section must teach"],
   "td_to_solve":["which TD/exam exercises to solve here, e.g. 'TD 2023 ex.2: pH of a buffer'"],"minutes":4}}]}}"""


def section_prompt(outline: dict, idx: int, part_no: int, parts_total: int, sec: dict, prev_recap: str,
                   is_last_overall: bool, is_last_in_part: bool, is_first_in_part: bool, lesson: Lesson) -> str:
    titles = "\n".join(f"{i + 1}. {s.get('title', '')}" for i, s in enumerate(outline["sections"]))
    minutes = max(2, min(8, int(sec.get("minutes", 4) or 4)))
    words = minutes * 150
    if idx == 0 and part_no == 1:
        opening = "This is the very beginning: friendly hello, say what the whole video covers and why it matters (a short hook), then the roadmap."
    elif is_first_in_part:
        opening = "This starts a new video part: begin with a 20-second 'welcome back' recap of the previous part (use the recap below), then continue."
    else:
        opening = "Continue naturally from the previous section with ONE short bridge sentence. Do not greet again."
    if is_last_overall:
        closing = "This is the FINAL section: build one cheat-sheet board of the whole lesson (short lines), then 5 self-test questions (ask, pause, answer), then an encouraging goodbye."
    elif is_last_in_part:
        closing = "End with a 2-sentence preview of what comes in the next part of the video."
    else:
        closing = "End with the memory hook and a one-board recap of this section."
    return f"""{BOARD_SPEC}

Example of the output format (content is only an illustration):
{STEP_EXAMPLE}

NOW WRITE SECTION {idx + 1} of {len(outline['sections'])} (video part {part_no} of {parts_total}).
VIDEO TITLE: {outline.get('video_title', '')}
WHOLE OUTLINE:
{titles}

THIS SECTION (spec): {json.dumps(sec, ensure_ascii=False)}
PREVIOUS RECAP (what the student already learned): {prev_recap or '(nothing yet)'}
Student's notes: {lesson.notes or '(none)'}

{opening}
{closing}
Teach it deeply and slowly: analogy first, then precise definition, then why, then a worked example solved step by
step, then a quick check with a pause. Solve every exercise listed in "td_to_solve" completely (read it from the
attached TD/exam files).
Length: about {words} words of speech in total (about {minutes} minutes); each step has 25-70 spoken words.
The first step of the section must have "new_board": true and a short "board_title" (<= 6 words). Start a new board
whenever the topic changes. A step may have an empty "board" when you only explain aloud.
Return JSON: {{"steps":[...], "recap":"2-3 sentences: what the student learned in this section"}}"""


# ============================================================================ JSON helpers
_LATEX_B = re.compile(r"(?<!\\)\\(?=(beta|bar|begin|frac|forall|neq|nabla|nu|rho|rightarrow|rightleftharpoons|"
                      r"right|text|theta|times|tau|to|triangle|tilde|therefore)\b)")
_BAD_ESC = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')


def parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b != -1:
        t = t[a:b + 1]
    t = _LATEX_B.sub(r"\\\\", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    t2 = _BAD_ESC.sub(r"\\\\", t)
    return json.loads(t2)


def _clean(s, n=400) -> str:
    s = re.sub(r"[\x00-\x08\x0b-\x1f]", "", str(s or ""))
    s = s.replace("\t", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()[:n]


@dataclass
class Step:
    say: list  # [(lang, text)]
    board: list  # normalized item dicts
    new_board: bool = False
    board_title: str = ""
    pause: float = 0.0
    section: int = 0
    section_title: str = ""


def normalize_steps(raw_steps, section_idx: int, section_title: str) -> list:
    out = []
    for rs in raw_steps or []:
        if not isinstance(rs, dict):
            continue
        say = []
        for p in rs.get("say") or []:
            if isinstance(p, dict):
                lang = "ar" if str(p.get("lang", "en")).lower().startswith("ar") else "en"
                txt = _clean(p.get("text"), 900)
                if txt:
                    say.append((lang, txt))
            elif isinstance(p, str) and p.strip():
                say.append(("en", _clean(p, 900)))
        if not say:
            continue
        board = []
        for it in (rs.get("board") or [])[:4]:
            if not isinstance(it, dict):
                continue
            t = str(it.get("type", "text")).lower()
            if t not in ITEM_TYPES:
                t = "text"
            d = {"type": t, "color": it.get("color") if it.get("color") in COLORS else None}
            if t in ("heading", "text", "bullet", "math", "box", "ar"):
                d["text"] = _clean(it.get("text"), 260)
                d["indent"] = max(0, min(3, int(it.get("indent", 0) or 0))) if str(it.get("indent", 0)).lstrip("-").isdigit() else 0
                if not d["text"]:
                    continue
            elif t == "flow":
                d["items"] = [_clean(x, 40) for x in (it.get("items") or [])][:8]
                d["direction"] = "vertical" if str(it.get("direction", "")).startswith("v") else "horizontal"
                if len(d["items"]) < 2:
                    continue
            elif t == "table":
                d["rows"] = [_clean(x, 200) for x in (it.get("rows") or [])][:9]
                if len(d["rows"]) < 2:
                    continue
            elif t == "page_image":
                try:
                    d["file"], d["page"] = int(it.get("file", 1)), int(it.get("page", 1))
                except Exception:
                    continue
                d["caption"] = _clean(it.get("caption"), 120)
                crop = it.get("crop")
                if isinstance(crop, list) and len(crop) == 4:
                    try:
                        d["crop"] = [max(0, min(1000, float(x))) for x in crop]
                    except Exception:
                        pass
            board.append(d)
        try:
            pause = max(0.0, min(15.0, float(rs.get("pause", 0) or 0)))
        except Exception:
            pause = 0.0
        out.append(Step(say, board, bool(rs.get("new_board")), _clean(rs.get("board_title"), 70), pause,
                        section_idx, section_title))
    if out:
        out[0].new_board = True
        if not out[0].board_title:
            out[0].board_title = section_title[:70]
    return out


# ============================================================================ client
class Gemini:
    def __init__(self, key: str):
        self.key = key
        self.s = requests.Session()
        self.s.headers.update({"x-goog-api-key": key})
        self.min_gap = env("LLM_MIN_GAP", 5.0, float)
        self._last = 0.0
        self._files = {}
        self.models = self._pick_models()
        self.mi = 0
        log("Gemini models to try:", self.models)

    # ---- models
    def _pick_models(self) -> list:
        forced = env("GEMINI_MODEL")
        avail = None
        try:
            names, token = [], None
            for _ in range(6):
                params = {"pageSize": 200}
                if token:
                    params["pageToken"] = token
                r = self.s.get(f"{BASE}/v1beta/models", params=params, timeout=30)
                if r.status_code in (400, 401, 403):
                    raise SystemExit(f"Gemini API key rejected ({r.status_code}): {r.text[:200]}")
                r.raise_for_status()
                j = r.json()
                names += [m["name"].split("/", 1)[1] for m in j.get("models", [])
                          if "generateContent" in m.get("supportedGenerationMethods", [])]
                token = j.get("nextPageToken")
                if not token:
                    break
            avail = set(names)
        except SystemExit:
            raise
        except Exception as e:
            log("  could not list models:", e)
        cands = list(PREFERRED_MODELS)
        if avail:
            cands = [m for m in PREFERRED_MODELS if m in avail]
            if not cands:
                flash = [n for n in avail if re.fullmatch(r"gemini-\d+(?:\.\d+)?-flash", n)]
                flash.sort(key=lambda n: float(re.match(r"gemini-(\d+(?:\.\d+)?)", n).group(1)), reverse=True)
                cands = flash or list(PREFERRED_MODELS)
        if forced:
            cands = [forced] + [m for m in cands if m != forced]
        return cands

    def _next_model(self, why: str):
        log(f"  model {self.models[self.mi]} unusable ({why})")
        self.mi += 1
        if self.mi >= len(self.models):
            raise RuntimeError("No usable Gemini model. Check the API key / quota, or set GEMINI_MODEL.")

    # ---- files
    def upload(self, path: Path, mime: str, fp: str) -> dict:
        if fp in self._files:
            return self._files[fp]
        size = path.stat().st_size
        for attempt in range(1, 5):
            try:
                st = self.s.post(f"{BASE}/upload/v1beta/files", headers={
                    "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(size),
                    "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": "application/json"},
                    json={"file": {"display_name": f"lesson-{fp[:8]}"}}, timeout=60)
                st.raise_for_status()
                url = st.headers["x-goog-upload-url"]
                with open(path, "rb") as fh:
                    up = self.s.post(url, headers={"X-Goog-Upload-Offset": "0",
                                                   "X-Goog-Upload-Command": "upload, finalize"},
                                     data=fh, timeout=(30, 900))
                up.raise_for_status()
                info = up.json()["file"]
                for _ in range(90):  # wait until ACTIVE
                    state = info.get("state", "ACTIVE")
                    if state == "ACTIVE":
                        break
                    if state == "FAILED":
                        raise RuntimeError("file processing failed")
                    time.sleep(3)
                    info = self.s.get(f"{BASE}/v1beta/{info['name']}", timeout=30).json()
                self._files[fp] = {"uri": info["uri"], "mime": info.get("mimeType", mime)}
                return self._files[fp]
            except Exception as e:
                log(f"  upload attempt {attempt} failed: {e}")
                time.sleep(5 * attempt)
        raise RuntimeError(f"Could not upload {path.name} to Gemini")

    def file_parts(self, lesson: Lesson) -> list:
        parts = []
        for i, f in enumerate(lesson.files, 1):
            role = "LECTURE" if f.role == "lecture" else "PAST TD / EXAM"
            head = f"=== FILE {i}: {f.name} ({role}) ==="
            if f.kind == "pdf":
                u = self.upload(f.pdf, "application/pdf", f.fp)
                parts += [{"text": head}, {"fileData": {"mimeType": u["mime"], "fileUri": u["uri"]}}]
            elif f.kind == "image":
                u = self.upload(f.image, IMAGE_MIME[f.image.suffix.lower()], f.fp)
                parts += [{"text": head}, {"fileData": {"mimeType": u["mime"], "fileUri": u["uri"]}}]
            else:
                parts.append({"text": head + "\n" + f.text[:250000]})
        return parts

    # ---- generation
    def _pace(self):
        wait = self.min_gap - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    @staticmethod
    def _extract(j: dict):
        cands = j.get("candidates") or []
        if not cands:
            log("  no candidates:", json.dumps(j.get("promptFeedback", {}))[:300])
            return None
        c = cands[0]
        text = "".join(p.get("text", "") for p in (c.get("content", {}).get("parts") or []) if not p.get("thought"))
        if c.get("finishReason") == "MAX_TOKENS":
            log("  response truncated (MAX_TOKENS)")
            return None
        return text or None

    def call_json(self, lesson: Lesson, prompt: str, label: str):
        """Send files + prompt, return parsed JSON. Cached on disk (survives re-runs)."""
        cache_file = CACHE / "llm" / f"{sha(label, SYSTEM, prompt, fingerprint(lesson))}.json"
        if cache_file.exists():
            try:
                data = parse_json(cache_file.read_text(encoding="utf-8"))
                log(f"  [cache] {label}")
                return data
            except Exception:
                cache_file.unlink(missing_ok=True)
        parts = self.file_parts(lesson) + [{"text": prompt}]
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseMimeType": "application/json"},
                "safetySettings": SAFETY}
        fails = 0
        while True:
            model = self.models[self.mi]
            self._pace()
            log(f"  Gemini [{model}] {label} ...")
            try:
                r = self.s.post(f"{BASE}/v1beta/models/{model}:generateContent", json=body, timeout=(30, 1200))
            except requests.RequestException as e:
                fails += 1
                if fails > 6:
                    raise
                log(f"  network error ({e}); retry in {15 * fails}s")
                time.sleep(15 * fails)
                continue
            if r.status_code == 200:
                text = self._extract(r.json())
                if text:
                    try:
                        data = parse_json(text)
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                        return data
                    except Exception as e:
                        log(f"  JSON parse failed: {e}")
                fails += 1
                if fails > 4:
                    raise RuntimeError(f"Gemini returned unusable output for {label}")
                time.sleep(10)
                continue
            code, msg = r.status_code, r.text[:800]
            if code in (401, 403):
                if "quota" in msg.lower() or "free tier" in msg.lower() or "not available" in msg.lower():
                    self._next_model(f"{code} {msg[:120]}")
                    continue
                raise RuntimeError(f"Gemini rejected the request ({code}): {msg}")
            if code == 404 or (code == 400 and re.search(r"not found|not supported|is not available", msg, re.I)):
                self._next_model(f"{code}")
                continue
            if code == 429:
                if "limit: 0" in msg or "PerDay" in msg or "per day" in msg.lower():
                    self._next_model("daily/free quota exhausted for this model")
                    continue
                m = re.search(r'retry(?:Delay)?\W+(?:in\s+)?"?([\d.]+)\s*s', msg, re.I)
                wait = min(120.0, max(float(m.group(1)) + 3 if m else 0, 25.0))
                fails += 1
                if fails > 10:
                    raise RuntimeError("Gemini rate limit keeps failing: " + msg[:300])
                log(f"  rate limited; waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if code >= 500:
                fails += 1
                if fails > 6:
                    raise RuntimeError(f"Gemini server error {code}: {msg[:300]}")
                time.sleep(min(20 * fails, 120))
                continue
            raise RuntimeError(f"Gemini error {code}: {msg}")

    # ---- high level
    def outline(self, lesson: Lesson, max_total: int, test: bool) -> dict:
        data = self.call_json(lesson, outline_prompt(lesson, max_total, test), "outline")
        secs = []
        for s in data.get("sections") or []:
            if isinstance(s, dict) and _clean(s.get("title")):
                s["title"] = _clean(s["title"], 80)
                try:
                    s["minutes"] = max(2, min(8, int(float(s.get("minutes", 4)))))
                except Exception:
                    s["minutes"] = 4
                secs.append(s)
        if not secs:
            raise RuntimeError("Outline has no sections")
        data["sections"] = secs
        data["video_title"] = _clean(data.get("video_title") or lesson.title_hint, 90)
        data["summary"] = _clean(data.get("summary"), 600)
        return data

    def section_steps(self, lesson, outline, idx, part_no, parts_total, prev_recap, is_last_overall,
                      is_last_in_part, is_first_in_part):
        sec = outline["sections"][idx]
        prompt = section_prompt(outline, idx, part_no, parts_total, sec, prev_recap, is_last_overall,
                                is_last_in_part, is_first_in_part, lesson)
        data = self.call_json(lesson, prompt, f"section-{idx + 1}-{sha(sec.get('title'))[:6]}")
        steps = normalize_steps(data.get("steps"), idx, sec["title"])
        if not steps:
            raise RuntimeError(f"Section {idx + 1} produced no steps")
        return steps, _clean(data.get("recap"), 500)


def split_parts(outline: dict, part_minutes: int) -> list:
    """Group section indexes into video parts of about part_minutes each."""
    parts, cur, acc = [], [], 0
    for i, s in enumerate(outline["sections"]):
        m = s["minutes"]
        if cur and acc + m > part_minutes:
            parts.append(cur)
            cur, acc = [], 0
        cur.append(i)
        acc += m
    if cur:
        if parts and acc < 0.4 * part_minutes:
            parts[-1] += cur
        else:
            parts.append(cur)
    return parts
