"""Story engine (original stories; manga/manhwa adaptation lives in manga.py).

    make_bible()                 -> series bible (premise, world, characters with booru-style `look` tags, style)
    write_week_outline(bible, st)-> 7 beats (one per daily short; part 7 = the episode climax + hook)
    write_short(bible, st, part) -> normalised `script.json` dict for ONE ~2-minute short (shot-level direction)

The LLM only supplies creative content; structure is enforced by `schema.normalize_script` and by the duration/lines/images
checks below (one corrective retry with the exact problems listed).
"""
from __future__ import annotations

import json

from . import llm, schema
from .config import cfg, env, log

GENRE_DEFAULT = ("dark fantasy action in the spirit of modern shonen/seinen anime: a young hunter, a world of demons/spirits, "
                 "swordplay, sacrifice, found family, breathtaking set pieces")


def _vocab():
    return (f"shot types: {', '.join(schema.SHOT_TYPES)}\n"
            f"camera moves: {', '.join(schema.CAMERA_MOVES)}\n"
            f"fx (scene effects, max 4 per shot): {', '.join(schema.FX_BLENDER + schema.FX_POST)}\n"
            f"grades (colour moods): {', '.join(schema.GRADES)}\n"
            f"emotions (voice acting): {', '.join(schema.EMOTIONS)}\n"
            f"transitions (trans_in): {', '.join(schema.TRANSITIONS)}\n"
            f"music moods: {', '.join(schema.MUSIC)}\n"
            f"sfx: {', '.join(schema.SFX)}\n"
            f"voice styles: {', '.join(schema.VOICE_STYLES)}")


# ------------------------------------------------------------------ bible
def make_bible(idea="", genre=""):
    idea = idea or env("STORY_IDEA", "")
    genre = genre or env("STORY_GENRE", GENRE_DEFAULT)
    prompt = f"""You are the head writer and character designer of an ORIGINAL anime series (never use existing characters, names,
titles or world elements from any existing manga/anime/game). Genre: {genre}.
{('Seed idea from the creator: ' + idea) if idea else 'Invent a fresh, specific premise with a strong hook.'}

Return ONE JSON object:
{{"title_ja": "...", "title_ar": "...", "logline": "one English sentence", "genre": "...", "tone": "...",
 "world": "3-5 English sentences: setting, rules of the supernatural, stakes",
 "style": "English comma-separated visual style tags used for EVERY image (palette, mood, era, e.g. 'dark fantasy, feudal japan inspired, moody lighting')",
 "characters": [ 5 to 7 entries {{"id": "lowercase_ascii_id", "name_ja": "...", "name_ar": "Arabic transliteration of the name",
     "gender": "male|female", "voice": "hero|villain|elder|young|narrator|girl|boy", "role": "protagonist|rival|mentor|ally|antagonist|...",
     "personality": "one English sentence", "look": "ENGLISH danbooru-style tags that fix the appearance in every image: hair colour/length/style, outfit with colours, accessories, scars, age, build (e.g. '1boy, spiky black hair, scar on left cheek, black haori with white flame pattern, katana')",
     "eyes": "ENGLISH tags for the EYES only - the signature of each design: colour, shape, pupil, lashes, brows (e.g. 'large crimson eyes, sharp upturned lashes, slit pupils, thick straight brows')"}}],
 "locations": [ 4 to 6 {{"id": "...", "name": "...", "look": "English tags describing the place"}}],
 "arc": "5-8 English sentences: the overall story arc of the first season, with a clear antagonist and an emotional core"}}
Rules: designs must be distinctive and visually readable at a glance (a signature colour per character), no two characters
alike. Names must sound natural Japanese. `look` tags must be concrete and reusable. Respond with JSON only."""
    b = llm.ask_json(prompt, temperature=1.0, max_tokens=12000)
    if not isinstance(b, dict) or not b.get("characters"):
        raise llm.LLMError("bible has no characters")
    chars = schema.normalize_characters(b["characters"])
    if len(chars) < 3:
        raise llm.LLMError(f"bible has only {len(chars)} usable characters")
    by_id = {c["id"]: c for c in b["characters"] if isinstance(c, dict)}
    for c in chars:                       # keep the creative extras (role/personality) next to the normalised fields
        src = next((x for x in b["characters"] if isinstance(x, dict) and schema.sid(x.get("id") or x.get("name_ja")) == c["id"]), {})
        c["role"], c["personality"] = schema.text(src.get("role"), 40), schema.text(src.get("personality"), 200)
        c["eyes"] = schema.text(src.get("eyes"), 240)
    b["characters"] = chars
    b["style"] = schema.text(b.get("style"), 300)
    for k in ("title_ja", "title_ar", "logline", "genre", "tone", "world", "arc"):
        b[k] = schema.text(b.get(k), 900)
    b["locations"] = [{"id": schema.sid(x.get("id"), f"loc{i}"), "name": schema.text(x.get("name"), 60),
                       "look": schema.text(x.get("look"), 240)} for i, x in enumerate(b.get("locations") or []) if isinstance(x, dict)]
    return b


# ------------------------------------------------------------------ pilot / premiere / finale direction
PILOT_BRIEF = """THIS IS THE PILOT - the very first short of the anime. In two minutes a stranger must fall in love with the series.
Structure (targets in seconds):
  0-12   COLD OPEN, no explanation: the most striking image of the whole story (an eye widening in extreme close-up, a blade being
         drawn, a blood-red moon, a silhouette against fire) and ONE line that raises a question. Mark the eye shot `"eyes": "closeup"`.
         Use sfx "heartbeat"/"boom", an impact_frame/flash, and finish the cold open with trans_in "flash" or "whip".
  12-18  TITLE STING: one epic wide shot with `"title_card": "series"` (the series title appears in 3D), music "epic", slow push_in,
         light_rays + embers. No dialogue.
  18-55  NORMAL WORLD + PROTAGONIST: who he is, what he loves, warmth (family/friends), one vivid detail of this world; 3-4 charming,
         specific dialogue lines; foreshadow the danger. Calm grades, petals/dust/wind_sway, soft push_ins.
  55-95  INCITING INCIDENT: the world breaks (attack, loss, discovery). Fear turns into resolve. Handheld/dolly_zoom, rain/embers/mist.
         The EYES carry the emotion: closeups with trembling pupils, tears (`"eyes": "tears"`), a hand reaching for the blade.
  95-120 FIRST SPECTACLE + CLIFFHANGER: a fast, impressive action beat (short shots, impact shots, speed_lines, slash, shockwave,
         hit_stop) ending on the most shocking reveal; the very last shot holds on the protagonist's eyes igniting (`"eyes": "glow"`)."""

PART_BRIEF = {
    1: "This is the episode PREMIERE: open with a 1-line recap/hook, and put `\"title_card\": \"episode\"` on one early establishing shot.",
    7: "This is the episode FINALE: the climax pays off the whole week, then a quiet emotional beat and a one-shot teaser of the next episode.",
}

EYE_RULES = """EYES are the soul of this look (ufotable-style): every emotional beat gets an eye close-up. Use 3-6 shots per short with
`"eyes": "closeup"` (framing zoom 2.8-3.4, cy about 0.36-0.46 if the face is in the upper half of its image, cx on the character).
Use `"eyes": "glow"` when a character awakens a power or is a demon/spirit, `"eyes": "tears"` when someone cries. Close-ups of eyes
are usually 0.8-1.6 s, often with push_in or a slow rack focus, and always carry a hit of sound (heartbeat, shimmer)."""


def is_pilot(state, part):
    return bool(cfg.pilot and state.get("episode", 1) == 1 and part == 1)


def _source_block(source, what="part"):
    if not source:
        return ""
    m = source.get("manga") or {}
    who = ", ".join(m.get("authors") or []) or "the original authors"
    out = [f"SOURCE MATERIAL - this is a faithful ADAPTATION of the real manga \"{m.get('title', '')}\" by {who}. Follow its events, order, "
           "character behaviour and the spirit of its dialogue; compress and re-stage for animation, but never invent plot, never swap who does what."]
    if source.get("digest_text"):
        out.append("Page-by-page digest of the chapters of this episode:\n" + source["digest_text"])
    if source.get("pages"):
        rows = [f"p{p['page']} (ch {p['chapter']}): {p['summary']}" + (f" | \"{p['dialogue'][0]}\"" if p.get("dialogue") else "")
                for p in source["pages"]]
        out.append(f"Manga pages {source.get('range', ['?', '?'])[0]}-{source.get('range', ['?', '?'])[1]} that this {what} adapts:\n" + "\n".join(rows)[:7000])
    if source.get("images"):
        out.append("The attached images are key manga pages of this part: take their composition, angles and staging as direction for the "
                   "image prompts. Do NOT copy speech bubbles, text or screentone into prompts.")
    return "\n".join(out)


# ------------------------------------------------------------------ week outline
def write_week_outline(bible, state, source=None):
    ep = state.get("episode", 1)
    prev = [e.get("summary", "") for e in (state.get("episodes") or [])[-2:]]
    n = cfg.parts_per_episode
    pilot = bool(cfg.pilot and ep == 1)
    prompt = f"""Series bible (JSON): {json.dumps({k: bible.get(k, '') for k in ('title_ja', 'logline', 'world', 'arc', 'tone')}, ensure_ascii=False)}
Characters: {json.dumps([{k: c.get(k) for k in ('id', 'name_ja', 'role', 'personality')} for c in bible['characters']], ensure_ascii=False)}
Locations: {json.dumps([{k: l[k] for k in ('id', 'name')} for l in bible.get('locations', [])], ensure_ascii=False)}
Previous episode summaries: {json.dumps(prev, ensure_ascii=False)}
{_source_block(source)}

Plan EPISODE {ep} of the series as exactly {n} consecutive daily shorts (each ~2 minutes). Together the {n} parts form one
complete episode with a three-act shape: setup (parts 1-2), escalation (3-5), climax (6), resolution + hook for the next
episode (7). Every part must end on a hook/cliffhanger that makes the viewer want the next day's short, and every part must
contain at least one visually spectacular moment (action, reveal, emotional peak).
{'EPISODE 1 IS THE PREMIERE OF THE WHOLE ANIME: part 1 must be an irresistible opening (cold open with the most striking image, title sting, the protagonist introduced with heart, an inciting incident and a cliffhanger).' if pilot else ''}
{'Cover the source events IN ORDER; each part adapts the pages listed for it in the digest (about equal page ranges) - do not skip ahead or invent.' if source else ''}

Return JSON: {{"episode_title_ja": "...", "episode_title_ar": "...", "episode_summary": "2-3 English sentences",
 "parts": [ exactly {n} entries {{"part": 1, "title_ja": "...", "title_ar": "...", "beat": "3-4 English sentences of what happens",
   "locations": ["location ids"], "characters": ["character ids"], "set_piece": "the one spectacular visual moment",
   "mood": "one of {', '.join(schema.GRADES)}", "hook": "how the part ends"}} ]}}
Respond with JSON only."""
    o = llm.ask_json(prompt, temperature=0.8 if source else 0.95, max_tokens=9000)
    parts = o.get("parts") if isinstance(o, dict) else None
    if not parts or len(parts) < n:
        raise llm.LLMError(f"week outline has {len(parts or [])} parts, need {n}")
    o["parts"] = [dict(p, part=i + 1) for i, p in enumerate(parts[:n])]
    o["episode"] = ep
    return o


# ------------------------------------------------------------------ one short
TARGET_IMAGES = (14, 26)          # distinct key images per short (more reuse = cheaper, fewer = repetitive)
TARGET_SHOTS = (34, 56)


def _short_prompt(bible, state, part, beat, feedback="", source=None):
    chars = [{k: c.get(k) for k in ("id", "name_ja", "name_ar", "gender", "voice", "look", "eyes", "role", "personality")}
             for c in bible["characters"]]
    prev = [s.get("summary", "") for s in (state.get("shorts") or [])[-3:]]
    secs = cfg.short_seconds
    n0, n1 = TARGET_IMAGES
    s0, s1 = TARGET_SHOTS
    pilot = is_pilot(state, part)
    brief = PILOT_BRIEF if pilot else PART_BRIEF.get(part, "")
    return f"""You are directing PART {part} of {cfg.parts_per_episode} of EPISODE {state.get('episode', 1)} of an anime series, as a SHORT of {secs:.0f} seconds
(Demon-Slayer / Solo-Leveling-class cinematic direction: heavy camera work, dramatic framing, spectacle, emotion).

SERIES: {bible.get('title_ja', '')} - {bible.get('logline', '')}
WORLD: {bible.get('world', '')}
TONE: {bible.get('tone', '')}
GLOBAL STYLE TAGS (appended to every image prompt): {bible.get('style', '')}
CHARACTERS (use these ids; their `look` and `eyes` are injected into image prompts automatically, so do NOT repeat appearance tags in image prompts):
{json.dumps(chars, ensure_ascii=False)}
LOCATIONS: {json.dumps(bible.get('locations', []), ensure_ascii=False)}
EPISODE PLAN FOR THIS PART: {json.dumps(beat, ensure_ascii=False)}
WHAT HAPPENED IN PREVIOUS SHORTS: {json.dumps(prev, ensure_ascii=False)}
{_source_block(source)}
{brief}

OUTPUT: ONE JSON object:
{{"meta": {{"title_ja": "...", "title_ar": "catchy Arabic title", "description_ar": "2-3 Arabic sentences, spoiler-light, with a hook",
   "summary": "2-3 English sentences of what happens in THIS short (used as memory for later parts)",
   "tags": ["arabic and english search tags, 8-12"], "hashtags": ["#..."]}},
 "images": [ {n0}-{n1} key images {{"id": "i01", "scene": "s1", "prompt": "ENGLISH: setting, action, camera angle, lighting, atmosphere. No text. No character appearance tags.",
     "characters": ["ids of characters visible"], "framing": "wide|medium|closeup", "palette": "colour mood e.g. 'cold blue moonlight'", "no_humans": false,
     "hero": "character id for 2-4 portrait images of main characters (they are guided by that character's reference art), else omit"}} ],
 "shots": [ {s0}-{s1} shots {{"image": "i01", "type": "<shot type>", "dur": 3.0,
     "framing": {{"zoom": 1.0, "cx": 0.5, "cy": 0.5}}, "camera": {{"move": "<camera move>", "strength": 0.5}},
     "fx": ["max 4"], "grade": "<grade>", "line": {{"speaker": "character id or narrator", "ja": "Japanese line for voice acting",
     "ar": "Arabic subtitle translation", "emotion": "<emotion>", "intensity": 3}} or null,
     "sfx": ["max 3"], "music": "<mood>", "trans_in": "<transition>", "eyes": "closeup|glow|tears or omit", "title_card": "series|episode or omit",
     "notes": "director note"}} ]}}

CONTROLLED VOCABULARY (use ONLY these words):
{_vocab()}

DIRECTION RULES (this is what makes it look like real anime, not a slideshow):
1. Total of all shot `dur` must be {secs:.0f} seconds (+-6). Average shot 2.5-3.5 s; action cuts 0.8-1.8 s; emotional holds 3-5 s.
2. Open with a striking shot and an immediate hook in the first 10 seconds. End on the hook.
3. REUSE images: the same key image appears in several shots with different `framing` (zoom 1.0-3.2, cx/cy choose the crop centre,
   e.g. a face close-up cut-in from a medium image: zoom 2.4, cy 0.38). Images cost real money - never make one image per shot.
4. Vary camera moves; never the same move 3 times in a row. Use push_in for emotion, whip_l/whip_r + speed_lines for attacks,
   punch_in on impacts, orbit for reveals, crane_up/pull_out for scale, handheld for panic, dolly_zoom for dread.
5. Action sequences: short shots, `impact` shots with fx like ["shockwave","sparks","speed_lines","flash"], sfx like ["hit","clash","boom"],
   trans_in "flash" or "whip", grade "battle", music "battle". Use hit_stop/speed_ramp/impact_frame on the biggest hits.
6. Atmosphere on EVERY shot: pick fx that match the scene (petals, embers, snow, rain, dust, fireflies, mist, light_rays, wind_sway,
   hair_sway, grass, pebbles, leaves for outdoor ground scenes). Everything should feel alive: trees, hair, cloth, particles.
7. Dialogue: 14-26 spoken lines in total. Japanese lines are short (max ~28 chars), natural and emotional; Arabic is fluent Modern
   Standard Arabic, concise (max ~60 chars, may use one line break), NOT a word-for-word translation. Use the narrator sparingly (<=4 lines).
   Reaction shots (no line) between lines make dialogue feel alive. Pick `emotion`/`intensity` carefully (shout/angry/whisper/fear/sad...).
8. Match `grade` and `music` to the mood of each scene (calm, warm, tense, dark, battle, sad, epic, mystic, dawn, night).
9. Keep it family-safe: no gore, no nudity. Stylised violence only.
10. Character names in Arabic must be consistent with the `name_ar` above.
11. {EYE_RULES}
{('PROBLEMS WITH YOUR PREVIOUS ANSWER - FIX THEM: ' + feedback) if feedback else ''}
Respond with JSON only."""


def _check(script, pilot=False):
    probs = []
    st = schema.script_stats(script)
    lo_s, hi_s = TARGET_SHOTS
    if not (cfg.short_seconds * 0.82 <= st["planned_seconds"] <= cfg.short_seconds * 1.2):
        probs.append(f"total duration is {st['planned_seconds']}s, must be about {cfg.short_seconds:.0f}s")
    if st["shots"] < lo_s * 0.7:
        probs.append(f"only {st['shots']} shots, need at least {int(lo_s * 0.8)}")
    if st["lines"] < 8:
        probs.append(f"only {st['lines']} spoken lines, need 14-26")
    if st["images"] < TARGET_IMAGES[0] * 0.6:
        probs.append(f"only {st['images']} images, need at least {TARGET_IMAGES[0]}")
    if st["images"] > TARGET_IMAGES[1] * 1.6:
        probs.append(f"{st['images']} images is too many; reuse images with different framing")
    eyes = sum(1 for s in script["shots"] if s.get("eyes"))
    if eyes < 3:
        probs.append(f"only {eyes} shots carry \"eyes\" (closeup/glow/tears), need 3-6 eye close-ups")
    if pilot and not any(s.get("title_card") == "series" for s in script["shots"]):
        probs.append("pilot has no shot with \"title_card\": \"series\" (title sting after the cold open)")
    return probs


def _stretch(script, target):
    """Scale shot durations so the planned total hits `target` (director refines with real voice lengths)."""
    tot = sum(s["dur"] for s in script["shots"])
    if tot <= 0:
        return script
    k = target / tot
    if 0.9 < k < 1.1:
        return script
    for s in script["shots"]:
        s["dur"] = round(schema.clamp(s["dur"] * k, 0.8, 14.0), 2)
    return script


def write_short(bible, state, part, source=None):
    outline = state.get("week_outline") or {}
    beats = outline.get("parts") or []
    beat = beats[part - 1] if 0 < part <= len(beats) else {"part": part, "beat": "continue the story with escalating stakes"}
    pilot = is_pilot(state, part)
    imgs = list((source or {}).get("images") or [])
    feedback, script = "", None
    for attempt in range(3):
        raw = llm.ask_json(_short_prompt(bible, state, part, beat, feedback, source), images=imgs[:5] or None,
                           temperature=0.9 if attempt == 0 else 0.7, max_tokens=30000)
        try:
            script = schema.normalize_script(raw, characters=bible["characters"])
        except schema.ScriptError as e:
            feedback = f"structure invalid: {e}"
            log.warning("story attempt %d: %s", attempt + 1, feedback)
            continue
        probs = _check(script, pilot)
        if not probs:
            break
        feedback = "; ".join(probs)
        log.warning("story attempt %d problems: %s", attempt + 1, feedback)
    if script is None:
        raise llm.LLMError("the model never produced a usable script: " + feedback)
    script["style"] = bible.get("style", "")
    meta_in = raw.get("meta") if isinstance(raw, dict) and isinstance(raw.get("meta"), dict) else {}
    script["meta"].update({"part": part, "of": cfg.parts_per_episode, "episode": state.get("episode", 1), "pilot": pilot})
    script["meta"].setdefault("title_ar", beat.get("title_ar", ""))
    script["meta"]["episode_title_ar"] = (outline.get("episode_title_ar") or "")
    script["meta"]["series_title_ar"] = bible.get("title_ar", "")
    script["meta"]["series_title_ja"] = bible.get("title_ja", "")
    if source and source.get("manga"):
        script["meta"]["source"] = {"title": source["manga"].get("title"), "url": source["manga"].get("url")}
    if not script["meta"].get("summary"):
        script["meta"]["summary"] = schema.text(meta_in.get("summary") or beat.get("beat"), 600)
    probs = _check(script, pilot)
    if probs:
        log.warning("script accepted with remaining issues: %s", "; ".join(probs))
    return _stretch(script, cfg.short_seconds)


def summarize(script):
    m = script.get("meta", {})
    return {"part": m.get("part"), "title_ar": m.get("title_ar"), "summary": m.get("summary", "")}
