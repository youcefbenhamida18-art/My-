"""MangaDex integration: pick a Demon-Slayer-like manga, read its chapters, extract character references, adapt it to scripts.

    pick()                    -> {"id","title",...}   automatic: tags of the reference series (MANGA_LIKE) rank the candidates;
                                                       MANGA_ID / MANGA_TITLE override
    readable_chapters(id)     -> chapters hosted on MangaDex itself (external/licensed-only chapters are skipped)
    download_chapter / covers -> local files (data-saver quality for the vision model, original quality for crops)
    ingest(state, part, W)    -> normalised script.json dict for one ~2-minute short (called by pipeline.make_script)

Public MangaDex API (https://api.mangadex.org, no key needed for reading). Rules followed: descriptive User-Agent, <= 4 requests/s,
at-home base URLs re-requested when they expire, image results reported to api.mangadex.network when served by the MD@Home network.
The pages are only read by the vision model and cropped for character references; the video contains generated art, not the pages.
"""
from __future__ import annotations

import io
import json
import math
import pathlib
import random
import re
import time

import requests
from PIL import Image

from . import llm, schema, story
from .config import STATE_DIR, WORK_ROOT, cfg, log, save_state

API = "https://api.mangadex.org"
UPLOADS = "https://uploads.mangadex.org"
REPORT = "https://api.mangadex.network/report"
UA = {"User-Agent": "MangaForgeStudio/2.0 (anime adaptation pipeline; contact via GitHub repository)"}
RATINGS = ["safe", "suggestive"]
DEFAULT_PROFILE = ["Action", "Supernatural", "Demons", "Historical", "Samurai", "Martial Arts", "Tragedy", "Drama"]
SIGNATURE = [["Demons", "Action"], ["Supernatural", "Action", "Historical"], ["Samurai", "Action", "Supernatural"],
             ["Monsters", "Action", "Martial Arts"], ["Demons", "Historical"], ["Supernatural", "Action", "Tragedy"]]


class MangaDexError(RuntimeError):
    def __init__(self, msg, status=0):
        super().__init__(msg)
        self.status = status


# ------------------------------------------------------------------ transport
_last = [0.0]


def _throttle(gap):
    d = time.time() - _last[0]
    if d < gap:
        time.sleep(gap - d)
    _last[0] = time.time()


def _detail(r):
    try:
        e = (r.json().get("errors") or [{}])[0]
        return f"{e.get('title', '')}: {e.get('detail', '')}".strip(": ") or r.text[:200]
    except Exception:  # noqa: BLE001
        return r.text[:200]


def _get(path, params=None, *, tries=5, timeout=45, raw=False, gap=0.26):
    url = path if path.startswith("http") else API + path
    last = None
    for i in range(tries):
        _throttle(gap)
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
        except requests.RequestException as e:
            last = MangaDexError(f"MangaDex unreachable: {e}")
            time.sleep(min(30, 2 ** i) + random.random())
            continue
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After") or 0)
            except ValueError:
                wait = 0
            last = MangaDexError("MangaDex rate limit (429)", 429)
            time.sleep(min(90, wait or 6 * (i + 1)))
            continue
        if r.status_code >= 500:
            last = MangaDexError(f"MangaDex {r.status_code}", r.status_code)
            time.sleep(min(40, 3 * 2 ** i) + random.random())
            continue
        if r.status_code >= 400:
            raise MangaDexError(f"MangaDex {r.status_code} on {path}: {_detail(r)}", r.status_code)
        return r if raw else r.json()
    raise last or MangaDexError("MangaDex request failed")


def ping():
    try:
        r = _get("/ping", raw=True, tries=2, timeout=15)
        return "pong" in r.text.lower()
    except MangaDexError:
        return False


# ------------------------------------------------------------------ entities
_TAGS = {}


def tag_map():
    if not _TAGS:
        for t in _get("/manga/tag").get("data", []):
            n = ((t.get("attributes") or {}).get("name") or {}).get("en")
            if n:
                _TAGS[n.lower()] = t["id"]
    return _TAGS


def _name(d):
    if isinstance(d, dict):
        for k in ("en", "ja-ro", "ja"):
            if d.get(k):
                return d[k]
        for v in d.values():
            if v:
                return v
    return ""


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def info_of(m):
    a = m.get("attributes") or {}
    rel = m.get("relationships") or []
    cover = next((r["attributes"]["fileName"] for r in rel if r.get("type") == "cover_art" and (r.get("attributes") or {}).get("fileName")), "")
    people = lambda kind: [r["attributes"]["name"] for r in rel if r.get("type") == kind and (r.get("attributes") or {}).get("name")]  # noqa: E731
    tags = [(_name((t.get("attributes") or {}).get("name")), t["id"]) for t in a.get("tags") or []]
    alts = [v for t in a.get("altTitles") or [] for v in t.values()]
    return {"id": m["id"], "title": _name(a.get("title")), "alt_titles": alts[:12], "description": _name(a.get("description"))[:1200],
            "tags": [n for n, _ in tags], "tag_ids": [i for _, i in tags], "demographic": a.get("publicationDemographic") or "",
            "lang": a.get("originalLanguage") or "", "status": a.get("status") or "", "year": a.get("year"),
            "rating": a.get("contentRating") or "", "cover": cover, "authors": people("author"), "artists": people("artist"),
            "url": f"https://mangadex.org/title/{m['id']}"}


def _manga(params):
    p = {"limit": 20, "includes[]": ["cover_art", "author", "artist"], "contentRating[]": RATINGS}
    p.update(params)
    try:
        return _get("/manga", p).get("data", [])
    except MangaDexError as e:
        if e.status == 400:                                  # an optional filter was refused: retry with the bare minimum
            for k in ("hasAvailableChapters", "availableTranslatedLanguage[]", "originalLanguage[]", "includedTagsMode", "excludedTags[]"):
                p.pop(k, None)
            return _get("/manga", p).get("data", [])
        raise


def find_reference(title):
    rows = _manga({"title": title, "limit": 10})
    want = _norm(title)
    for m in rows:
        if want in {_norm(t) for t in [_name((m.get("attributes") or {}).get("title"))] +
                    [v for t in (m.get("attributes") or {}).get("altTitles") or [] for v in t.values()]}:
            return m
    return rows[0] if rows else None


def profile():
    ref, tm = None, tag_map()
    try:
        ref = find_reference(cfg.manga_like)
    except MangaDexError as e:
        log.warning("reference series %r not found (%s) - using the default tag profile", cfg.manga_like, e)
    if ref:
        i = info_of(ref)
        log.info("reference %r -> %s tags=%s", cfg.manga_like, i["title"], i["tags"])
        return {"tags": set(i["tag_ids"]), "demo": i["demographic"], "lang": i["lang"] or "ja", "ref_id": i["id"], "ref_title": i["title"]}
    return {"tags": {tm[n.lower()] for n in DEFAULT_PROFILE if n.lower() in tm}, "demo": "shounen", "lang": "ja", "ref_id": "", "ref_title": cfg.manga_like}


# ------------------------------------------------------------------ chapters
def _num(c):
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def readable_chapters(manga_id, lang=None):
    """Chapters whose pages MangaDex itself hosts (not external links), one per chapter number, in reading order."""
    lang = lang or cfg.manga_lang
    out, offset = [], 0
    while offset < 1500:
        p = {"limit": 500, "offset": offset, "translatedLanguage[]": [lang], "order[chapter]": "asc", "includeExternalUrl": 0,
             "includeEmptyPages": 0, "contentRating[]": RATINGS + ["erotica"], "includes[]": ["scanlation_group"]}
        try:
            j = _get(f"/manga/{manga_id}/feed", p)
        except MangaDexError as e:
            if e.status != 400:
                raise
            for k in ("includeExternalUrl", "includeEmptyPages", "includes[]", "contentRating[]"):
                p.pop(k, None)
            j = _get(f"/manga/{manga_id}/feed", p)
        rows = j.get("data", [])
        for c in rows:
            a = c.get("attributes") or {}
            if a.get("externalUrl") or int(a.get("pages") or 0) <= 0:
                continue
            n = _num(a.get("chapter"))
            if n is None:
                continue
            grp = next((r["attributes"].get("name") for r in c.get("relationships", []) if r.get("type") == "scanlation_group"
                        and r.get("attributes")), "")
            out.append({"id": c["id"], "num": n, "chapter": a.get("chapter"), "volume": a.get("volume"), "title": a.get("title") or "",
                        "pages": int(a.get("pages") or 0), "group": grp or ""})
        if len(rows) < 500:
            break
        offset += 500
    best = {}
    for c in out:                                           # several scanlation groups: keep the longest version of each chapter
        if c["num"] not in best or c["pages"] > best[c["num"]]["pages"]:
            best[c["num"]] = c
    return [best[k] for k in sorted(best)]


# ------------------------------------------------------------------ selection
def score(info, prof, rank):
    inter = len(set(info["tag_ids"]) & prof["tags"])
    union = len(set(info["tag_ids"]) | prof["tags"]) or 1
    s = 4.0 * inter / union
    s += 0.7 if info["lang"] == prof["lang"] else 0.0
    s += 0.5 if info["demographic"] and info["demographic"] == prof["demo"] else (0.25 if info["demographic"] in ("shounen", "seinen") else 0.0)
    s += 1.2 / (1 + rank / 8.0)
    s += 0.25 if info["cover"] else 0.0
    return round(s, 3)


def candidates(prof):
    tm = tag_map()
    excl = [tm[n.lower()] for n in cfg.manga_exclude_tags if n.lower() in tm]
    found = {}
    for combo in SIGNATURE:
        ids = [tm[n.lower()] for n in combo if n.lower() in tm]
        if len(ids) < 2:
            continue
        p = {"limit": 50, "includedTags[]": ids, "includedTagsMode": "AND", "originalLanguage[]": ["ja"],
             "availableTranslatedLanguage[]": [cfg.manga_lang], "hasAvailableChapters": "true", "order[followedCount]": "desc"}
        if excl:
            p["excludedTags[]"] = excl
        for rank, m in enumerate(_manga(p)):
            if m["id"] != prof.get("ref_id") or _flag("MANGA_ALLOW_REFERENCE"):
                found.setdefault(m["id"], (rank, m))
    return found


def _flag(name):
    import os
    return os.getenv(name, "0").lower() in ("1", "true", "yes")


def _entry(info, chapters, **extra):
    return {**{k: info[k] for k in ("id", "title", "alt_titles", "description", "tags", "demographic", "lang", "status", "year", "cover",
                                    "authors", "artists", "url")},
            "readable_chapters": len(chapters), "first_chapter": chapters[0]["num"] if chapters else None,
            "picked_at": time.strftime("%Y-%m-%d"), **extra}


def pick():
    """Choose (or resolve) the manga to adapt. Needs the network."""
    if cfg.manga_id:
        m = _get(f"/manga/{cfg.manga_id}", {"includes[]": ["cover_art", "author", "artist"]})["data"]
        info = info_of(m)
        chs = readable_chapters(info["id"])
        if not chs:
            raise MangaDexError(f"{info['title']} has no chapters hosted on MangaDex in '{cfg.manga_lang}' (only external links)")
        return _entry(info, chs, reason="MANGA_ID")
    if cfg.manga_title:
        for m in _manga({"title": cfg.manga_title, "limit": 10}):
            info = info_of(m)
            chs = readable_chapters(info["id"])
            if chs:
                return _entry(info, chs, reason=f"MANGA_TITLE={cfg.manga_title}")
        raise MangaDexError(f"no readable manga found for MANGA_TITLE={cfg.manga_title!r}")
    prof = profile()
    cands = candidates(prof)
    if not cands:
        raise MangaDexError("MangaDex returned no candidates (check the network or set MANGA_ID)")
    ranked = sorted(((score(info_of(m), prof, rk), rk, m) for rk, m in cands.values()), key=lambda x: -x[0])
    log.info("manga candidates: %s", [(s, info_of(m)["title"]) for s, _, m in ranked[:8]])
    fallback = None
    for s, rk, m in ranked[:14]:
        info = info_of(m)
        try:
            chs = readable_chapters(info["id"])
        except MangaDexError as e:
            log.warning("skip %s: %s", info["title"], e)
            continue
        if len(chs) < cfg.manga_min_chapters:
            continue
        begins = chs[0]["num"] <= 1.0
        e = _entry(info, chs, score=s, like=prof["ref_title"], reason="automatic (tags similar to the reference series)", begins_at_1=begins)
        if begins:
            return e
        fallback = fallback or e                           # readable but starts mid-story: only if nothing begins at chapter 1
    if fallback:
        return fallback
    raise MangaDexError(f"none of the {len(cands)} similar titles has >= {cfg.manga_min_chapters} chapters readable on MangaDex "
                        "(many are licensed / external-only). Set MANGA_ID to a title you can read, or lower MANGA_MIN_CHAPTERS.")


def mdir(manga_id):
    d = STATE_DIR / "manga" / manga_id                         # small, committed: info, refs, digests
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache(manga_id):
    d = WORK_ROOT / "manga" / manga_id                          # big, ephemeral: pages and covers
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_manga(state):
    if state.get("manga") and not (cfg.manga_id and cfg.manga_id != state["manga"].get("id")):
        return state["manga"]
    state["manga"] = pick()
    (mdir(state["manga"]["id"]) / "info.json").write_text(json.dumps(state["manga"], ensure_ascii=False, indent=1), "utf-8")
    state["bible"] = None                                        # a new source means a new bible
    state["week_outline"] = None
    save_state(state)
    log.info("manga chosen: %s (%s) - %d readable chapters", state["manga"]["title"], state["manga"]["url"], state["manga"]["readable_chapters"])
    return state["manga"]


# ------------------------------------------------------------------ files
def _report(url, ok, nbytes, ms, cached):
    if ".mangadex.org" in url or "uploads.mangadex" in url:
        return
    try:
        requests.post(REPORT, json={"url": url, "success": ok, "cached": cached, "bytes": nbytes, "duration": ms}, headers=UA, timeout=5)
    except requests.RequestException:
        pass


def _download(url, dest, tries=3):
    for i in range(tries):
        t0 = time.time()
        try:
            r = _get(url, raw=True, tries=1, timeout=60, gap=0.12)
            _report(url, True, len(r.content), int((time.time() - t0) * 1000), r.headers.get("X-Cache", "").startswith("HIT"))
            Image.open(io.BytesIO(r.content)).verify()
            dest.write_bytes(r.content)
            return True
        except Exception as e:  # noqa: BLE001
            _report(url, False, 0, int((time.time() - t0) * 1000), False)
            if i == tries - 1:
                log.warning("download failed %s: %s", url, e)
            time.sleep(1.5 * (i + 1))
    return False


def download_chapter(chapter, dest, quality="data-saver", max_pages=None):
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if quality == "data-saver" else None
    got = sorted(dest.glob("p*.*"))
    if got and len(got) >= min(chapter["pages"], max_pages or 10 ** 6):
        return got
    j = _get(f"/at-home/server/{chapter['id']}")
    base, h = j["baseUrl"], j["chapter"]["hash"]
    names = j["chapter"]["dataSaver" if quality == "data-saver" else "data"]
    folder = quality
    out, fails = [], 0
    names = names[:max_pages] if max_pages else names
    for i, name in enumerate(names, 1):
        p = dest / f"p{i:03d}{pathlib.Path(name).suffix or '.' + (ext or 'png')}"
        if p.exists() and p.stat().st_size > 2000:
            out.append(p)
            continue
        if not _download(f"{base}/{folder}/{h}/{name}", p):
            fails += 1
            if fails == 2:                                         # base URL probably expired: ask for a fresh one
                j = _get(f"/at-home/server/{chapter['id']}")
                base = j["baseUrl"]
            if not _download(f"{base}/{folder}/{h}/{name}", p):
                continue
        out.append(p)
    if len(out) < max(3, len(names) // 2):
        raise MangaDexError(f"chapter {chapter['chapter']}: only {len(out)}/{len(names)} pages could be downloaded")
    return out


def covers(manga_id, limit=4):
    dest = cache(manga_id) / "covers"
    dest.mkdir(exist_ok=True)
    rows = _get("/cover", {"manga[]": [manga_id], "limit": 100, "order[volume]": "asc"}).get("data", [])
    rows.sort(key=lambda c: (_num((c.get("attributes") or {}).get("volume")) or 999))
    out = []
    for c in rows[:limit]:
        a = c.get("attributes") or {}
        p = dest / f"v{a.get('volume') or 'x'}_{pathlib.Path(a['fileName']).suffix.lstrip('.') or 'jpg'}.jpg"
        if p.exists() or _download(f"{UPLOADS}/covers/{manga_id}/{a['fileName']}", p):
            out.append(p)
    return out


def local_pages():
    d = pathlib.Path(cfg.manga_dir)
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]) if d.exists() else []


# ------------------------------------------------------------------ vision helpers
def _small(path, side=1280, q=85):
    im = Image.open(path).convert("RGB")
    k = side / max(im.size)
    if k < 1:
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return b.getvalue()


def crop_box(path, box, out, *, pad=0.12, max_side=768, min_px=90):
    """box = [ymin, xmin, ymax, xmax] in 0..1000 (Gemini's convention) on the image `path`."""
    try:
        y0, x0, y1, x1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    im = Image.open(path).convert("RGB")
    W, H = im.size
    bw, bh = (x1 - x0) / 1000 * W, (y1 - y0) / 1000 * H
    if bw < min_px * 0.6 or bh < min_px * 0.6 or bw <= 0 or bh <= 0:
        return None
    cx, cy = (x0 + x1) / 2000 * W, (y0 + y1) / 2000 * H
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    L, T, R, B = max(0, cx - bw / 2), max(0, cy - bh / 2), min(W, cx + bw / 2), min(H, cy + bh / 2)
    if R - L < min_px or B - T < min_px:
        return None
    c = im.crop((int(L), int(T), int(R), int(B)))
    k = max_side / max(c.size)
    if k < 1:
        c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
    c.save(out)
    return pathlib.Path(out)


def _ask(prompt, images, **kw):
    try:
        return llm.ask_json(prompt, images=[_small(p) for p in images], **kw)
    except llm.LLMError as e:
        if len(images) > 4:                                       # vision refused / too large: retry with half the pages
            log.warning("vision call failed (%s) - retrying with %d images", e, len(images) // 2)
            return llm.ask_json(prompt, images=[_small(p, 1024, 80) for p in images[: len(images) // 2]], **kw)
        raise


# ------------------------------------------------------------------ the series bible from the real manga
def build_bible(state, info, cover_paths, page_paths):
    imgs = list(cover_paths[:3]) + list(page_paths[:9])
    labels = [f"image {i}: " + ("COLOUR COVER" if i < len(cover_paths[:3]) else "manga page (read right-to-left, black & white)")
              for i in range(len(imgs))]
    prompt = f"""You are the head character designer of the anime adaptation of the manga "{info['title']}" ({', '.join(info['tags'][:8])}).
Synopsis (MangaDex): {info['description']}
The attached images, in order: {'; '.join(labels)}.
Study them closely and return ONE JSON object describing the series AS IT REALLY IS in the manga (real names as written in the pages,
real designs). Where the pages are black & white, take colours from the colour covers; when a colour is not shown anywhere choose the
most plausible one for the tone and keep it consistent.

{{"title_ja": "Japanese title", "title_ar": "Arabic title (natural, short)", "logline": "one English sentence", "genre": "...", "tone": "...",
 "world": "3-5 English sentences: setting, supernatural rules, stakes",
 "style": "ENGLISH comma-separated visual style tags for EVERY image: palette, mood, era, lighting (match the manga's world; anime key-art look)",
 "characters": [ 4 to 7 main characters visible or named in the images, most important first {{
     "id": "lowercase_ascii_id", "name_ja": "...", "name_ar": "Arabic transliteration", "gender": "male|female",
     "voice": "hero|villain|elder|young|narrator|girl|boy", "role": "protagonist|rival|mentor|ally|antagonist|...",
     "personality": "one English sentence",
     "look": "ENGLISH danbooru-style tags that fix the appearance in every image: hair colour/length/style, outfit with colours, accessories, scars, age, build, weapon",
     "eyes": "ENGLISH tags for the EYES only, as exact as possible: colour, shape (sharp/round/narrow/droopy), pupil type, lashes, brows, signature detail e.g. 'large amber eyes, sharp upturned lashes, slit pupils, thick straight brows'",
     "ref": {{"image": <index of the image where this character is drawn clearest>, "box_2d": [ymin, xmin, ymax, xmax] of the whole upper body or figure, 0-1000,
              "face_box_2d": [ymin, xmin, ymax, xmax] of the face only, 0-1000}} }} ],
 "locations": [ 4 to 6 {{"id": "...", "name": "...", "look": "English tags describing the place"}} ],
 "arc": "5-8 English sentences: the story arc as in the manga"}}
Rules: no invented characters; if a character appears only as a silhouette give `ref` null. Respond with JSON only."""
    b = _ask(prompt, imgs, temperature=0.5, max_tokens=14000)
    if not isinstance(b, dict) or not b.get("characters"):
        raise llm.LLMError("manga bible has no characters")
    raw = [c for c in b["characters"] if isinstance(c, dict)]
    chars = schema.normalize_characters(raw)
    refdir = mdir(info["id"]) / "refs"
    refdir.mkdir(exist_ok=True)
    for c in chars:
        src = next((x for x in raw if schema.sid(x.get("id") or x.get("name_ja")) == c["id"]), {})
        c["role"], c["personality"] = schema.text(src.get("role"), 40), schema.text(src.get("personality"), 200)
        c["eyes"] = schema.text(src.get("eyes"), 220)
        ref = src.get("ref") if isinstance(src.get("ref"), dict) else None
        if ref and isinstance(ref.get("image"), int) and 0 <= ref["image"] < len(imgs):
            body = crop_box(imgs[ref["image"]], ref.get("box_2d"), refdir / f"{c['id']}.png", pad=0.08)
            face = crop_box(imgs[ref["image"]], ref.get("face_box_2d"), refdir / f"{c['id']}_face.png", pad=0.35, min_px=70)
            c["ref"] = {"body": str(body.relative_to(STATE_DIR.parent)) if body else None,
                        "face": str(face.relative_to(STATE_DIR.parent)) if face else None,
                        "colour": ref["image"] < len(cover_paths[:3])}
    b["characters"] = chars
    b["style"] = schema.text(b.get("style"), 300)
    for k in ("title_ja", "title_ar", "logline", "genre", "tone", "world", "arc"):
        b[k] = schema.text(b.get(k), 900)
    b["locations"] = [{"id": schema.sid(x.get("id"), f"loc{i}"), "name": schema.text(x.get("name"), 60), "look": schema.text(x.get("look"), 240)}
                      for i, x in enumerate(b.get("locations") or []) if isinstance(x, dict)]
    b["source"] = {"manga_id": info["id"], "title": info["title"], "url": info["url"], "authors": info["authors"]}
    return b


# ------------------------------------------------------------------ episode digest (what happens, page by page)
def digest_episode(state, info, episode, bible):
    f = mdir(info["id"]) / f"digest_e{episode:02d}.json"
    if f.exists():
        return json.loads(f.read_text("utf-8"))
    chs = readable_chapters(info["id"])
    k = cfg.manga_chapters_per_episode
    sel = chs[(episode - 1) * k: episode * k]
    if not sel:
        raise MangaDexError(f"episode {episode}: the manga has no more readable chapters after #{chs[-1]['chapter'] if chs else '?'} - "
                            "set MANGA_ID to another title or STORY_SOURCE=original to continue with an invented story")
    names = [{"id": c["id"], "name_ja": c.get("name_ja"), "name_ar": c.get("name_ar"), "look": c.get("look")} for c in bible.get("characters", [])]
    pages, rec = [], []
    for ch in sel:
        files = download_chapter(ch, cache(info["id"]) / f"ch{ch['num']:g}")
        for i, p in enumerate(files, 1):
            pages.append((ch, i, p))
    prev_note, digest_pages = "", []
    B = cfg.manga_vision_pages
    for s in range(0, len(pages), B):
        batch = pages[s: s + B]
        prompt = f"""Manga "{info['title']}". These {len(batch)} images are consecutive pages (read RIGHT-TO-LEFT, top to bottom) of chapter(s)
{sorted({c['chapter'] for c, _, _ in batch})}. Known characters: {json.dumps(names, ensure_ascii=False)}.
Story so far (previous batch): {prev_note or '(this is the beginning)'}
For EACH image in order return one entry. JSON: {{"pages": [{{"n": 1, "summary": "English, 1-2 sentences: what happens, who, where, the visual composition",
 "dialogue": ["the important spoken lines on the page (English as printed), max 4"], "characters": ["ids from the list or new lowercase ids"],
 "intensity": 1-5, "key": true if this page is a spectacular/emotional full-page or splash moment}}], "recap": "2 sentences: where the story stands after this batch"}}
Respond with JSON only."""
        o = _ask(prompt, [p for _, _, p in batch], temperature=0.3, max_tokens=12000)
        rows = (o.get("pages") if isinstance(o, dict) else None) or []
        for j, (ch, i, p) in enumerate(batch):
            r = rows[j] if j < len(rows) and isinstance(rows[j], dict) else {}
            digest_pages.append({"chapter": ch["chapter"], "page": i, "file": str(p.relative_to(WORK_ROOT.parent)),
                                 "summary": schema.text(r.get("summary"), 300), "dialogue": [schema.text(x, 140) for x in (r.get("dialogue") or [])[:4]],
                                 "characters": [schema.sid(x) for x in (r.get("characters") or []) if x][:6],
                                 "intensity": schema.as_int(r.get("intensity"), 2), "key": bool(r.get("key"))})
        prev_note = schema.text((o or {}).get("recap") if isinstance(o, dict) else "", 400)
    d = {"episode": episode, "manga": info["title"], "chapters": [{"chapter": c["chapter"], "title": c["title"], "pages": c["pages"]} for c in sel],
         "pages": digest_pages}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    return d


def part_source(digest, part, n_parts):
    pages = digest["pages"]
    a, b = int(len(pages) * (part - 1) / n_parts), int(math.ceil(len(pages) * part / n_parts))
    seg = pages[a:max(b, a + 1)]
    keys = [p for p in seg if p.get("key")] or sorted(seg, key=lambda p: -p["intensity"])[:3]
    files = [pathlib.Path(WORK_ROOT.parent) / p["file"] for p in keys[:5]]
    return {"pages": seg, "images": [f for f in files if f.exists()], "range": [a + 1, b]}


def digest_text(digest, limit=9000):
    lines = [f"p{i + 1} (ch {p['chapter']}.{p['page']}): {p['summary']}" + (f" | \"{p['dialogue'][0]}\"" if p["dialogue"] else "")
             for i, p in enumerate(digest["pages"])]
    t = "\n".join(lines)
    return t if len(t) <= limit else t[:limit] + " ..."


# ------------------------------------------------------------------ entry point used by the pipeline
def _bible(state, info):
    if state.get("bible") and (state["bible"].get("source") or {}).get("manga_id") == info["id"]:
        return state["bible"]
    cv = covers(info["id"])
    chs = readable_chapters(info["id"])
    pages = download_chapter(chs[0], cache(info["id"]) / f"ch{chs[0]['num']:g}", max_pages=12)
    log.info("building the series bible from %d covers and %d pages of %s ...", len(cv), len(pages), info["title"])
    state["bible"] = build_bible(state, info, cv, pages)
    save_state(state)
    return state["bible"]


def ingest(state, part, W):
    """-> normalised script for this part of the current episode (adapting the manga)."""
    info = ensure_manga(state)
    bible = _bible(state, info)
    ep = state["episode"]
    digest = digest_episode(state, info, ep, bible)
    o = state.get("week_outline")
    if not (o and o.get("episode") == ep and len(o.get("parts", [])) >= cfg.parts_per_episode):
        log.info("planning episode %s from the manga chapters %s ...", ep, [c["chapter"] for c in digest["chapters"]])
        state["week_outline"] = story.write_week_outline(bible, state, source={"manga": info, "digest_text": digest_text(digest)})
        save_state(state)
    ps = part_source(digest, part, cfg.parts_per_episode)
    src = {"manga": info, "pages": ps["pages"], "images": ps["images"], "range": ps["range"]}
    return story.write_short(bible, state, part, source=src)


def credits(state):
    m = state.get("manga")
    if not m:
        return []
    who = ", ".join(dict.fromkeys(m.get("authors", []) + m.get("artists", [])))
    return [f"مقتبس من المانغا: {m['title']}" + (f" - {who}" if who else ""), f"المصدر: {m['url']} (MangaDex)",
            "عمل معجبين غير رسمي، الرسومات مولَّدة بالذكاء الاصطناعي."]
