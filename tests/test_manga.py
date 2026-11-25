"""MangaDex client + adaptation flow against a fake MangaDex (no network) and a fake LLM."""
import io
import json
import pathlib

import pytest
from PIL import Image

from mangaforge import manga, schema, story
from mangaforge.config import cfg


def png_bytes(w=200, h=300, color=(200, 200, 200)):
    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, "PNG")
    return b.getvalue()


TAGS = {"Action": "t-action", "Supernatural": "t-super", "Demons": "t-demons", "Historical": "t-hist", "Samurai": "t-sam",
        "Martial Arts": "t-ma", "Tragedy": "t-trag", "Monsters": "t-mon", "Drama": "t-drama", "Gore": "t-gore", "Harem": "t-harem"}


def entity(mid, title, tags, demo="shounen", lang="ja", cover="c.jpg", alt=None):
    return {"id": mid, "type": "manga", "attributes": {
        "title": {"en": title}, "altTitles": [{"en": a} for a in (alt or [])], "description": {"en": f"{title} synopsis"},
        "tags": [{"id": TAGS[t], "attributes": {"name": {"en": t}}} for t in tags], "publicationDemographic": demo,
        "originalLanguage": lang, "status": "completed", "year": 2016, "contentRating": "safe"},
        "relationships": [{"type": "cover_art", "attributes": {"fileName": cover}}, {"type": "author", "attributes": {"name": "Author A"}}]}


def chapter(cid, num, pages=20, ext=None, group="G1"):
    return {"id": cid, "type": "chapter", "attributes": {"chapter": str(num), "volume": "1", "title": f"Ch {num}", "pages": pages,
                                                           "externalUrl": ext, "translatedLanguage": "en"},
            "relationships": [{"type": "scanlation_group", "attributes": {"name": group}}]}


class Resp:
    def __init__(self, data=None, status=200, content=b"", headers=None):
        self._d, self.status_code, self.content, self.headers = data, status, content, headers or {}
        self.text = content.decode("latin1") if content else json.dumps(data or {})

    def json(self):
        return self._d


class FakeMD:
    REF = entity("ref-1", "Kimetsu no Yaiba", ["Action", "Supernatural", "Demons", "Historical"], alt=["Demon Slayer"])
    A = entity("m-a", "Blade of Ash", ["Action", "Demons", "Historical", "Supernatural"])          # perfect match, begins at 1
    B = entity("m-b", "Moon Hunters", ["Action", "Demons"])                                          # fewer chapters (rejected)
    C = entity("m-c", "Licensed Only", ["Action", "Demons", "Historical", "Supernatural", "Samurai"])  # all chapters external

    def __init__(self):
        self.calls = []
        self.reports = []

    def feed(self, mid):
        if mid == "m-a":
            rows = [chapter(f"a{i}", i, pages=18 + i) for i in range(1, 13)] + [chapter("a3b", 3, pages=30, group="G2"),
                                                                                  chapter("aext", 13, pages=0, ext="https://x"),
                                                                                  chapter("a-oneshot", "x", pages=5)]
        elif mid == "m-b":
            rows = [chapter(f"b{i}", i) for i in range(1, 4)]
        else:
            rows = [chapter(f"c{i}", i, pages=0, ext="https://official") for i in range(1, 30)]
        return rows

    def __call__(self, url, params=None, headers=None, timeout=0):
        params = params or {}
        path = url.replace(manga.API, "")
        self.calls.append((path, dict(params)))
        if path == "/ping":
            return Resp(content=b"pong")
        if path == "/manga/tag":
            return Resp({"data": [{"id": i, "attributes": {"name": {"en": n}}} for n, i in TAGS.items()]})
        if path == "/manga" and "title" in params:
            return Resp({"data": [self.REF]})
        if path == "/manga":
            ids = params.get("includedTags[]", [])
            rows = [e for e in (self.REF, self.A, self.B, self.C)
                    if all(i in {t["id"] for t in e["attributes"]["tags"]} for i in ids)]
            return Resp({"data": rows})
        if path.startswith("/manga/") and path.endswith("/feed"):
            return Resp({"data": self.feed(path.split("/")[2])})
        if path.startswith("/manga/") and path.count("/") == 2:
            return Resp({"data": {"m-a": self.A, "m-b": self.B, "m-c": self.C}[path.split("/")[2]]})
        if path.startswith("/at-home/server/"):
            return Resp({"baseUrl": "https://node1.mangadex.network", "chapter": {"hash": "h1", "data": [f"{i}.png" for i in range(1, 21)],
                                                                             "dataSaver": [f"{i}.jpg" for i in range(1, 21)]}})
        if path == "/cover":
            return Resp({"data": [{"attributes": {"volume": "2", "fileName": "v2.jpg"}}, {"attributes": {"volume": "1", "fileName": "v1.jpg"}}]})
        if url.startswith("https://node1.mangadex.network/") or url.startswith(manga.UPLOADS):
            return Resp(content=png_bytes(), headers={"X-Cache": "HIT"})
        return Resp({"errors": [{"title": "nf", "detail": url}]}, status=404)

    def post(self, url, json=None, headers=None, timeout=0):
        self.reports.append((url, json))
        return Resp({})


@pytest.fixture()
def md(monkeypatch, tmp_path):
    fake = FakeMD()
    monkeypatch.setattr(manga.requests, "get", fake)
    monkeypatch.setattr(manga.requests, "post", fake.post)
    monkeypatch.setattr(manga, "_throttle", lambda gap: None)
    monkeypatch.setattr(manga, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(manga, "WORK_ROOT", tmp_path / "work")
    manga._TAGS.clear()
    cfg.manga_id = cfg.manga_title = ""
    cfg.manga_min_chapters = 8
    return fake


def test_readable_chapters_filters_external_and_dedupes(md):
    chs = manga.readable_chapters("m-a")
    assert [c["num"] for c in chs] == [float(i) for i in range(1, 13)]           # external, one-shot "x", and duplicate removed
    assert next(c for c in chs if c["num"] == 3)["pages"] == 30                    # the longer scanlation of chapter 3 wins
    assert manga.readable_chapters("m-c") == []


def test_pick_prefers_similar_readable_title_starting_at_chapter_1(md):
    e = manga.pick()
    assert e["id"] == "m-a" and e["begins_at_1"] and e["readable_chapters"] == 12
    assert e["like"] == "Kimetsu no Yaiba"
    assert e["url"].endswith("/title/m-a")
    # the reference itself is excluded, bad-tag candidates were filtered by the API params
    inc = [c for c in md.calls if c[0] == "/manga" and "includedTags[]" in c[1]]
    assert inc and "t-gore" in inc[0][1]["excludedTags[]"] and inc[0][1]["contentRating[]"] == ["safe", "suggestive"]


def test_pick_fails_loudly_when_nothing_readable(md, monkeypatch):
    cfg.manga_min_chapters = 50
    with pytest.raises(manga.MangaDexError, match="MANGA_ID"):
        manga.pick()


def test_manga_id_override(md):
    cfg.manga_id = "m-b"
    e = manga.pick()
    assert e["id"] == "m-b" and e["reason"] == "MANGA_ID"
    cfg.manga_id = ""


def test_download_and_covers(md, tmp_path):
    ch = manga.readable_chapters("m-a")[0]
    files = manga.download_chapter(ch, tmp_path / "pages", max_pages=5)
    assert len(files) == 5 and all(f.stat().st_size > 100 for f in files)
    # images served by the MD@Home network are reported (success/bytes/duration), uploads.mangadex.org covers are not
    assert len(md.reports) == 5 and all(u == manga.REPORT and b["success"] and "/data-saver/h1/" in b["url"] for u, b in md.reports)
    cv = manga.covers("m-a")
    assert len(cv) == 2 and cv[0].name.startswith("v1")
    assert len(md.reports) == 5


def test_crop_box(tmp_path):
    src = tmp_path / "p.png"
    Image.new("RGB", (800, 1200), (255, 255, 255)).save(src)
    out = manga.crop_box(src, [100, 100, 500, 600], tmp_path / "c.png")
    assert out and Image.open(out).size[0] > 200
    assert manga.crop_box(src, [10, 10, 12, 12], tmp_path / "tiny.png") is None
    assert manga.crop_box(src, "bad", tmp_path / "x.png") is None


# ------------------------------------------------------------------ the whole adaptation flow with a fake LLM
def fake_llm(prompt, *, images=None, temperature=1.0, max_tokens=0):
    if "head character designer" in prompt:
        return {"title_ja": "灰の刃", "title_ar": "نصل الرماد", "logline": "A boy hunts demons.", "genre": "dark fantasy", "tone": "grim",
                "world": "Demons roam at night.", "style": "dark fantasy, taisho era, moody lighting",
                "characters": [
                    {"id": "Ren", "name_ja": "蓮", "name_ar": "رِن", "gender": "male", "voice": "hero", "role": "protagonist",
                     "personality": "kind", "look": "1boy, black hair, scar, green checkered haori", "eyes": "large amber eyes, upturned lashes",
                     "ref": {"image": 0, "box_2d": [50, 100, 900, 800], "face_box_2d": [60, 250, 330, 650]}},
                    {"id": "mei", "name_ja": "芽衣", "name_ar": "مي", "gender": "female", "voice": "girl", "role": "ally", "personality": "bold",
                     "look": "1girl, long black hair, pink kimono", "eyes": "round pink eyes", "ref": None},
                    {"id": "kuro", "name_ja": "黒", "name_ar": "كورو", "gender": "male", "voice": "villain", "role": "antagonist",
                     "personality": "cold", "look": "1boy, white hair, red eyes", "eyes": "slit crimson pupils"}],
                "locations": [{"id": "village", "name": "Village", "look": "mountain village at dusk"}], "arc": "A long arc."}
    if "consecutive pages" in prompt:
        n = len(images or [])
        return {"pages": [{"n": i + 1, "summary": f"page {i + 1} summary", "dialogue": ["Hello"], "characters": ["ren"],
                           "intensity": 2 + i % 3, "key": i % 5 == 0} for i in range(n)], "recap": "so far"}
    if "Plan EPISODE" in prompt:
        return {"episode_title_ja": "始まり", "episode_title_ar": "البداية", "episode_summary": "It begins.",
                "parts": [{"part": i, "title_ja": "x", "title_ar": f"جزء {i}", "beat": f"beat {i}", "locations": ["village"], "characters": ["ren"],
                           "set_piece": "fight", "mood": "dark", "hook": "hook"} for i in range(1, 8)]}
    if "You are directing PART" in prompt:
        imgs = [{"id": f"i{i:02d}", "scene": "s1", "prompt": f"scene {i}", "characters": ["ren"], "framing": "medium"} for i in range(1, 17)]
        imgs[1]["hero"] = "ren"
        shots = []
        for i in range(40):
            sh = {"image": f"i{i % 16 + 1:02d}", "type": "medium", "dur": 3.0, "framing": {"zoom": 1, "cx": .5, "cy": .5},
                  "camera": {"move": "push_in", "strength": .5}, "fx": ["dust"], "grade": "dark"}
            if i % 3 == 0:
                sh["line"] = {"speaker": "ren", "ja": "行くぞ", "ar": "هيا بنا", "emotion": "shout", "intensity": 3}
            if i in (0, 8, 20, 39):
                sh["eyes"] = "closeup"
            if i == 5:
                sh["title_card"] = "series"
            shots.append(sh)
        return {"meta": {"title_ja": "t", "title_ar": "عنوان", "description_ar": "وصف", "summary": "He fights.", "tags": ["انمي"], "hashtags": ["#anime"]},
                "images": imgs, "shots": shots}
    raise AssertionError("unexpected prompt: " + prompt[:80])


def test_ingest_builds_bible_digest_outline_and_script(md, monkeypatch, tmp_path):
    from mangaforge import llm
    calls = []

    def spy(prompt, **kw):
        calls.append(("img" if kw.get("images") else "txt", prompt[:40]))
        return fake_llm(prompt, **kw)

    monkeypatch.setattr(llm, "ask_json", spy)
    monkeypatch.setattr(manga, "save_state", lambda s: None)
    monkeypatch.setattr(story, "log", story.log)
    from mangaforge.config import new_state
    st = new_state()
    cfg.pilot = True
    script = manga.ingest(st, 1, tmp_path)
    assert st["manga"]["id"] == "m-a"
    b = st["bible"]
    assert b["source"]["manga_id"] == "m-a" and {c["id"] for c in b["characters"]} == {"ren", "mei", "kuro"}
    ren = next(c for c in b["characters"] if c["id"] == "ren")
    assert ren["eyes"].startswith("large amber") and ren["ref"]["colour"] is True            # image 0 is a cover
    assert (pathlib.Path(manga.STATE_DIR).parent / ren["ref"]["face"]).exists() or (pathlib.Path(ren["ref"]["face"]).exists())
    assert st["week_outline"]["episode"] == 1 and len(st["week_outline"]["parts"]) == 7
    d = json.loads((manga.STATE_DIR / "manga" / "m-a" / "digest_e01.json").read_text("utf-8"))
    assert len(d["pages"]) == (18 + 1) + (18 + 2) or len(d["pages"]) > 30                    # 2 chapters worth of pages
    assert d["chapters"][0]["chapter"] == "1"
    assert script["meta"]["pilot"] is True and script["meta"]["source"]["title"] == "Blade of Ash"
    assert any(s.get("title_card") == "series" for s in script["shots"]) and sum(1 for s in script["shots"] if s.get("eyes")) >= 3
    assert next(i for i in script["images"] if i["id"] == "i02")["hero"] == "ren"
    assert abs(sum(s["dur"] for s in script["shots"]) - cfg.short_seconds) < 8
    assert story._check(script, True) == []
    # part 2 reuses everything (no new bible / digest / outline calls)
    n0 = len(calls)
    st["part"] = 2
    manga.ingest(st, 2, tmp_path)
    assert [c for c in calls[n0:] if "head character" in c[1] or "consecutive" in c[1] or "Plan EPISODE" in c[1]] == []
    assert manga.credits(st)[0].startswith("مقتبس من المانغا: Blade of Ash")
