"""Orchestration. Every stage is resumable and reads/writes plain files under work/<job>/, so the same code runs in one go on
a laptop (`short`) or sharded across GitHub Actions jobs (`prepare` -> `render --shard i/n` x N -> `assemble`).

    prepare  : story (or manga) -> script.json, voices.json, art.json (+ AI motion), shots.json         [LLM + art + TTS]
    render   : shots.json -> shots/shot_XXXX.mp4 for this shard                                           [Blender / cv + ffmpeg]
    assemble : shots -> out/short_<job>.mp4 (+ vertical, thumbnail, credits)                              [ffmpeg]
    weekly   : the 7 shorts of an episode -> out/episode_<n>.mp4 with title card + chapters
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

from . import art as art_pkg
from . import assemble, director, render, schema, story, voice
from .config import OUT, WORK_ROOT, cfg, load_state, log, rebase_paths, save_state, work_dir


class PipelineError(RuntimeError):
    pass


def job_id(state):
    return f"e{state['episode']:02d}p{state['part']}"


def _jload(p):
    p = pathlib.Path(p)
    obj = json.loads(p.read_text("utf-8"))
    return rebase_paths(obj, p.parent.name) if p.parent.parent == WORK_ROOT else obj


def _jsave(p, obj):
    pathlib.Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1), "utf-8")


# ------------------------------------------------------------------ story / bible / outline
def ensure_bible(state):
    if state.get("bible"):
        return state["bible"]
    if cfg.source == "manga":
        return {"title_ja": "", "title_ar": "", "style": "", "characters": [], "locations": [], "world": "", "tone": ""}
    log.info("creating the series bible ...")
    state["bible"] = story.make_bible()
    save_state(state)
    return state["bible"]


def ensure_outline(state, bible):
    o = state.get("week_outline")
    if o and o.get("episode") == state["episode"] and len(o.get("parts", [])) >= cfg.parts_per_episode:
        return o
    log.info("planning episode %s ...", state["episode"])
    state["week_outline"] = story.write_week_outline(bible, state)
    save_state(state)
    return state["week_outline"]


def make_script(state, W):
    part = state["part"]
    if cfg.source == "manga":
        from . import manga
        script = manga.ingest(state, part, W)
    else:
        bible = ensure_bible(state)
        ensure_outline(state, bible)
        script = story.write_short(bible, state, part)
    return script


# ------------------------------------------------------------------ AI motion (optional hero shots)
def choose_i2v(script, art, budget):
    if budget <= 0:
        return []
    scored = []
    for i, s in enumerate(script["shots"]):
        sc = {"impact": 3.0, "action": 2.2, "closeup": 1.6, "establishing": 1.4, "wide": 1.0}.get(s["type"], 0.5)
        if s.get("line"):
            sc += 0.15 * s["line"]["intensity"]
        if i == 0:
            sc += 1.0
        scored.append((sc, i))
    picked, seen_img = [], set()
    for sc, i in sorted(scored, reverse=True):
        img = script["shots"][i]["image"]
        if img in seen_img:
            continue
        seen_img.add(img)
        picked.append(i)
        if len(picked) >= budget:
            break
    return sorted(picked)


def make_sequences(script, art, W):
    if cfg.i2v_backend == "off":
        return {}
    from . import i2v
    seqs = {}
    imgs = {im["id"]: im for im in script["images"]}
    for i in choose_i2v(script, art, cfg.i2v_budget):
        s = script["shots"][i]
        a = art[s["image"]]
        try:
            seq = i2v.animate(a["image"], imgs[s["image"]]["prompt"], W / "i2v" / f"shot_{s['id']}", seconds=min(cfg.i2v_seconds, s["dur"] + 0.5),
                              fps=cfg.fps, seed=1000 + s["id"])
            if seq:
                seqs[s["id"]] = seq
        except Exception as e:  # noqa: BLE001
            log.warning("AI motion skipped for shot %s: %s", s["id"], e)
    log.info("AI motion: %d hero shot(s) animated", len(seqs))
    return seqs


# ------------------------------------------------------------------ stages
def prepare(job=None, *, force=False, state=None):
    state = state or load_state()
    job = job or job_id(state)
    W = work_dir(job)
    sp = W / "script.json"
    if sp.exists() and not force:
        script = _jload(sp)
        log.info("prepare: reusing %s", sp)
    else:
        script = make_script(state, W)
        _jsave(sp, script)
    st = schema.script_stats(script)
    log.info("script: %s", st)

    vp = W / "voices.json"
    voices = None
    if vp.exists() and not force:
        voices = {int(k): v for k, v in _jload(vp).items()}
        if not all(pathlib.Path(v["path"]).exists() for v in voices.values()):
            voices = None
    if voices is None:
        voices = voice.synthesize(script, W)
        _jsave(vp, voices)
    arts = art_pkg.generate_art(script, W, progress=lambda m: log.info("art: %s", m))
    _jsave(W / "art.json", arts)
    seqs = make_sequences(script, arts, W)
    shots = director.plan(script, voices, arts, job, sequences=seqs)
    (WORK_ROOT / "current_job.txt").write_text(job, "utf-8")
    return {"job": job, "stats": st, "shots": len(shots), "seconds": round(sum(s["dur"] for s in shots), 1)}


def render_stage(job, *, shard=None, only=None, deadline=None):
    shots = render.load_shots(job)
    return render.render_shots(shots, shard=shard, only=only, deadline=deadline)


def assemble_stage(job):
    W = work_dir(job)
    shots = render.load_shots(job)
    script = _jload(W / "script.json")
    voices = {int(k): v for k, v in _jload(W / "voices.json").items()} if (W / "voices.json").exists() else {}
    missing = [s["idx"] for s in shots if not pathlib.Path(s["paths"]["final"]).exists()]
    if missing:
        raise PipelineError(f"{len(missing)} shot(s) not rendered yet: {missing[:12]} - run `render` first")
    return assemble.assemble_short(shots, script, job, voices=voices)


def _advance(state, job, script, res):
    today = dt.date.today().isoformat()
    meta = script.get("meta", {})
    state["shorts"].append({"job": job, "episode": state["episode"], "part": state["part"], "title_ar": meta.get("title_ar", ""),
                            "summary": meta.get("summary", ""), "video": res["video"], "thumbnail": res.get("thumbnail"),
                            "vertical": res.get("vertical"), "duration": res["duration"], "credits": res.get("credits", []),
                            "date": today})
    state["last_day"] = today
    state["part"] += 1
    save_state(state)


def run_short(*, job=None, state=None):
    """Everything for one daily short on this machine. Returns the result dict."""
    state = state or load_state()
    if state["part"] > cfg.parts_per_episode:
        log.info("the week is complete - assembling the episode first")
        run_weekly(state=state)
        state = load_state()
    job = job or job_id(state)
    info = prepare(job, state=state)
    log.info("prepared %s: %s", job, info)
    render_stage(job)
    res = assemble_stage(job)
    script = _jload(work_dir(job) / "script.json")
    _advance(state, job, script, res)
    res["job"] = job
    if state["part"] > cfg.parts_per_episode and cfg.auto_weekly:
        res["weekly"] = run_weekly(state=state)
    return res


def collect_parts(state, parts_dir=None):
    ep = state["episode"]
    items = [s for s in state["shorts"] if s.get("episode") == ep]
    items.sort(key=lambda s: s["part"])
    out = []
    for s in items:
        v = pathlib.Path(s["video"])
        if not v.exists() and parts_dir:
            for cand in (pathlib.Path(parts_dir) / f"short_{s['job']}.mp4", pathlib.Path(parts_dir) / v.name):
                if cand.exists():
                    v = cand
                    break
        if not v.exists():
            raise PipelineError(f"video for part {s['part']} ({s['job']}) not found at {s['video']}"
                                + (f" or in {parts_dir}" if parts_dir else "") + " - pass --parts-dir")
        out.append({**s, "video": str(v)})
    return out


def run_weekly(*, state=None, parts_dir=None):
    state = state or load_state()
    parts = collect_parts(state, parts_dir)
    if len(parts) < cfg.parts_per_episode:
        log.warning("only %d/%d shorts exist for episode %s - assembling what is there", len(parts), cfg.parts_per_episode, state["episode"])
    if not parts:
        raise PipelineError("no shorts to join")
    outline = state.get("week_outline") or {}
    last = next((s for s in state["shorts"] if s["job"] == parts[0]["job"]), {})
    bible = state.get("bible") or {}
    meta = {"series_title_ar": bible.get("title_ar", ""), "episode": state["episode"],
            "episode_title_ar": outline.get("episode_title_ar", "")}
    out = OUT / f"episode_{state['episode']:02d}.mp4"
    res = assemble.assemble_weekly(parts, meta, out)
    state["episodes"].append({"episode": state["episode"], "video": res["video"], "duration": res["duration"],
                              "summary": outline.get("episode_summary", ""), "title_ar": meta["episode_title_ar"],
                              "chapters": res["chapters"], "date": dt.date.today().isoformat(), "last_job": last.get("job")})
    state["episode"] += 1
    state["part"] = 1
    state["week_outline"] = None
    save_state(state)
    res["episode"] = state["episode"] - 1
    return res


def describe(state, job, *, weekly=None):
    """YouTube title/description/tags for a short (or the weekly episode)."""
    W = work_dir(job) if job else None
    bible = state.get("bible") or {}
    if weekly:
        ep = weekly
        title = f"{bible.get('title_ar', '')} | {ep.get('title_ar') or 'الحلقة ' + str(ep['episode'])}".strip(" |")
        desc = (ep.get("summary") or "") + "\n\n" + "\n".join(ep.get("chapters", []))
        return title, desc, ["انمي", "anime", "الحلقة كاملة"]
    script = _jload(W / "script.json")
    meta = script.get("meta", {})
    res = _jload(W / "out" / "result.json")
    title = f"{meta.get('title_ar', '')} | {bible.get('title_ar', '')}".strip(" |")
    credits = "\n".join(res.get("credits", []))
    desc = f"{meta.get('description_ar', '')}\n\n{' '.join(meta.get('hashtags', []))}\n\nالأصوات: {credits}\nمحتوى مولَّد بالذكاء الاصطناعي (صور وأصوات)."
    return title, desc, list(meta.get("tags", []))[:15]
