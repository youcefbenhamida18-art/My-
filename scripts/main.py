"""Entry point used by the GitHub Actions workflow.

For every NEW lesson found under lessons/ (or the one given in LESSON_PATH):
  Gemini writes the whole script  ->  edge-tts speaks it  ->  the lesson is cut into videos that are each
  shorter than VIDEO_MAX_MIN minutes (default 13: YouTube refuses > 15 min on channels that are not verified)
  ->  a whiteboard video is rendered for each part  ->  uploaded to YouTube as PRIVATE.
"""
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import board
import llm
import sources
import tts
import youtube
from common import (OUT, ROOT, WORK, die, endgroup, env, fmt_ts, group, load_state, log, now_iso, save_state,
                    sha, slugify)

TEST = env("TEST_MODE", False, bool)
UPLOAD = env("UPLOAD", True, bool) and not TEST
LESSON_PATH = (env("LESSON_PATH", "") or "").strip().strip("/")
PART_MIN = env("PART_MINUTES", 35, int)            # how the AI writes the script (keep: it keeps the cache valid)
VIDEO_MAX_MIN = env("VIDEO_MAX_MIN", 13.0, float)  # every uploaded video is shorter than this (minutes)
MAX_TOTAL = env("MAX_VIDEO_MINUTES", 120, int)
MAX_LESSONS = env("MAX_LESSONS", 3, int)
TEST_SECTIONS = env("TEST_SECTIONS", 2, int)
W, H, FPS = env("VIDEO_W", 1920, int), env("VIDEO_H", 1080, int), env("FPS", 15, int)
THEME = env("BOARD_THEME", "chalk")
STATE_VERSION = 2


def summary(line: str):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def description(outline: dict, chapters: list, part_info: str) -> str:
    lines = [outline.get("summary", ""), ""]
    ch = []
    for t, name in chapters:
        if not ch:
            t = 0.0
        if ch and t - ch[-1][0] < 12:
            continue
        ch.append((t, name))
    if len(ch) >= 3:
        lines += [f"{fmt_ts(t)} {name}" for t, name in ch] + [""]
    lines.append(f"Private study video{part_info} - generated automatically from the lecture files.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cutting into videos
def split_ranges(steps, audios, limit_s: float) -> list:
    """Balanced cut into videos of about <= limit_s seconds, cutting where the board changes when possible."""
    n_steps = len(steps)
    total = sum(a.total for a in audios)
    n = max(1, math.ceil(total / limit_s))
    if n == 1 or n_steps < 2:
        return [(0, n_steps)]
    cum = [0.0]
    for a in audios:
        cum.append(cum[-1] + a.total)
    cand = [i for i in range(1, n_steps) if steps[i].new_board] or list(range(1, n_steps))
    cuts, prev = [], 0
    for k in range(1, n):
        goal = total * k / n
        opts = [i for i in cand if i > prev and n_steps - i >= n - k]
        if not opts:
            break
        prev = min(opts, key=lambda i: abs(cum[i] - goal))
        cuts.append(prev)
    b = [0] + cuts + [n_steps]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def final_ranges(steps, audios, ctx) -> list:
    """Cut the lesson, then check every piece with the real planner (writing time included)."""
    limit = VIDEO_MAX_MIN * 60 * 0.97
    hard = VIDEO_MAX_MIN * 60 + 40  # ~15 s are added later for the intro/outro of a part
    out = []

    def rec(lo, hi, lim, depth):
        for a, b in [(lo + x, lo + y) for x, y in split_ranges(steps[lo:hi], audios[lo:hi], lim)]:
            total = board.plan(steps[a:b], audios[a:b], ctx)[3]
            if total <= hard or b - a < 2 or depth >= 5:
                out.append((a, b))
            else:  # still too long: cut this piece in two again
                rec(a, b, sum(x.total for x in audios[a:b]) / 2 + 1, depth + 1)

    rec(0, len(steps), limit, 0)
    return out


def intro_step(title: str, part: int, total: int):
    return llm.Step(
        say=[("en", f"Welcome back! This is part {part} of {total}. Here we continue with: {title}."),
             ("ar", "أهلاً بك من جديد، نكمل الدرس.")],
        board=[{"type": "box", "text": f"Part {part}/{total}: {title}", "color": "yellow", "indent": 0}],
        new_board=True, board_title=f"Part {part} of {total}", pause=0.0, section=-1, section_title=title)


def outro_step(next_title: str, part: int, total: int):
    return llm.Step(
        say=[("en", f"That is the end of part {part}. In the next video we continue with: {next_title}. See you there!"),
             ("ar", "انتهى هذا الجزء، وسنكمل في الفيديو التالي.")],
        board=[{"type": "box", "text": f"Next video: {next_title}", "color": "green", "indent": 0}],
        new_board=False, board_title="", pause=0.0, section=-2, section_title=next_title)


def make_video(ctx, steps, audios, label: str):
    """Plan + render one video. Returns (mp4_path, chapters, seconds)."""
    events, step_len, chapters, total = board.plan(steps, audios, ctx)
    log(f"  video length: {fmt_ts(total)} ({len(steps)} steps, {len(events)} board events)")
    WORK.mkdir(parents=True, exist_ok=True)
    wav, mp4 = WORK / f"{label}.wav", WORK / f"{label}.mp4"
    board.write_wav(audios, step_len, total, wav)
    board.render_video(events, total, ctx, wav, mp4, FPS)
    wav.unlink(missing_ok=True)
    log(f"  rendered {mp4.name} ({mp4.stat().st_size / 1e6:.0f} MB)")
    return mp4, chapters, total


def process(lesson, state, gem, families):
    group(f"Lesson: {lesson.key}")
    try:
        sources.prepare(lesson)
        if not any(f.role == "lecture" for f in lesson.files):
            raise RuntimeError("no readable lecture file in this lesson")
        log(sources.catalog(lesson))
        outline = gem.outline(lesson, MAX_TOTAL, TEST)
        gparts = llm.split_parts(outline, PART_MIN)
        if TEST:
            gparts = [gparts[0][:TEST_SECTIONS]]
        log(f"  outline: {len(outline['sections'])} sections")

        # 1) the whole script (answers are cached on disk, so re-runs are instant)
        steps, prev = [], ""
        for pi, idxs in enumerate(gparts, 1):
            for j, si in enumerate(idxs):
                st, prev = gem.section_steps(
                    lesson, outline, si, pi, len(gparts), prev,
                    is_last_overall=(pi == len(gparts) and j == len(idxs) - 1 and not TEST),
                    is_last_in_part=(j == len(idxs) - 1), is_first_in_part=(j == 0))
                steps += st

        # 2) the voice for everything
        audios = tts.build_audio(steps)

        # 3) cut into videos shorter than VIDEO_MAX_MIN
        srcs = {i: (f.pdf or f.image) for i, f in enumerate(lesson.files, 1) if (f.pdf or f.image)}
        ctx = board.Ctx(W, H, THEME, families, srcs)
        ranges = final_ranges(steps, audios, ctx)
        n = len(ranges)
        log(f"  speech {fmt_ts(sum(a.total for a in audios))} -> {n} video(s), each shorter than {VIDEO_MAX_MIN:g} min")

        if UPLOAD:
            entry = state.get(lesson.key)
            if not entry or entry.get("v") != STATE_VERSION:
                entry = state[lesson.key] = {"v": STATE_VERSION, "title": outline["video_title"], "parts": {}}
            entry.update({"parts_total": n, "done": False})
        else:
            entry = {"parts": {}}

        # 4) render + upload every part
        for idx, (lo, hi) in enumerate(ranges, 1):
            c_steps, c_aud = list(steps[lo:hi]), list(audios[lo:hi])
            fp = sha(*[t for st in c_steps for _, t in st.say])
            if UPLOAD and entry["parts"].get(str(idx), {}).get("fp") == fp:
                log(f"  part {idx}/{n} already uploaded, skipping")
                continue
            if n > 1:
                if idx > 1:
                    s = intro_step(c_steps[0].section_title, idx, n)
                    c_steps.insert(0, s)
                    c_aud.insert(0, tts.build_audio([s])[0])
                if idx < n:
                    s = outro_step(steps[hi].section_title, idx, n)
                    c_steps.append(s)
                    c_aud.append(tts.build_audio([s])[0])
            title = outline["video_title"] + (f" - Part {idx}/{n}" if n > 1 else "")
            mp4, chapters, total = make_video(ctx, c_steps, c_aud, f"{slugify(lesson.name)}-p{idx}")
            if total >= 14.9 * 60:
                log(f"  WARNING: this video is {fmt_ts(total)}; YouTube may refuse it on an unverified channel")
            if UPLOAD:
                desc = description(outline, chapters, f" (part {idx}/{n})" if n > 1 else "")
                vid = youtube.upload(mp4, title, desc, tags=["study", "pharmacy", "lecture", "whiteboard"])
                entry["parts"][str(idx)] = {"video_id": vid, "url": f"https://youtu.be/{vid}", "title": title,
                                            "minutes": round(total / 60, 1), "fp": fp, "uploaded_at": now_iso()}
                entry["done"] = all(str(i) in entry["parts"] for i in range(1, n + 1))
                save_state(state)
                summary(f"- [{title}](https://youtu.be/{vid}) ({total / 60:.1f} min, private)")
                mp4.unlink(missing_ok=True)
            else:
                OUT.mkdir(parents=True, exist_ok=True)
                shutil.move(str(mp4), str(OUT / mp4.name))
                log(f"  saved to out/{mp4.name} (not uploaded)")
                summary(f"- `out/{mp4.name}` ({total / 60:.1f} min) saved as an artifact (not uploaded)")
        return True
    except SystemExit:
        raise
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"::error title=Lesson failed::{lesson.key}: {e}", flush=True)
        summary(f"- FAILED `{lesson.key}`: {e}")
        return False
    finally:
        endgroup()


def main():
    if not llm.collect_keys():
        die("Secret GEMINI_API_KEY is missing. Create a free key at https://aistudio.google.com/apikey and add it "
            "in Settings > Secrets and variables > Actions. (More keys: GEMINI_API_KEY_2 ... GEMINI_API_KEY_10)")
    if UPLOAD:
        youtube.check_credentials()
    state = load_state()
    lessons = sources.discover(ROOT / "lessons")
    if LESSON_PATH:
        chosen = [l for l in lessons if l.key == LESSON_PATH or l.key.startswith(LESSON_PATH + "/")]
        if not chosen:
            die(f"No lesson found at '{LESSON_PATH}'. Existing: " + ", ".join(l.key for l in lessons))
        for l in chosen:  # forced re-run: forget the old upload record
            state.pop(l.key, None)
    elif TEST:
        chosen = lessons[:1]
    else:
        def finished(l):
            e = state.get(l.key, {})
            return bool(e.get("done")) and e.get("v") == STATE_VERSION
        chosen = [l for l in lessons if not finished(l)]
    if not chosen:
        log("No new lessons to process. Add files under lessons/ and push.")
        summary("No new lessons to process.")
        return
    if len(chosen) > MAX_LESSONS:
        log(f"{len(chosen)} lessons are waiting; doing {MAX_LESSONS} now - run the workflow again for the rest.")
        summary(f"{len(chosen) - MAX_LESSONS} more lesson(s) waiting - run the workflow again.")
        chosen = chosen[:MAX_LESSONS]
    log(f"Mode: {'TEST (2 sections, no upload)' if TEST else 'upload' if UPLOAD else 'no upload'}; "
        f"lessons: {[l.key for l in chosen]}")
    families = board.setup_fonts()
    gem = llm.Gemini()
    ok = [process(l, state, gem, families) for l in chosen]
    if not all(ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
