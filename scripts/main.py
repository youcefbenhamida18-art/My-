"""Entry point used by the GitHub Actions workflow.

For every NEW lesson found under lessons/ (or the one given in LESSON_PATH):
  Gemini writes an outline + a deep step-by-step script  ->  edge-tts speaks it  ->
  a whiteboard video is rendered with ffmpeg             ->  uploaded to YouTube as PRIVATE.
"""
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
                    slugify)

TEST = env("TEST_MODE", False, bool)
UPLOAD = env("UPLOAD", True, bool) and not TEST
LESSON_PATH = (env("LESSON_PATH", "") or "").strip().strip("/")
PART_MIN = env("PART_MINUTES", 35, int)
MAX_TOTAL = env("MAX_VIDEO_MINUTES", 120, int)
MAX_LESSONS = env("MAX_LESSONS", 3, int)
TEST_SECTIONS = env("TEST_SECTIONS", 2, int)
W, H, FPS = env("VIDEO_W", 1920, int), env("VIDEO_H", 1080, int), env("FPS", 15, int)
THEME = env("BOARD_THEME", "chalk")


def summary(line: str):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def description(outline: dict, chapters: list, title: str, part_info: str) -> str:
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


def make_video(lesson, outline, steps, label: str, families):
    """Voice -> plan -> render. Returns (mp4_path, chapters, seconds)."""
    audios = tts.build_audio(steps)
    srcs = {i: (f.pdf or f.image) for i, f in enumerate(lesson.files, 1) if (f.pdf or f.image)}
    ctx = board.Ctx(W, H, THEME, families, srcs)
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
        parts = llm.split_parts(outline, PART_MIN)
        if TEST:
            parts = [parts[0][:TEST_SECTIONS]]
        log(f"  outline: {len(outline['sections'])} sections -> {len(parts)} video part(s)")
        entry = state.setdefault(lesson.key, {"title": outline["video_title"], "parts_total": len(parts),
                                              "parts": {}, "done": False}) if UPLOAD else {"parts": {}}
        prev = ""
        for pi, idxs in enumerate(parts, 1):
            if UPLOAD and str(pi) in entry["parts"]:
                prev = entry["parts"][str(pi)].get("recap", "")
                log(f"  part {pi} already uploaded, skipping")
                continue
            steps = []
            for j, si in enumerate(idxs):
                st, prev = gem.section_steps(
                    lesson, outline, si, pi, len(parts), prev,
                    is_last_overall=(pi == len(parts) and j == len(idxs) - 1 and not TEST),
                    is_last_in_part=(j == len(idxs) - 1), is_first_in_part=(j == 0))
                steps += st
            multi = len(parts) > 1
            title = outline["video_title"] + (f" - Part {pi}/{len(parts)}" if multi else "")
            label = f"{slugify(lesson.name)}-p{pi}"
            mp4, chapters, total = make_video(lesson, outline, steps, label, families)
            if UPLOAD:
                desc = description(outline, chapters, title, f" (part {pi}/{len(parts)})" if multi else "")
                vid = youtube.upload(mp4, title, desc, tags=["study", "pharmacy", "lecture", "whiteboard"])
                entry["parts"][str(pi)] = {"video_id": vid, "url": f"https://youtu.be/{vid}", "title": title,
                                           "recap": prev, "uploaded_at": now_iso()}
                entry["done"] = len(entry["parts"]) == len(parts)
                save_state(state)
                summary(f"- [{title}](https://youtu.be/{vid}) (private)")
                mp4.unlink(missing_ok=True)
            else:
                OUT.mkdir(parents=True, exist_ok=True)
                shutil.move(str(mp4), str(OUT / mp4.name))
                log(f"  saved to out/{mp4.name} (not uploaded)")
                summary(f"- `out/{mp4.name}` saved as an artifact (not uploaded)")
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
        chosen = [l for l in lessons if not state.get(l.key, {}).get("done")]
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
