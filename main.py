"""MangaForge Studio: one run = one ~10 min episode (16:9) + one vertical trailer (<= 3 min) + scheduled upload."""
import argparse
import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import shutil

import cv2
import numpy as np

from . import art, audio, engine, story, subs
from .config import (FPS, EPISODE_MIN, ROOT, TRAILER_SEC, VH, VW, W, H, WORK, WORKERS, ACTS,
                     load_state, log, media_duration, run, save_state)

MOVES_FIRST = ["pull_out", "pan_l", "pan_r", "orbit"]
MOVES_NEXT = ["push_in", "drift", "tilt_up", "orbit", "pan_r", "pan_l", "tilt_dn"]
FOCS = [(0, 0), (-0.10, 0.04), (0.10, -0.03), (0.0, -0.07)]
LEAD, TAIL = 0.12, 0.42


# ------------------------------------------------------------------ planning
def plan(scenes, imgs, voices, size, tag="e"):
    """One shot per spoken line; several camera setups per image so a single painting plays as a whole scene."""
    shots = []
    for si, sc in enumerate(scenes):
        for li, ln in enumerate(sc["lines"]):
            vpath, vdur = voices[(si, li)]
            impact = ln["intensity"] >= 5 or (bool(sc.get("impact")) and li == 0)
            if impact:
                move = "punch"
            elif li == 0:
                move = MOVES_FIRST[si % len(MOVES_FIRST)]
            else:
                move = MOVES_NEXT[(si + li) % len(MOVES_NEXT)]
            trans = "cut"
            if li == 0:
                trans = "flash" if (impact or si % 2 == 0) else "whip"
            if not shots:
                trans = "fade"
            idx = len(shots)
            shots.append(dict(
                idx=idx, scene=si, act=sc["act"], img=imgs[si][0], depth=imgs[si][1], size=size,
                frames=max(int(FPS * 2.2), math.ceil((LEAD + vdur + TAIL) * FPS)), fps=FPS, move=move,
                mood=sc["mood"], intensity=ln["intensity"], impact=impact, trans=trans, fade_out=False,
                act_end=False, ar=ln["ar"], ja=ln["ja"], zoom_bias=1.0 if li == 0 else 1.12 + 0.10 * (li % 2),
                foc_off=FOCS[li % 4], seed=1000 * si + li, voice=vpath, out=str(WORK / f"{tag}_{idx:04d}.mp4")))
    for a in {s["act"] for s in shots}:
        last = [s for s in shots if s["act"] == a][-1]
        last["act_end"] = last["fade_out"] = True
    return shots


def pick_trailer(shots):
    """Hook + cliffhanger are forced in; the rest is the most intense moments, kept in story order."""
    budget = TRAILER_SEC - 5.0
    chosen = {0, shots[-1]["idx"]}
    used = sum(shots[i]["frames"] for i in chosen) / FPS
    score = lambda s: s["intensity"] + (2 if s["impact"] else 0) + (0.5 if s["act_end"] else 0)  # noqa: E731
    for s in sorted(shots, key=lambda s: -score(s)):
        d = s["frames"] / FPS
        if s["idx"] not in chosen and used + d <= budget:
            chosen.add(s["idx"])
            used += d
    out = []
    for n, i in enumerate(sorted(chosen)):
        out.append(dict(shots[i], size=(VW, VH), out=str(WORK / f"t_{n:04d}.mp4"),
                        trans="fade" if n == 0 else "flash", fade_out=False))
    last = out[-1]
    out.append(dict(last, idx=len(out), frames=int(4 * FPS), ar="الحلقة الكاملة على القناة", ja="", voice=None,
                    move="push_in", intensity=2, impact=False, act_end=False, trans="flash", fade_out=True,
                    out=str(WORK / f"t_{len(out):04d}.mp4")))
    return out


# ------------------------------------------------------------------ rendering / assembly
def render_all(shots):
    log.info("rendering %d shots on %d workers", len(shots), WORKERS)
    with cf.ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as ex:
        for n, _ in enumerate(ex.map(engine.render_shot, shots, chunksize=1), 1):
            if n % 10 == 0:
                log.info("  %d/%d shots", n, len(shots))


def build(shots, tag, final):
    render_all(shots)
    lst = WORK / f"{tag}_list.txt"
    lst.write_text("".join(f"file '{s['out']}'\n" for s in shots))
    silent, wav = WORK / f"{tag}_silent.mp4", WORK / f"{tag}.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])
    total = audio.build_mix(shots, FPS, wav)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ac", "2", "-shortest", "-movflags", "+faststart", final])
    log.info("%s ready: %.0fs -> %s", tag, total, final)
    return total


def thumbnail(shots, video, title, out):
    best = max(shots, key=lambda s: s["intensity"] + (2 if s["impact"] else 0))
    t = sum(s["frames"] for s in shots[:best["idx"]]) / FPS + 0.6
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", t, "-i", video, "-frames:v", "1", "-vf", "scale=1280:720", out])
    fr = cv2.imread(str(out))
    col, alpha, y0 = subs.render(title, 1280, 720, size=84)
    y0 = 720 - alpha.shape[0] - 70
    reg = fr[y0:y0 + alpha.shape[0]]
    reg[:] = (reg * (1 - alpha[..., None]) + col * alpha[..., None]).astype(np.uint8)
    cv2.imwrite(str(out), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ------------------------------------------------------------------ main flows
def produce():
    state = load_state()
    dry = os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")
    WORK.mkdir(exist_ok=True)
    if not state["bible"]:
        log.info("creating the series bible")
        state["bible"] = story.make_bible()
        save_state(state)
    bible, ep = state["bible"], len(state["episodes"]) + 1

    log.info("writing episode %d", ep)
    script = story.write_episode(bible, state, ep)
    scenes = script["scenes"]
    log.info("%d scenes / %d lines", len(scenes), sum(len(s["lines"]) for s in scenes))

    imgs = art.generate(bible, scenes, WORK)
    voices = audio.synthesize(bible, scenes, ep, WORK)

    shots = plan(scenes, imgs, voices, (W, H))
    total = sum(s["frames"] for s in shots) / FPS
    log.info("planned runtime %.1f min (target %.0f)", total / 60, EPISODE_MIN)
    if total < EPISODE_MIN * 60 * 0.7:
        log.warning("episode is shorter than target - raise EPISODE_MINUTES or check Gemini output")

    final = ROOT / "out"
    final.mkdir(exist_ok=True)
    ep_mp4, tr_mp4, thumb = final / f"ep{ep:03d}.mp4", final / f"ep{ep:03d}_trailer.mp4", final / f"ep{ep:03d}.jpg"
    build(shots, "ep", ep_mp4)
    trailer = pick_trailer(shots)
    build(trailer, "trailer", tr_mp4)
    meta = script["meta"]
    thumbnail(shots, ep_mp4, meta["title_ar"], thumb)

    if dry:
        log.info("DRY_RUN: files are in %s", final)
        return
    from . import youtube
    yt = youtube.service()
    t1, t2, day = youtube.next_slots(state)
    tags = list(meta.get("tags", []))[:12]
    tag_line = " ".join(bible.get("hashtags", [])[:5])
    ep_id = youtube.upload(
        yt, ep_mp4, f"{bible['title_ar']} | {meta['title_ar']}",
        f"{meta['description_ar']}\n\n{bible['title_ja']} - Ep {ep}\nياباني مع ترجمة عربية\n{tag_line}", tags, t2, thumb)
    sh_id = youtube.upload(
        yt, tr_mp4, f"{meta['title_ar']} #Shorts",
        f"{meta['description_ar']}\n\nالحلقة الكاملة: https://youtu.be/{ep_id}\n#Shorts {tag_line}", tags + ["Shorts"], t1)
    state["episodes"].append({"n": ep, "title": meta["title_ar"], "summary": script["summary"],
                              "yt": ep_id, "short": sh_id})
    state["last_day"] = day.isoformat()
    save_state(state)
    shutil.rmtree(WORK, ignore_errors=True)


def selftest():
    """No network needed: synthetic art + tones -> proves engine, audio, concat and mux work on this machine."""
    d = WORK / "selftest"
    d.mkdir(parents=True, exist_ok=True)
    h, w = 768, 1344
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0], img[..., 2] = 40 + 80 * yy / h, 30 + 60 * (1 - yy / h)
    cv2.circle(img, (w // 2, h // 2), 180, (60, 60, 230), -1)
    cv2.rectangle(img, (100, 400), (420, 768), (20, 20, 20), -1)
    cv2.imwrite(str(d / "i.jpg"), cv2.GaussianBlur(img, (0, 0), 2))
    cv2.imwrite(str(d / "d.png"), (np.clip(1 - np.hypot(xx - w / 2, yy - h / 2) / 700, 0, 1) * 255).astype(np.uint8))
    moves = ["push_in", "pan_l", "orbit", "punch", "pull_out", "drift"]
    shots = []
    for i, mv in enumerate(moves):
        vp = d / f"v{i}.wav"
        audio.save_wav(vp, (0.3 * np.sin(2 * np.pi * (200 + 40 * i) * np.arange(int(1.2 * audio.SR)) / audio.SR)
                            ).astype(np.float32))
        shots.append(dict(idx=i, scene=0, act=1 + i // 3, img=str(d / "i.jpg"), depth=str(d / "d.png"), size=(640, 360),
                          frames=48, fps=FPS, move=mv, mood=["embers", "shadow", "rain", "aura", "dust", "snow"][i],
                          intensity=5 if mv == "punch" else 2, impact=mv == "punch", trans=["fade", "whip", "flash"][i % 3],
                          fade_out=i in (2, 5), act_end=i in (2, 5), ar="اختبار الترجمة العربية", ja="", zoom_bias=1.0,
                          foc_off=(0, 0), seed=i, voice=str(vp), out=str(d / f"s{i}.mp4")))
    build(shots, "selftest", d / "selftest.mp4")
    vshots = [dict(s, size=(360, 640), out=str(d / f"v{s['idx']}.mp4")) for s in shots[:3]]
    build(vshots, "selftest_v", d / "selftest_v.mp4")
    print("SELFTEST OK", media_duration(d / "selftest.mp4"), media_duration(d / "selftest_v.mp4"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else produce()
