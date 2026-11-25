"""Final assembly: shots -> one short (video + score + voices + ASS Arabic subtitles), thumbnail, weekly episode with chapters."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

from . import sound, subs
from .config import cfg, log, media_duration, render_size, work_dir


class AssembleError(RuntimeError):
    pass


def _ff(args, timeout=3600):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssembleError((r.stderr or "").strip()[-1500:] or f"ffmpeg exit {r.returncode}")


def _q(p):
    return str(p).replace("'", "'\\''")


def concat_videos(files, out, *, reencode=False):
    files = [pathlib.Path(f) for f in files]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise AssembleError(f"missing rendered files: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    lst = pathlib.Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{_q(f.resolve())}'\n" for f in files), "utf-8")
    try:
        if not reencode:
            _ff(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])
        else:
            _ff(["-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-preset", cfg.x264_preset, "-crf", cfg.crf,
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out])
    finally:
        lst.unlink(missing_ok=True)
    return pathlib.Path(out)


def _png_overlay_burn(video, audio, shots, out, size):
    """Fallback when ffmpeg has no libass: one transparent PNG per line, overlaid in its time window."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="subpng_"))
    try:
        inputs, chain, last, n = [], [], "0:v", 0
        for t0, t1, s in subs.timeline(shots):
            tx = s.get("text")
            if not tx or not tx.get("ar"):
                continue
            a = s.get("audio") or {}
            st = t0 + (a.get("voice_start") or 0.15)
            en = min(t1 - 0.05, st + (a.get("voice_dur") or (t1 - t0 - 0.4)) + 0.35)
            n += 1
            png = tmp / f"s{n:03d}.png"
            subs.render_line_png(tx["ar"], size).save(png)
            inputs += ["-loop", "1", "-i", png]
            chain.append(f"[{last}][{n}:v]overlay=shortest=1:enable='between(t,{st:.2f},{en:.2f})'[o{n}]")
            last = f"o{n}"
        graph = ";".join(chain) if chain else "[0:v]null[o0]"
        last = last if chain else "o0"
        _ff(["-i", video, "-i", audio, *inputs[0:0], *inputs, "-filter_complex", graph, "-map", f"[{last}]", "-map", "1:a",
             "-c:v", "libx264", "-preset", cfg.x264_preset, "-crf", cfg.crf, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-ar", "48000", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-shortest", "-movflags", "+faststart", out])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mux(video, audio, shots, out, *, size, vertical=False):
    """video + audio + Arabic subtitles burned in (libass when available; PNG overlays otherwise)."""
    if subs._ffmpeg_has_libass():
        ass = subs.write_ass(shots, pathlib.Path(out).with_suffix(".ass"), size=size, vertical=vertical)
        _ff(["-i", video, "-i", audio, "-vf", subs.burn_filter(ass), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset",
             cfg.x264_preset, "-crf", cfg.crf, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-shortest", "-movflags", "+faststart", out])
        return pathlib.Path(out), ass
    log.warning("ffmpeg has no libass: using the PNG subtitle fallback (slower). Install a full ffmpeg build for ASS.")
    vs = (size[1], size[0]) if vertical else size
    _png_overlay_burn(video, audio, shots, out, vs)
    return pathlib.Path(out), None


def make_thumbnail(video, shots, title_ar, part, out, *, frame_at=None):
    """1280x720 JPEG: a dramatic frame + the Arabic title (Raqm-shaped once) + a part badge."""
    dur = media_duration(video)
    t = frame_at
    if t is None:
        t0 = 0.0
        for s in shots:                                   # first impact/action shot, mid-way
            if s["type"] in ("impact", "action"):
                t = t0 + s["dur"] * 0.55
                break
            t0 += s["dur"]
    t = min(max(0.5, t if t is not None else dur * 0.35), max(0.5, dur - 0.5))
    tmp = pathlib.Path(out).with_suffix(".raw.png")
    _ff(["-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720", tmp])
    im = Image.open(tmp).convert("RGBA")
    if title_ar:
        shade = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(shade)
        for i in range(260):
            d.rectangle([0, 720 - i, 1280, 721 - i], fill=(0, 0, 0, int(190 * (1 - i / 260) ** 1.6)))
        im.alpha_composite(shade)
        try:
            im.alpha_composite(subs.render_line_png(title_ar, (1280, 720), px=84, bottom=44))
        except Exception as e:  # noqa: BLE001
            log.warning("thumbnail title skipped: %s", e)
    if part:
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([28, 28, 150, 92], 14, fill=(200, 30, 50, 235))
        try:
            f = ImageFont.truetype(subs.find_arabic_font(True)[0] or "", 40)
            d.text((89, 60), f"#{part}", font=f, fill=(255, 255, 255, 255), anchor="mm")
        except Exception:  # noqa: BLE001
            d.text((60, 48), f"#{part}", fill=(255, 255, 255, 255))
    im.convert("RGB").save(out, quality=90)
    tmp.unlink(missing_ok=True)
    return pathlib.Path(out)


def assemble_short(shots, script, job, *, voices=None, out_dir=None):
    """-> {"video", "vertical", "audio", "ass", "thumbnail", "duration", "credits"}"""
    W = work_dir(job)
    out_dir = pathlib.Path(out_dir or (W / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    size = render_size()
    finals = [s["paths"]["final"] for s in shots]
    silent = out_dir / "video_silent.mp4"
    concat_videos(finals, silent)
    audio = sound.build_soundtrack(shots, W / "audio.wav", seed=sum(map(ord, str(job))))
    video = out_dir / f"short_{job}.mp4"
    _, ass = mux(silent, audio, shots, video, size=size)
    res = {"video": str(video), "audio": str(audio), "ass": str(ass) if ass else None, "vertical": None}
    vfiles = [pathlib.Path(f).with_name(pathlib.Path(f).stem + "_v.mp4") for f in finals]
    if cfg.vertical and all(v.exists() for v in vfiles):
        vs = out_dir / "video_silent_v.mp4"
        concat_videos(vfiles, vs)
        vv = out_dir / f"short_{job}_vertical.mp4"
        mux(vs, audio, shots, vv, size=size, vertical=True)
        res["vertical"] = str(vv)
    silent.unlink(missing_ok=True)
    dur = media_duration(video)
    planned = sum(s["dur"] for s in shots)
    if abs(dur - planned) > 0.6:
        log.warning("assembled duration %.2fs differs from the plan %.2fs", dur, planned)
    meta = (script or {}).get("meta", {})
    try:
        res["thumbnail"] = str(make_thumbnail(video, shots, meta.get("title_ar", ""), meta.get("part"), out_dir / f"thumb_{job}.jpg"))
    except Exception as e:  # noqa: BLE001
        log.warning("thumbnail failed: %s", e)
        res["thumbnail"] = None
    res["duration"] = round(dur, 2)
    res["credits"] = sorted({v.get("credit") for v in (voices or {}).values() if v.get("credit")})
    (out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    log.info("short assembled: %s (%.1fs)", video, dur)
    return res


# ------------------------------------------------------------------ weekly episode
def _card_png(lines, size, out):
    """Title card: lines = [(text, px, (r,g,b))]. Arabic is shaped once by Raqm (or reshaper+BASIC)."""
    W, H = size
    img = Image.new("RGBA", (W, H), (8, 8, 14, 255))
    d = ImageDraw.Draw(img)
    for i in range(H):
        d.line([0, i, W, i], fill=(int(10 + 22 * i / H), int(8 + 10 * i / H), int(20 + 36 * i / H), 255))
    y = H * 0.30
    for text, px, col in lines:
        if not text:
            y += px * 0.6
            continue
        layer = subs.render_line_png(text, (W, H), px=px, bottom=int(H - y - px * 1.3), color=(*col, 255))
        img.alpha_composite(layer)
        y += px * 1.55
    img.convert("RGB").save(out)


def title_card(lines, out, seconds=3.5, size=None):
    size = size or render_size()
    png = pathlib.Path(out).with_suffix(".png")
    _card_png(lines, size, png)
    _ff(["-loop", "1", "-framerate", cfg.fps, "-i", png, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", seconds,
         "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={seconds - 0.6:.2f}:d=0.6,format=yuv420p", "-c:v", "libx264", "-preset", cfg.x264_preset,
         "-crf", cfg.crf, "-r", cfg.fps, "-c:a", "aac", "-b:a", "192k", "-shortest", out])
    return pathlib.Path(out)


def _stamp(sec):
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}" if sec < 3600 else f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def assemble_weekly(parts, meta, out_path):
    """parts: [{"video","title_ar","summary"}] in order. meta: {"series_title_ar","episode_title_ar","episode"}.
    -> {"video","duration","chapters"} (chapters = YouTube-style text lines)."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="weekly_", dir=str(out_path.parent)))
    try:
        card = title_card([(meta.get("series_title_ar") or "", 56, (230, 200, 120)), ("", 30, (0, 0, 0)),
                           (meta.get("episode_title_ar") or f"الحلقة {meta.get('episode', 1)}", 92, (255, 255, 255)),
                           (f"الحلقة {meta.get('episode', 1)}", 52, (200, 200, 220))], tmp / "card.mp4")
        files = [card] + [pathlib.Path(p["video"]) for p in parts]
        joined = tmp / "joined.mp4"
        try:
            concat_videos(files, joined)
        except AssembleError as e:
            log.warning("stream-copy concat failed (%s) - re-encoding", str(e)[:200])
            concat_videos(files, joined, reencode=True)
        starts, t = [], 0.0
        for f in files:
            starts.append(t)
            t += media_duration(f)
        total = t
        meta_txt = [";FFMETADATA1"]
        chapters = []
        for i, st in enumerate(starts):
            en = starts[i + 1] if i + 1 < len(starts) else total
            title = "المقدمة" if i == 0 else f"الجزء {i}: {parts[i - 1].get('title_ar') or ''}".strip()
            meta_txt += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(st * 1000)}", f"END={int(en * 1000) - 1}", f"title={title}"]
            chapters.append(f"{_stamp(st)} {title}")
        mf = tmp / "chapters.txt"
        mf.write_text("\n".join(meta_txt) + "\n", "utf-8")
        _ff(["-i", joined, "-i", mf, "-map", "0", "-map_metadata", "1", "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart", out_path])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dur = media_duration(out_path)
    log.info("weekly episode assembled: %s (%.1f min)", out_path, dur / 60)
    return {"video": str(out_path), "duration": round(dur, 1), "chapters": chapters}
