"""Post-production on the rendered drawings with ffmpeg (every step is a plain, widely available filter).

    raw drawings (PNG sequence or video) --time_map--> numbered frames --filter_complex--> graded shot (mp4, exact length)

Steps: grade (eq/colorbalance/curves) -> bloom -> anamorphic streaks -> chroma split -> flash / impact frame / glitch ->
overlays (key / screen) -> grain -> vignette -> letterbox -> transitions. Time remapping (hit-stop, speed ramps) happens in
python (`time_map`) so ffmpeg only ever sees a plain numbered sequence. `finish_shot` writes `<final>.json` with the
per-feature report and degrades step by step (never silently: every dropped step is logged and reported).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from .camera_paths import PLATE_MARGIN
from .config import ASSETS, cfg, log
from .schema import GRADES

# ------------------------------------------------------------------ presets
# eq: contrast/saturation/brightness/gamma, colorbalance (shadows/mids/highs RGB shifts), curve, bloom/grain/vignette/chroma
_BASE = dict(contrast=1.0, saturation=1.0, brightness=0.0, gamma=1.0, shadows=(0, 0, 0), mids=(0, 0, 0), highs=(0, 0, 0),
             curve="none", bloom=0.0, bloom_knee=0.60, grain=0.0, vignette=0.0, chroma=0.0, anamorphic=0.0,
             anamorphic_color=(0.55, 0.78, 1.0), letterbox=0.0, sharpen=0.0)

GRADE_PRESETS = {
    "calm":   dict(contrast=1.05, saturation=1.08, shadows=(0.00, 0.01, 0.03), highs=(0.02, 0.01, 0.00), curve="soft",
                   bloom=0.30, grain=4, vignette=0.22, chroma=0.4, sharpen=0.25),
    "warm":   dict(contrast=1.07, saturation=1.16, shadows=(0.02, 0.00, -0.02), highs=(0.06, 0.02, -0.04), curve="soft",
                   bloom=0.42, grain=4, vignette=0.22, chroma=0.4, sharpen=0.25),
    "tense":  dict(contrast=1.18, saturation=0.92, brightness=-0.02, shadows=(-0.02, 0.00, 0.05), highs=(0.00, 0.01, 0.02),
                   curve="contrast", bloom=0.26, grain=7, vignette=0.45, chroma=0.7, sharpen=0.3),
    "dark":   dict(contrast=1.22, saturation=0.86, brightness=-0.05, shadows=(-0.02, 0.02, 0.07), highs=(0.00, 0.00, 0.02),
                   curve="contrast", bloom=0.28, grain=8, vignette=0.55, chroma=0.6, sharpen=0.3),
    "battle": dict(contrast=1.28, saturation=1.22, shadows=(0.00, -0.01, 0.03), highs=(0.05, 0.00, -0.03),
                   curve="contrast", bloom=0.52, grain=5, vignette=0.35, chroma=1.3, anamorphic=0.35, sharpen=0.4),
    "sad":    dict(contrast=1.05, saturation=0.70, brightness=-0.02, shadows=(0.00, 0.02, 0.06), highs=(0.01, 0.01, 0.02),
                   curve="soft", bloom=0.32, grain=6, vignette=0.38, chroma=0.3, sharpen=0.2),
    "epic":   dict(contrast=1.22, saturation=1.25, shadows=(-0.01, 0.02, 0.06), highs=(0.06, 0.03, -0.03),
                   curve="contrast", bloom=0.60, grain=4, vignette=0.32, chroma=0.9, anamorphic=0.5, sharpen=0.4),
    "mystic": dict(contrast=1.12, saturation=1.20, shadows=(0.03, -0.01, 0.07), highs=(-0.02, 0.04, 0.05), curve="soft",
                   bloom=0.60, grain=4, vignette=0.34, chroma=0.7, anamorphic=0.25, anamorphic_color=(0.7, 0.5, 1.0),
                   sharpen=0.25),
    "dawn":   dict(contrast=1.08, saturation=1.12, shadows=(0.00, 0.01, 0.04), highs=(0.07, 0.03, -0.02), curve="soft",
                   bloom=0.50, grain=4, vignette=0.24, chroma=0.4, anamorphic=0.2, anamorphic_color=(1.0, 0.8, 0.5),
                   sharpen=0.25),
    "night":  dict(contrast=1.18, saturation=0.95, brightness=-0.04, shadows=(-0.03, 0.00, 0.08), highs=(-0.02, 0.00, 0.04),
                   curve="contrast", bloom=0.44, grain=6, vignette=0.46, chroma=0.6, sharpen=0.3),
}
for _g in GRADES:                                         # guarantee a preset for every grade in the shared vocabulary
    GRADE_PRESETS.setdefault(_g, {})
CURVES = {"none": None, "soft": "0/0 0.25/0.22 0.5/0.5 0.75/0.79 1/1", "contrast": "0/0 0.22/0.17 0.5/0.5 0.78/0.84 1/1"}
OVERRIDABLE = ("bloom", "grain", "vignette", "chroma", "contrast", "saturation", "letterbox", "anamorphic", "sharpen")


class PostError(RuntimeError):
    pass


def grade_params(preset, overrides=None):
    p = dict(_BASE)
    p.update(GRADE_PRESETS.get(preset or "calm", GRADE_PRESETS["calm"]))
    for k in OVERRIDABLE:
        v = (overrides or {}).get(k)
        if v is not None:
            p[k] = v
    p["preset"] = preset or "calm"
    return p


# ------------------------------------------------------------------ time remapping (hit-stop / speed ramps)
def time_map(shot, drawings=None):
    """-> list (length = shot["frames"]) of 0-based drawing indices to show on every output frame."""
    fps, step, frames = shot["fps"], max(1, shot.get("step", 1)), shot["frames"]
    drawings = drawings or max(1, math.ceil(frames / step))
    segs = []
    for fx in shot.get("post_fx") or []:
        k, at = fx.get("kind"), float(fx.get("at") or 0.0)
        d = fx.get("dur")
        if k == "hit_stop":
            segs.append((at, at + float(d or 0.12), 0.0, 0.0))
        elif k == "speed_ramp":
            segs.append((at, at + float(d or 0.8), max(0.05, float(fx.get("speed") or 0.35)), 0.2))

    def speed(t):
        s = 1.0
        for a, b, target, ease in segs:
            if a <= t < b:
                e = min(1.0, min(t - a, b - t) / max(1e-6, ease * (b - a))) if ease else 1.0
                s = 1.0 + (target - 1.0) * e
        return s

    out, src = [], 0.0
    for n in range(frames):
        out.append(min(drawings - 1, int(src * fps / step + 1e-6)))
        src += speed(n / fps) / fps
    return out


# ------------------------------------------------------------------ filter graph
def _en(a, b):
    return f"enable='between(t,{a:.3f},{b:.3f})'"


def _fx_list(shot, kind):
    return [f for f in (shot.get("post_fx") or []) if f.get("kind") == kind]


def overlay_files(shot):
    """Resolved overlay clips in input order (inputs 1..n of the ffmpeg command)."""
    out = []
    for fx in _fx_list(shot, "overlay"):
        clip = str(fx.get("clip") or "")
        for cand in (pathlib.Path(clip), ASSETS / "overlays" / clip):
            if clip and cand.is_file():
                out.append((cand, fx))
                break
        else:
            log.warning("overlay clip %r not found (assets/overlays) - skipped", clip)
    return out


def vertical_crop_filter(shot):
    """9:16 window from the 16:9 frame, centred on the subject as the framing places it."""
    w, h = shot.get("size", (1920, 1080))
    zoom = max(1.0, float((shot.get("framing") or {}).get("zoom", 1.0)))
    cx = float((shot.get("framing") or {}).get("cx", 0.5))
    sx = float(((shot.get("plate") or {}).get("subject") or {}).get("x", cx))
    u = 0.5 + (sx - cx) / (PLATE_MARGIN / zoom)
    u = min(0.72, max(0.28, u))
    cw = int(h * 9 / 16) // 2 * 2
    return (f"crop={cw}:{h}:x='max(0,min({w}-{cw},{u:.4f}*{w}-{cw}/2))':y=0,"
            f"scale=1080:1920:flags=lanczos,unsharp=5:5:0.5:5:5:0,setsar=1")


def build_filtergraph(shot, in_fps=None, *, level=3, vertical=False, src_size=None):
    """Pure function: full `-filter_complex` string. Input `[0:v]` (numbered sequence), outputs `[v]` (+ `[vv]` if vertical).

    level 3 = everything, 2 = no overlay/anamorphic/glitch/bloom, 1 = grade + fades only, 0 = plain format conversion.
    Returns the graph string; `report_for(shot, level)` says which features it contains.
    """
    ow, oh = shot.get("size", (1920, 1080))          # the size the rest of the pipeline expects
    w, h = src_size or (ow, oh)                      # the size of the frames we are really given (drafts, a stale cache ...)
    fps = shot.get("fps", cfg.fps)
    g = grade_params((shot.get("grade") or {}).get("preset"), shot.get("grade") or {})
    segs, cur, n = [], "0:v", [0]

    def lab():
        n[0] += 1
        return f"s{n[0]}"

    def add(filters, *, extra_in=None, out=None):
        nonlocal cur
        o = out or lab()
        ins = f"[{cur}]" + "".join(f"[{e}]" for e in (extra_in or []))
        segs.append(f"{ins}{filters}[{o}]")
        cur = o

    def mix(side_filters, blend, opacity, *, enable=None):
        """split current -> side chain -> blend back on the untouched branch."""
        nonlocal cur
        a, b, s, o = lab(), lab(), lab(), lab()
        segs.append(f"[{cur}]split[{a}][{b}]")
        segs.append(f"[{b}]{side_filters}[{s}]")
        en = f":{enable}" if enable else ""
        segs.append(f"[{a}][{s}]blend=all_mode={blend}:all_opacity={opacity:.3f}{en}[{o}]")
        cur = o

    qw, qh = f"trunc(iw/8)*2", f"trunc(ih/8)*2"           # quarter-size, even
    add("format=gbrp,setsar=1")
    # ---- grade
    eq = f"eq=contrast={g['contrast']:.3f}:saturation={g['saturation']:.3f}:brightness={g['brightness']:.3f}:gamma={g['gamma']:.3f}"
    add(eq)
    s, m, hi = g["shadows"], g["mids"], g["highs"]
    if any(abs(x) > 1e-6 for x in (*s, *m, *hi)):
        add(f"colorbalance=rs={s[0]:.3f}:gs={s[1]:.3f}:bs={s[2]:.3f}:rm={m[0]:.3f}:gm={m[1]:.3f}:bm={m[2]:.3f}"
            f":rh={hi[0]:.3f}:gh={hi[1]:.3f}:bh={hi[2]:.3f}")
    if CURVES.get(g["curve"]):
        add(f"curves=all='{CURVES[g['curve']]}'")
    if g["sharpen"] > 0:
        add(f"unsharp=5:5:{min(1.2, g['sharpen']):.2f}:5:5:0")
    if level >= 3:
        # ---- bloom: bright pass -> big blur at 1/4 size -> screen
        if g["bloom"] > 0.01:
            k = g["bloom_knee"]
            sig = max(2.0, h * 0.012 / 4 * 4)
            mix(f"curves=all='0/0 {k:.2f}/0 {min(0.99, k + 0.2):.2f}/0.25 1/1',scale={qw}:{qh}:flags=bicubic,"
                f"gblur=sigma={sig / 4:.2f},scale={w}:{h}:flags=bicubic",
                "screen", min(0.9, g["bloom"] * 0.75))
        # ---- anamorphic streaks (horizontal flares off the highlights)
        amt = g["anamorphic"]
        for fx in _fx_list(shot, "anamorphic"):
            amt = max(amt, float(fx.get("amount") or 0.5))
        if amt > 0.01:
            col = g["anamorphic_color"]
            for fx in _fx_list(shot, "anamorphic"):
                if isinstance(fx.get("color"), (list, tuple)) and len(fx["color"]) == 3:
                    col = fx["color"]
            mix(f"curves=all='0/0 0.72/0 0.9/0.45 1/1',scale={qw}:{qh}:flags=bicubic,"
                f"gblur=sigma={w / 14 / 4:.2f}:sigmaV=0.8,colorchannelmixer=rr={col[0]:.2f}:gg={col[1]:.2f}:bb={col[2]:.2f},"
                f"scale={w}:{h}:flags=bicubic", "screen", min(0.9, amt * 0.9))
    # ---- chromatic split (static fringe + stronger on `chromatic` windows)
    px = g["chroma"] * h / 1080.0
    if px > 0.25:
        add(f"rgbashift=rh={-px:.2f}:bh={px:.2f}:rv={px * 0.15:.2f}:bv={-px * 0.15:.2f}")
    for fx in _fx_list(shot, "chromatic"):
        at, d = float(fx.get("at") or 0), float(fx.get("dur") or 0.25)
        a = float(fx.get("amount") or 0.6) * 14 * h / 1080.0
        add(f"rgbashift=rh={-a:.1f}:bh={a:.1f}:gv={a * 0.3:.1f}:{_en(at, at + d)}")
    # ---- flashes / impact frames / glitch
    for fx in _fx_list(shot, "flash"):
        at, d = float(fx.get("at") or 0), max(0.05, float(fx.get("dur") or 0.2))
        add(f"eq=brightness='if(between(t,{at:.3f},{at + d:.3f}),0.95*(1-(t-{at:.3f})/{d:.3f}),0)':"
            f"saturation='if(between(t,{at:.3f},{at + d:.3f}),1-0.8*(1-(t-{at:.3f})/{d:.3f}),1)':eval=frame")
    for fx in _fx_list(shot, "impact_frame"):
        at, d = float(fx.get("at") or 0), max(1.5 / fps, float(fx.get("dur") or 0.08))
        e = _en(at, at + d)
        add(f"hue=s=0:{e},eq=contrast=3.2:brightness=0.06:{e},negate={e}")
    if level >= 3:
        for fx in _fx_list(shot, "glitch"):
            at, d = float(fx.get("at") or 0), max(2.0 / fps, float(fx.get("dur") or 0.18))
            e = _en(at, at + d)
            add(f"rgbashift=rh=-16:bh=18:gv=6:gh=-4:{e},noise=alls=55:allf=t:{e},hue=s=1.6:{e}")
        # ---- stock overlays (sparks, light leaks, smoke...) : inputs 1..n
        for i, (clip, fx) in enumerate(overlay_files(shot), start=1):
            at, d = float(fx.get("at") or 0), fx.get("dur")
            en = _en(at, at + float(d)) if d else f"enable='gte(t,{at:.3f})'"
            ov, mode = lab(), (fx.get("mode") or "screen")
            op = min(1.0, max(0.05, float(fx.get("opacity") or 1.0)))
            pre = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=gbrp"
            if mode == "key":
                segs.append(f"[{i}:v]{pre}[{ov}k];[{ov}k]colorkey=0x000000:0.28:0.12,format=rgba[{ov}]")
                o = lab()
                segs.append(f"[{cur}][{ov}]overlay=format=auto:{en}[{o}]")
                cur = o
            else:
                segs.append(f"[{i}:v]{pre}[{ov}]")
                o = lab()
                segs.append(f"[{cur}][{ov}]blend=all_mode=screen:all_opacity={op:.2f}:{en}[{o}]")
                cur = o
    # ---- grain, vignette, letterbox
    if g["grain"] > 0.2:
        add(f"noise=alls={g['grain']:.1f}:allf=t+u")
    if g["vignette"] > 0.01:
        add(f"vignette=angle={0.16 + 0.52 * min(1.0, g['vignette']):.3f}:mode=forward:eval=init")
    if g["letterbox"] > 0.005:
        bar = int(h * g["letterbox"]) // 2 * 2
        add(f"drawbox=x=0:y=0:w=iw:h={bar}:color=black@1:t=fill,drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black@1:t=fill")
    # ---- transitions
    dur = shot["frames"] / fps
    tin, tout = shot.get("trans_in") or "cut", shot.get("trans_out") or "cut"
    td = min(0.35, dur / 4)
    if tin in ("fade", "dip_black"):
        add(f"fade=t=in:st=0:d={td:.3f}:color=black")
    elif tin == "dip_white":
        add(f"fade=t=in:st=0:d={td:.3f}:color=white")
    elif tin == "flash":
        add("eq=brightness='if(lt(t,0.14),0.9*(1-t/0.14),0)':saturation='if(lt(t,0.14),0.3+0.7*t/0.14,1)':eval=frame")
    elif tin == "whip":
        add(f"gblur=sigma={w / 36:.1f}:sigmaV=0.6:enable='between(t,0,0.14)'")
    if tout in ("fade", "dip_black"):
        add(f"fade=t=out:st={max(0.0, dur - td):.3f}:d={td:.3f}:color=black")
    elif tout == "dip_white":
        add(f"fade=t=out:st={max(0.0, dur - td):.3f}:d={td:.3f}:color=white")
    elif tout == "whip":
        add(f"gblur=sigma={w / 36:.1f}:sigmaV=0.6:enable='between(t,{max(0.0, dur - 0.14):.3f},{dur:.3f})'")
    if (w, h) != (ow, oh):
        add(f"scale={ow}:{oh}:flags=lanczos")
    add("format=yuv420p", out="v16" if vertical else "v")
    if vertical:
        segs.append(f"[v16]split[v][vsrc];[vsrc]{vertical_crop_filter(shot)}[vv]")
    return ";".join(segs)


def report_for(shot, level=3):
    """Which post features the graph at `level` contains (what we may truthfully call 'ok' after a successful run)."""
    g = grade_params((shot.get("grade") or {}).get("preset"), shot.get("grade") or {})
    kinds = {f.get("kind") for f in shot.get("post_fx") or []}
    r = {"post_grade": "ok", "post_curves": "ok" if CURVES.get(g["curve"]) else "off"}
    r["post_bloom"] = "ok" if level >= 3 and g["bloom"] > 0.01 else "off"
    r["post_anamorphic"] = "ok" if level >= 3 and (g["anamorphic"] > 0.01 or "anamorphic" in kinds) else "off"
    r["post_chroma"] = "ok" if g["chroma"] * shot.get("size", (1920, 1080))[1] / 1080 > 0.25 or "chromatic" in kinds else "off"
    r["post_grain"] = "ok" if g["grain"] > 0.2 else "off"
    r["post_vignette"] = "ok" if g["vignette"] > 0.01 else "off"
    r["post_flash"] = "ok" if "flash" in kinds else "off"
    r["post_impact_frame"] = "ok" if "impact_frame" in kinds else "off"
    r["post_glitch"] = ("ok" if level >= 3 else "skipped:degraded") if "glitch" in kinds else "off"
    r["post_overlay_key"] = (("ok" if level >= 3 else "skipped:degraded") if overlay_files(shot) else "off")
    r["post_letterbox"] = "ok" if g["letterbox"] > 0.005 else "off"
    r["post_transitions"] = "ok" if (shot.get("trans_in") not in (None, "cut") or shot.get("trans_out") not in (None, "cut")) else "off"
    r["post_hit_stop"] = "ok" if "hit_stop" in kinds else "off"
    r["post_time_remap"] = "ok" if ({"speed_ramp", "hit_stop"} & kinds) else "off"
    return r


# ------------------------------------------------------------------ running ffmpeg
def _ffmpeg(args, timeout=1800):
    cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise PostError((r.stderr or "").strip()[-1200:] or f"ffmpeg exited {r.returncode}")
    return r


def _frame_files(raw, tmp):
    """-> list of image paths (one per drawing). Accepts a directory of PNGs, or decodes a video."""
    raw = pathlib.Path(raw)
    if raw.is_dir():
        files = sorted(p for p in raw.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".exr", ".bmp", ".tif", ".tiff")
                       and not p.name.startswith("."))
        if not files:
            raise PostError(f"no frames in {raw}")
        return files
    if raw.is_file():
        d = pathlib.Path(tmp) / "decoded"
        d.mkdir(parents=True, exist_ok=True)
        _ffmpeg(["-i", raw, "-vsync", "0", d / "d_%05d.png"])
        files = sorted(d.glob("d_*.png"))
        if not files:
            raise PostError(f"could not decode any frame from {raw}")
        return files
    raise PostError(f"raw render not found: {raw}")


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001      (EXR and friends: trust the plan)
        return None


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copyfile(src, dst)


def finish_shot(shot, raw, in_fps=None, *, vertical=None):
    """Grade + time-map `raw` into `shot["paths"]["final"]` (mp4, exactly `shot["frames"]` frames at `shot["fps"]`).

    Returns the final path. Also writes `<final>.json` (feature report + degradation info) and, when `vertical`
    (default cfg.vertical), `<final stem>_v.mp4` (1080x1920 crop centred on the subject).
    """
    vertical = cfg.vertical if vertical is None else vertical
    final = pathlib.Path(shot["paths"]["final"])
    final.parent.mkdir(parents=True, exist_ok=True)
    vfinal = final.with_name(final.stem + "_v.mp4")
    fps, frames = int(shot.get("fps", cfg.fps)), int(shot["frames"])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"post_{final.stem}_", dir=str(final.parent)))
    degraded, report = [], {}
    try:
        files = _frame_files(raw, tmp)
        src_size = _image_size(files[0])
        if src_size and tuple(src_size) != tuple(shot.get("size", src_size)):
            log.warning("post: shot %s frames are %sx%s but the plan says %sx%s - scaling to the plan", shot.get("idx"), *src_size,
                        *shot["size"])
        if not pathlib.Path(raw).is_dir():                       # decoded video: its frames are already at video fps
            shot = {**shot, "step": 1}
            if in_fps and in_fps != fps:
                log.info("raw video at %s fps will be mapped 1:1 onto %s fps output", in_fps, fps)
        seq = tmp / "seq"
        seq.mkdir()
        tmap = time_map(shot, drawings=len(files))
        for i, di in enumerate(tmap, start=1):
            _link(files[di], seq / f"o_{i:05d}.png")
        ovs = [p for p, _ in overlay_files(shot)]
        last_err = None
        for level in (3, 2, 1, 0):
            if level < 3 and not last_err:
                break
            try:
                graph = build_filtergraph(shot, in_fps, level=level, vertical=bool(vertical), src_size=src_size)
                args = ["-framerate", fps, "-i", seq / "o_%05d.png"]
                if level >= 3:
                    for p in ovs:
                        args += ["-stream_loop", "-1", "-i", p]
                enc = ["-c:v", "libx264", "-preset", cfg.x264_preset, "-crf", max(12, cfg.crf - 2), "-pix_fmt", "yuv420p",
                       "-r", fps, "-an", "-movflags", "+faststart"]
                maps = ["-map", "[v]", "-frames:v", frames, *enc, final]
                if vertical:
                    maps += ["-map", "[vv]", "-frames:v", frames, *enc, vfinal]
                _ffmpeg([*args, "-filter_complex", graph, *maps])
                report = report_for(shot, level)
                if level < 3:
                    degraded.append(f"level {level}: {last_err}")
                    for k, v in report_for(shot, 3).items():
                        if v == "ok" and report.get(k) != "ok":
                            report[k] = f"skipped:degraded ({str(last_err)[:80]})"
                break
            except PostError as e:
                last_err = str(e).splitlines()[-1] if str(e) else "ffmpeg failed"
                log.error("post: ffmpeg failed at level %d for shot %s: %s", level, shot.get("idx"), last_err)
        else:
            raise PostError(f"post failed at every level for shot {shot.get('idx')}: {last_err}")
        report["time_remap_len"] = len(tmap)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    final.with_suffix(".json").write_text(json.dumps({"features": report, "degraded": degraded,
                                                      "vertical": bool(vertical)}, indent=1), "utf-8")
    return final


def count_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    m = re.search(r"\d+", r.stdout or "")
    return int(m.group()) if m else -1
