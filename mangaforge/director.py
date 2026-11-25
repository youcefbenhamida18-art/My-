"""The director: script + voices + art  ->  render-ready shot list (SPEC section 3).

This is where "slideshow" becomes "anime": real voice timing, cut-in framings from one image, camera language with variety,
hit points (flash / impact frame / hit-stop / shake), atmosphere on every shot, scene-consistent grading, transitions, sfx
placement and 3D dressing. Everything is deterministic for a given job id (seeded), so a re-run produces the same film.
"""
from __future__ import annotations

import json
import random
import re

from . import camera_paths as cp
from . import eye_anim, schema, titles
from . import eyes as eyes_mod
from .config import cfg, log, render_size, slug, work_dir

LEAD, TAIL = 0.25, 0.50                # silence before / after a spoken line inside its shot
MIN_DUR = 0.9
OUTDOOR = re.compile(r"grass|field|meadow|forest|garden|path|road|ruin|mountain|cliff|river|village|battlefield|ground|"
                     r"shrine|bamboo|hill|beach|stone|rock|temple|courtyard|bridge|canyon|valley|wasteland", re.I)
GRASSY = re.compile(r"grass|field|meadow|forest|garden|bamboo|hill|valley", re.I)
IMPACT_FX = {"shockwave", "slash", "lightning", "sparks", "debris", "glass_shards"}
FOCAL = {"establishing": 24, "wide": 28, "transition": 30, "action": 32, "impact": 30, "medium": 40, "insert": 55,
         "reaction": 58, "closeup": 70, "eyes": 85}
EYE_EMOTIONS = ("shout", "angry", "fear", "sad", "tender", "cold")          # lines that deserve an eye close-up on their own
EYE_GAP = 11.0                                                              # seconds between automatic eye close-ups
STRENGTH = {"establishing": 0.55, "wide": 0.5, "medium": 0.55, "closeup": 0.6, "reaction": 0.5, "insert": 0.55,
            "action": 0.85, "impact": 1.0, "transition": 0.8}
GRADE_MUSIC = {g: g for g in schema.MUSIC if g in schema.GRADES}
GRADE_MUSIC.update({"dawn": "calm", "night": "mystic"})


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ timing
def _timing(script, voices, target):
    durs, needs = [], []
    for s in script["shots"]:
        v = voices.get(s["id"])
        need = (LEAD + v["dur"] + TAIL) if v else 0.0
        needs.append(need)
        durs.append(max(s["dur"], need, MIN_DUR))
    total = sum(durs)
    if total > target * 1.04:                                       # shrink what can be shrunk (never below a spoken line)
        slack = [max(0.0, d - max(n, MIN_DUR)) for d, n in zip(durs, needs)]
        k = min(1.0, (total - target) / max(1e-6, sum(slack)))
        durs = [d - sl * k for d, sl in zip(durs, slack)]
        if sum(durs) > target * 1.08:
            log.warning("director: %.1fs of spoken lines + minimum shots exceed the %.0fs target (%.1fs). Shorten the dialogue.",
                        sum(durs), target, sum(durs))
    elif total < target * 0.93:                                     # breathe: lengthen establishing / reaction holds
        grow = [i for i, s in enumerate(script["shots"]) if not s.get("line") and s["type"] in
                ("establishing", "wide", "reaction", "insert", "closeup", "medium")]
        room = [durs[i] * 0.6 for i in grow]
        k = min(1.0, (target - total) / max(1e-6, sum(room)))
        for i, r in zip(grow, room):
            durs[i] += r * k
    step = cfg.anim_step
    out = []
    for d in durs:
        frames = max(step * 2, int(round(d * cfg.fps / step)) * step)
        out.append(frames)
    return out


# ------------------------------------------------------------------ framing
def _auto_framing(stype, img_framing, subject, k):
    sx, sy = subject["x"], subject["y"]
    if stype in ("closeup", "reaction"):
        zoom = 1.18 if img_framing == "closeup" else 1.85 + 0.25 * (k % 3)
        cx, cy = sx, sy
    elif stype == "medium":
        zoom = 1.0 if img_framing in ("medium", "closeup") else 1.4 + 0.1 * (k % 2)
        cx, cy = sx, sy + 0.12
    elif stype == "insert":
        zoom = 2.1
        cx, cy = sx + (0.04 if k % 2 else -0.04), sy + 0.14
    elif stype in ("action", "impact"):
        zoom = 1.2 + 0.25 * (k % 3) if img_framing != "closeup" else 1.1
        cx, cy = sx, sy + 0.08
    else:
        return {"zoom": 1.0, "cx": 0.5, "cy": 0.5}
    zoom = _clamp(zoom, 1.0, 2.2)
    return {"zoom": round(zoom, 3), "cx": cx, "cy": cy}


def _clamp_window(fr):
    hs = cp.PLATE_MARGIN / max(1.0, fr["zoom"]) / 2
    return {"zoom": round(fr["zoom"], 3), "cx": round(_clamp(fr["cx"], hs, 1 - hs), 4), "cy": round(_clamp(fr["cy"], hs, 1 - hs), 4)}


def _is_default(fr):
    return abs(fr["zoom"] - 1.0) < 1e-6 and abs(fr["cx"] - 0.5) < 1e-6 and abs(fr["cy"] - 0.5) < 1e-6


# ------------------------------------------------------------------ title cards (pilot sting / episode premiere)
def _title_cards(script, frames_list, meta, pilot):
    """{shot index: "series"|"episode"} from the script, completed so the pilot always has its title sting."""
    tc = {i: s["title_card"] for i, s in enumerate(script["shots"]) if s.get("title_card")}
    n = len(script["shots"])
    if pilot and "series" not in tc.values():
        t, cand = 0.0, None
        for i, (s, fr) in enumerate(zip(script["shots"], frames_list)):
            if t >= 8.0 and s["type"] in ("establishing", "wide", "transition"):
                cand = i
                break
            t += fr / cfg.fps
        tc[cand if cand is not None else min(3, n - 1)] = "series"
    elif not pilot and not tc and meta.get("part") == 1:
        tc[0 if script["shots"][0]["type"] in ("establishing", "wide") else min(1, n - 1)] = "episode"
    return tc


def _eye_info(a, cache):
    p = a.get("eyes")
    if not cfg.eyes or not p:
        return None
    if p not in cache:
        cache[p] = eyes_mod.load(p)
    return cache[p]


# ------------------------------------------------------------------ main
def plan(script, voices, art, job, *, sequences=None, target_seconds=None, write=True):
    """-> list of render-ready shot dicts (also written to work/<job>/shots.json)."""
    sequences = sequences or {}
    target = target_seconds or cfg.short_seconds
    W = work_dir(job)
    rng = random.Random(sum(ord(c) * (i + 1) for i, c in enumerate(str(job))) + 7)
    frames_list = _timing(script, voices, target)
    images = {im["id"]: im for im in script["images"]}
    size = list(render_size())
    shots, last_moves = [], []
    prev_scene, prev_grade, prev_music = None, "calm", "calm"
    t_clock, last_impact_t, last_stop_t, k_cut = 0.0, -99.0, -99.0, 0
    meta = script.get("meta") or {}
    pilot = bool(meta.get("pilot"))
    title_at = _title_cards(script, frames_list, meta, pilot)
    eye_cache, last_eye_t, n_eye_cuts = {}, -99.0, 0
    for idx, (s, frames) in enumerate(zip(script["shots"], frames_list)):
        im = images[s["image"]]
        a = art.get(im["id"])
        if not a:
            raise KeyError(f"no art for image {im['id']!r} (shot {s['id']}): run the art stage first")
        dur = frames / cfg.fps
        stype = s["type"]
        line = s.get("line")
        v = voices.get(s["id"])
        inten = line["intensity"] if line else {"impact": 5, "action": 4, "closeup": 3}.get(stype, 2)
        new_scene = im["scene"] != prev_scene
        # ---- grade / music (scene-consistent; action overrides)
        grade = s.get("grade") or (prev_grade if not new_scene else None)
        if stype in ("action", "impact") and not s.get("grade"):
            grade = "battle"
        grade = grade or (s.get("music") if s.get("music") in schema.GRADES else None) or prev_grade
        music = s.get("music") or (prev_music if not new_scene else GRADE_MUSIC.get(grade, "calm"))
        if stype in ("action", "impact") and not s.get("music"):
            music = "battle"
        # ---- hit point
        fxs = list(s.get("fx") or [])
        hit_at = None
        if stype == "impact":
            hit_at = round(max(0.25, dur * 0.3), 3)
        elif stype == "action" and (inten >= 4 or IMPACT_FX & set(fxs)):
            hit_at = round(max(0.3, dur * 0.42), 3)
        elif IMPACT_FX & set(fxs):
            hit_at = round(max(0.3, dur * 0.4), 3)
        # ---- atmosphere defaults
        blender_fx = [f for f in fxs if f in schema.FX_BLENDER]
        post_kinds = [f for f in fxs if f in schema.FX_POST]
        for d in schema.DEFAULT_FX.get(grade, [])[:2]:
            if len(blender_fx) < 2 and d not in blender_fx:
                blender_fx.append(d)
        person = bool(im["characters"]) and not im["no_humans"]
        if person and stype != "establishing":
            for d in ("hair_sway", "wind_sway"):
                if d not in blender_fx and len(blender_fx) < 5:
                    blender_fx.append(d)
        elif "wind_sway" not in blender_fx:
            blender_fx.append("wind_sway")
        if stype == "impact":
            for d in ("shockwave", "sparks", "speed_lines"):
                if d not in blender_fx and len(blender_fx) < 6:
                    blender_fx.append(d)
            if "flash" not in post_kinds:
                post_kinds.append("flash")
        if stype == "action" and "speed_lines" not in blender_fx and inten >= 4:
            blender_fx.append("speed_lines")
        amount = round(_clamp(0.35 + 0.1 * inten, 0.3, 0.95), 2)
        fx_list = []
        for k in blender_fx:
            at = hit_at if (k in IMPACT_FX and hit_at is not None) else 0.0
            fx_list.append({"kind": k, "amount": amount, "at": at, "dur": None if k not in IMPACT_FX else round(min(dur, 1.4), 2)})
        # ---- post fx (explicit + auto, budgeted so hit-stops stay special)
        post_fx = []
        for k in post_kinds:
            if k == "flash":
                post_fx.append({"kind": "flash", "at": hit_at or 0.0, "dur": 0.16})
            elif k == "impact_frame":
                post_fx.append({"kind": "impact_frame", "at": hit_at or 0.0, "dur": 0.083})
            elif k == "hit_stop":
                post_fx.append({"kind": "hit_stop", "at": (hit_at or dur * 0.4) + 0.05, "dur": 0.14})
            elif k == "speed_ramp":
                post_fx.append({"kind": "speed_ramp", "at": dur * 0.25, "dur": min(1.2, dur * 0.5), "speed": 0.35})
            elif k == "chromatic":
                post_fx.append({"kind": "chromatic", "at": hit_at or 0.0, "dur": 0.3, "amount": 0.7})
            elif k == "glitch":
                post_fx.append({"kind": "glitch", "at": hit_at or 0.0, "dur": 0.2})
            elif k == "anamorphic":
                post_fx.append({"kind": "anamorphic", "amount": 0.6})
        have = {p["kind"] for p in post_fx}
        t_abs = t_clock + (hit_at or 0)
        if hit_at is not None and stype == "impact":
            if "impact_frame" not in have and t_abs - last_impact_t > 11:
                post_fx.append({"kind": "impact_frame", "at": round(hit_at + 0.04, 3), "dur": 0.083})
                last_impact_t = t_abs
            if "hit_stop" not in have and t_abs - last_stop_t > 9 and inten >= 4:
                post_fx.append({"kind": "hit_stop", "at": round(hit_at + 0.1, 3), "dur": 0.14})
                last_stop_t = t_abs
            if "chromatic" not in have:
                post_fx.append({"kind": "chromatic", "at": hit_at, "dur": 0.3, "amount": 0.7})
        # ---- eyes: automatic close-ups on emotional lines + the script's hints (closeup / glow / tears)
        einfo = _eye_info(a, eye_cache) if person else None
        has_eyes = bool(einfo and einfo.get("eyes"))
        emode = s.get("eyes")
        if not emode and has_eyes and stype in ("closeup", "reaction") and dur >= 1.0 and line and line["emotion"] in EYE_EMOTIONS \
                and t_clock - last_eye_t > EYE_GAP and n_eye_cuts < 7:
            emode = "closeup"
        if emode and stype in ("wide", "establishing", "transition", "action", "impact") and emode == "closeup":
            emode = None                                                # an eye close-up never replaces a wide / action beat
        # ---- framing (cut-ins from the same image)
        fr = dict(s["framing"])
        if _is_default(fr):
            fr = _auto_framing(stype, im["framing"], a["subject"], k_cut)
            k_cut += 1
        eye_cut = False
        if emode and has_eyes:
            ef = eyes_mod.closeup_framing(einfo, fill=0.36 if emode == "glow" else 0.5, zmax=3.6)
            if ef:
                fr = {"zoom": ef[0], "cx": ef[1], "cy": ef[2]}
                eye_cut = True
                last_eye_t, n_eye_cuts = t_clock, n_eye_cuts + 1
        fr = _clamp_window(fr)
        # ---- camera
        mv = s["camera"]["move"] or cp.default_move_for(stype, inten, idx)
        if len(last_moves) >= 2 and last_moves[-1] == last_moves[-2] == mv:
            mv = cp.default_move_for(stype, inten, idx + 1)
            if mv == last_moves[-1]:
                mv = "drift"
        last_moves.append(mv)
        strength = s["camera"]["strength"] if s["camera"]["move"] else STRENGTH.get(stype, 0.55)
        if dur < 1.3:
            strength = min(1.0, strength * 0.8)
        subj_depth = a["subject"]["depth"]
        rack = None
        if stype in ("closeup", "reaction") and dur >= 2.4 and rng.random() < 0.45:
            rack = {"from": round(_clamp(subj_depth - 0.4, 0.0, 1.0), 3), "to": round(subj_depth, 3)}
        shake = 0.9 if stype == "impact" else (0.35 if stype == "action" else 0.0)
        focal = FOCAL.get(stype, 35)
        if eye_cut:                                                     # slow portrait-lens push towards the eyes, focus pulled lash -> iris
            mv, strength, focal = "push_in", 0.4, FOCAL["eyes"]
            if dur >= 1.4:
                rack = {"from": round(_clamp(subj_depth - 0.22, 0.0, 1.0), 3), "to": round(subj_depth, 3)}
            last_moves[-1] = mv
        camera = {"move": mv, "strength": round(strength, 3), "dof": eye_cut or stype in ("closeup", "reaction", "medium", "insert") or
                  (stype == "action" and inten < 5), "shake": shake, "hit_at": hit_at, "focal": focal, "rack": rack}
        # ---- transitions
        tin = s.get("trans_in")
        if not tin:
            if idx == 0:
                tin = "fade"
            elif new_scene:
                tin = "dip_black" if grade in ("dark", "sad", "night") else "fade"
            elif stype == "impact":
                tin = "flash"
            elif mv in ("whip_l", "whip_r") and stype in ("action", "transition"):
                tin = "whip"
            else:
                tin = "cut"
        tout = "dip_black" if idx == len(script["shots"]) - 1 else "cut"
        # ---- 3D dressing for ground scenes
        auto = []
        if OUTDOOR.search(im["prompt"] or "") and stype in ("establishing", "wide", "action", "medium"):
            auto.append("pebbles")
            if GRASSY.search(im["prompt"] or ""):
                auto.append("grass")
                if grade in ("warm", "dawn", "calm"):
                    auto.append("leaves")
        # ---- audio
        sfx = []
        for name in s.get("sfx") or []:
            at_ = hit_at if (name in ("hit", "boom", "clash", "slash", "glass") and hit_at is not None) else 0.0
            sfx.append({"kind": name, "t": round(at_, 3)})
        kinds = {e["kind"] for e in sfx}
        if hit_at is not None and stype in ("impact", "action") and "hit" not in kinds:
            sfx.append({"kind": "hit", "t": hit_at})
            if stype == "impact" and "boom" not in kinds:
                sfx.append({"kind": "boom", "t": hit_at})
        if tin == "whip" or mv in ("whip_l", "whip_r"):
            if "whoosh" not in kinds:
                sfx.append({"kind": "whoosh", "t": 0.0})
        if "rain" in blender_fx and new_scene and "rain" not in kinds:
            sfx.append({"kind": "rain", "t": 0.0})
        if new_scene and stype == "establishing" and grade in ("calm", "warm", "dawn") and "wind" not in kinds:
            sfx.append({"kind": "wind", "t": 0.0})
        sfx = sfx[:4]
        vs = round(max(LEAD, min(0.7, (dur - v["dur"]) * 0.3)), 3) if v else 0.0
        audio = {"voice": v["path"] if v else None, "voice_start": vs, "voice_dur": round(v["dur"], 3) if v else 0.0,
                 "sfx": sfx, "music": music, "intensity": inten}
        wsign = -1 if rng.random() < 0.5 else 1
        wind = [round(wsign * (0.55 if "wind_sway" in blender_fx else 0.25) * (1.4 if "rain" in blender_fx else 1.0), 3), 0.05]
        plate = {"image": a["image"], "backdrop": a["backdrop"], "depth": a["depth"], "sway": a.get("sway"), "rim": a.get("rim"),
                 "subject": a["subject"], "palette": a.get("palette"), "ground_y": a.get("ground_y", 0.85),
                 "parallax": 0.35, "backdrop_scale": 1.5, "sequence": sequences.get(s["id"]),
                 "matte_method": (a.get("methods") or {}).get("matte", "rembg")}
        sid_ = f"shot_{idx:04d}"
        seed_ = rng.randrange(1, 2 ** 31)
        if has_eyes:
            plate["eyes"] = {"size": einfo["size"], "mask": einfo.get("mask"), "eyes": einfo["eyes"]}
        step_ = 1 if (stype == "impact" and cfg.anim_step == 2) else cfg.anim_step          # hits are drawn on ones
        shot = {"idx": idx, "id": s["id"], "scene": im["scene"], "type": stype, "dur": round(dur, 4), "frames": frames,
                "fps": cfg.fps, "step": step_, "size": size, "plate": plate, "framing": fr, "camera": camera,
                "fx": fx_list, "post_fx": post_fx,
                "grade": {"preset": grade, "bloom": None, "grain": None, "vignette": None, "chroma": None, "contrast": None,
                          "saturation": None},
                "trans_in": tin, "trans_out": tout, "seed": seed_, "wind": wind, "physics": cfg.physics,
                "auto_dress": auto,
                "text": ({"ar": line["ar"], "ja": line["ja"], "speaker": line["speaker"], "emotion": line["emotion"]} if line else None),
                "audio": audio,
                "paths": {"frames_dir": str(W / "frames" / sid_), "raw": str(W / "raw" / f"{sid_}.mkv"),
                          "final": str(W / "shots" / f"{sid_}.mp4")}}
        if has_eyes:
            emo = (line or {}).get("emotion") or ("cold" if grade in ("dark", "night", "tense") else "neutral")
            glow_at = hit_at if hit_at is not None else round(dur * 0.25, 3)
            shot["eye_plan"] = eye_anim.plan(dur, seed=seed_, emotion=emo, intensity=inten, mode=emode, hit_at=hit_at, glow_at=glow_at)
            shot["eyes"] = {"mode": emode, "cut": eye_cut}
        kind = title_at.get(idx)
        if kind:
            lay = titles.render_title(meta.get("series_title_ja") or "" if kind == "series" else "",
                                      (meta.get("series_title_ar") or meta.get("title_ar") or "") if kind == "series"
                                      else (meta.get("episode_title_ar") or meta.get("title_ar") or ""),
                                      kind, W / "titles", f"{kind}_{idx:03d}",
                                      label=None if kind == "series" else f"الحلقة {meta.get('episode', 1)}")
            if lay:
                tdur = min(dur - 0.3, 4.6 if kind == "series" else 3.4)
                shot["title3d"] = {"layers": lay, "kind": kind, "at": 0.35, "dur": round(max(1.2, tdur), 2)}
                shot["camera"]["dof"] = False
                if kind == "series":
                    for d in ("embers", "light_rays"):
                        if not any(f["kind"] == d for f in shot["fx"]):
                            shot["fx"].append({"kind": d, "amount": 0.7, "at": 0.0, "dur": None})
                    shot["audio"]["sfx"] = (shot["audio"]["sfx"] + [{"kind": "boom", "t": 0.35}, {"kind": "shimmer", "t": 0.5}])[:4]
                    shot["audio"]["music"] = "epic"
            else:
                log.warning("director: title card %s of shot %d skipped (no usable font)", kind, idx)
        shots.append(shot)
        prev_scene, prev_grade, prev_music = im["scene"], grade, music
        t_clock += dur
    total = sum(sh["dur"] for sh in shots)
    log.info("director: %d shots, %.1fs (target %.0fs), %d hit-stops, %d impact frames, %d with 3D dressing, %d eye close-ups, %d animated "
             "eyes, %d title cards", len(shots), total, target, sum(any(p["kind"] == "hit_stop" for p in sh["post_fx"]) for sh in shots),
             sum(any(p["kind"] == "impact_frame" for p in sh["post_fx"]) for sh in shots), sum(bool(sh["auto_dress"]) for sh in shots),
             n_eye_cuts, sum("eye_plan" in sh for sh in shots), sum("title3d" in sh for sh in shots))
    if write:
        (W / "shots.json").write_text(json.dumps(shots, ensure_ascii=False, indent=1), "utf-8")
    return shots
