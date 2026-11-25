"""Render stage: shot -> drawings (Blender, or the cv fallback) -> post (grade, fx, time remap) -> `shots/shot_XXXX.mp4`.

    render_shots(shots, shard=(i, n), only=None) -> {idx: {"engine","final","seconds"}}

Engine choice (RENDER_ENGINE): `auto` = Blender if found else the OpenCV 2.5D engine (loud warning); `cycles`/`eevee` = Blender
or fail; `cv` = OpenCV engine. A Blender failure on one shot falls back to the cv engine for that shot unless RENDER_STRICT=1.
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import hashlib
import json
import pathlib
import shutil
import time

from .. import post
from ..config import WORK_ROOT, cfg, log


class RenderError(RuntimeError):
    pass


class RenderBudgetExceeded(RenderError):
    """The time budget (--deadline-minutes, e.g. the 6 h CI limit) ended while a shot was being drawn: its drawings are kept."""


def resolve_engine():
    """-> ("blender", path) | ("cv", None)."""
    want = cfg.render_engine
    if want == "cv":
        return "cv", None
    from . import blender_runner
    b = blender_runner.find_blender()
    if b:
        return "blender", b
    if want in ("cycles", "eevee"):
        raise RenderError(f"RENDER_ENGINE={want} but Blender was not found. Install Blender (https://www.blender.org/download/) "
                          "and set BLENDER_BIN, or use RENDER_ENGINE=cv.")
    log.warning("Blender not found - using the OpenCV 2.5D engine (parallax, sway, particles, camera) instead. "
                "Install Blender 4.2+ and set BLENDER_BIN for the full cinematic render.")
    return "cv", None


@functools.lru_cache(4096)
def _file_hash(path, size, mtime_ns):
    h = hashlib.blake2b(digest_size=12)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shot_signature(shot, engine):
    """Fingerprint of everything that decides the pixels of a shot: the plan, the CONTENT of every file it points to (art layers,
    eye maps, AI-motion frames), the engine and the draft scale. Rendered frames are only ever reused when this matches, so a
    restart after the 6 h CI limit resumes exactly what is still valid and nothing stale (an older plan, another resolution)."""
    files = {}
    root = str(WORK_ROOT)

    def portable(x):                       # the same plan checked out in another directory (CI runner, your PC) keeps its signature
        return "<work>" + x[len(root):] if x.startswith(root) else x

    def walk(x, key=""):
        if isinstance(x, dict):
            return {k: walk(v, k) for k, v in sorted(x.items()) if k not in ("paths", "signature")}
        if isinstance(x, (list, tuple)):
            return [walk(v, key) for v in x]
        if isinstance(x, str) and len(x) < 600 and ("/" in x or "\\" in x) and "." in x.rsplit("/", 1)[-1]:
            p = pathlib.Path(x)
            try:
                if p.is_file():
                    st = p.stat()
                    files[portable(x)] = _file_hash(str(p), st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        return portable(x) if isinstance(x, str) else x

    body = walk(shot)
    blob = json.dumps({"shot": body, "files": files, "engine": engine, "scale": cfg.render_scale, "samples": cfg.cycles_samples,
                       "v": 2}, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _sig_path(shot):
    return pathlib.Path(shot["paths"]["frames_dir"]) / ".signature"


def _claim_frames(shot, engine):
    """Make `frames_dir` safe to resume: keep it only if it was rendered from this exact plan, otherwise start clean."""
    fd = pathlib.Path(shot["paths"]["frames_dir"])
    sig = shot_signature(shot, engine)
    sp = _sig_path(shot)
    have = sp.read_text("utf-8").strip() if sp.exists() else None
    if fd.exists() and have != sig:
        n = len(list(fd.glob("f_*.png")))
        if n:
            log.info("shot %s: %d cached drawing(s) belong to a different plan/resolution - rendering them again", shot.get("idx"), n)
        shutil.rmtree(fd, ignore_errors=True)
        for stale in (pathlib.Path(shot["paths"]["final"]),):
            for suffix in (".mp4", ".json", ".engine", ".sig"):
                stale.with_suffix(suffix).unlink(missing_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    sp.write_text(sig, "utf-8")
    return sig


def _is_done(shot, engine=None):
    f = pathlib.Path(shot["paths"]["final"])
    j = f.with_suffix(".json")
    if not (f.exists() and f.stat().st_size > 2000 and j.exists()):
        return False
    sig_file = f.with_suffix(".sig")
    if engine is not None:
        # the final file counts only if it was produced from this exact plan (the .sig is written after the post stage succeeds)
        try:
            if sig_file.read_text("utf-8").strip() != shot_signature(shot, engine):
                return False
        except OSError:
            return False
    try:
        return post.count_frames(f) == shot["frames"]
    except Exception:  # noqa: BLE001
        return False


def render_one(shot, engine, binary=None, deadline=None):
    t0 = time.time()
    frames_dir = pathlib.Path(shot["paths"]["frames_dir"])
    used = engine
    sig = _claim_frames(shot, engine)
    if engine == "blender":
        from . import blender_runner
        try:
            left = int(deadline - time.time()) if deadline else None
            frames_dir = blender_runner.render_shot(shot, binary, timeout=max(60, left) if left else None)
        except blender_runner.BlenderTimeout as e:
            if deadline and time.time() >= deadline - 10:
                raise RenderBudgetExceeded(f"shot {shot['idx']}: {e}") from e     # out of time, not a failure: a restart continues here
            if cfg.render_strict:
                raise RenderError(f"Blender failed on shot {shot['idx']}: {e}") from e
            log.error("Blender timed out on shot %s - falling back to the cv engine for this shot", shot["idx"])
            from . import cv_engine
            shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir = cv_engine.render_shot(shot, {"resume": False})
            used = "cv-fallback"
            sig = "fallback:" + sig
            _sig_path(shot).write_text(sig, "utf-8")
        except Exception as e:  # noqa: BLE001
            if cfg.render_strict:
                raise RenderError(f"Blender failed on shot {shot['idx']}: {e}") from e
            log.error("Blender FAILED on shot %s (%s) - falling back to the cv engine for this shot", shot["idx"], str(e)[:400])
            from . import cv_engine
            shutil.rmtree(frames_dir, ignore_errors=True)          # nothing of the failed Blender attempt may leak into the fallback
            frames_dir = cv_engine.render_shot(shot, {"resume": False})
            used = "cv-fallback"
            sig = "fallback:" + sig        # a later run (after you fixed Blender) must NOT treat this as finished, and re-renders it
            _sig_path(shot).write_text(sig, "utf-8")
    else:
        from . import cv_engine
        frames_dir = cv_engine.render_shot(shot, {"resume": True})
    final = post.finish_shot(shot, frames_dir)
    (pathlib.Path(final).with_suffix(".engine")).write_text(used, "utf-8")
    (pathlib.Path(final).with_suffix(".sig")).write_text(sig, "utf-8")      # written LAST: its presence means "this shot is finished"
    return {"engine": used, "final": str(final), "seconds": round(time.time() - t0, 1)}


def render_shots(shots, *, shard=None, only=None, engine=None, progress=None, deadline=None):
    """deadline: absolute time.time() after which no new shot is started and the running Blender shot is stopped (its drawings are kept);
    those shots come back as {"incomplete": True} and `render` exits with code 3 so the workflow can restart the rest."""
    eng, binary = (engine, None) if engine else resolve_engine()
    if eng == "blender" and binary is None:
        from . import blender_runner
        binary = blender_runner.find_blender()
    todo = [s for s in shots if (not shard or s["idx"] % shard[1] == shard[0]) and (not only or s["idx"] in only)]
    log.info("render: engine=%s shots=%d%s", eng, len(todo), f" shard={shard[0]}/{shard[1]}" if shard else "")
    results = {}
    pending = []
    for s in todo:
        if _is_done(s, eng):
            results[s["idx"]] = {"engine": "cached", "final": s["paths"]["final"], "seconds": 0}
        else:
            pending.append(s)
    workers = max(1, cfg.render_workers if eng == "cv" or cfg.render_engine == "eevee" else 1)
    done = len(results)
    if workers == 1:
        for s in pending:
            if deadline and deadline - time.time() < 90:
                results[s["idx"]] = {"engine": "incomplete", "final": None, "seconds": 0, "incomplete": True}
                continue
            try:
                results[s["idx"]] = render_one(s, eng, binary, deadline)
            except RenderBudgetExceeded as e:
                log.warning("time budget reached: %s - the drawings made so far are kept", e)
                results[s["idx"]] = {"engine": "incomplete", "final": None, "seconds": 0, "incomplete": True}
                continue
            done += 1
            if progress:
                progress(f"rendered {done}/{len(todo)}")
            log.info("shot %s done in %ss (%s) [%d/%d]", s["idx"], results[s["idx"]]["seconds"], results[s["idx"]]["engine"], done, len(todo))
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(render_one, s, eng, binary, deadline): s for s in pending}
            for fu in cf.as_completed(futs):
                s = futs[fu]
                try:
                    results[s["idx"]] = fu.result()
                except RenderBudgetExceeded as e:
                    log.warning("time budget reached: %s - the drawings made so far are kept", e)
                    results[s["idx"]] = {"engine": "incomplete", "final": None, "seconds": 0, "incomplete": True}
                    continue
                done += 1
                log.info("shot %s done (%s) [%d/%d]", s["idx"], results[s["idx"]]["engine"], done, len(todo))
    return results


def load_shots(job):
    from ..config import rebase_paths, work_dir
    p = work_dir(job) / "shots.json"
    if not p.exists():
        raise RenderError(f"{p} not found - run `python -m mangaforge prepare` first")
    return rebase_paths(json.loads(p.read_text("utf-8")), job)
