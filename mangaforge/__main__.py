"""python -m mangaforge <command>   (see README.md; `python -m mangaforge --help`)"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import config
from .config import cfg, log


def _shard(s):
    if not s:
        return None
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError("--shard must look like 0/4 (index/count)")
    return i, n


def _gha(**kv):
    f = os.environ.get("GITHUB_OUTPUT")
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            for k, v in kv.items():
                fh.write(f"{k}={v}\n")


def _job(a):
    if getattr(a, "job", None):
        return a.job
    p = config.WORK_ROOT / "current_job.txt"
    if p.exists():
        return p.read_text("utf-8").strip()
    from . import pipeline
    return pipeline.job_id(config.load_state())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mangaforge", description="Daily 2-minute anime shorts -> weekly episode")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("doctor", help="check the installation"); s.add_argument("--live", action="store_true"); s.add_argument("--blender", action="store_true")
    sub.add_parser("selftest", help="render a tiny shot through the real Blender scene script")
    s = sub.add_parser("features", help="how the 100 Blender capabilities map to this pipeline"); s.add_argument("--json", action="store_true"); s.add_argument("--job")
    s = sub.add_parser("demo", help="offline end-to-end test (no keys needed)"); s.add_argument("--engine", choices=["cv", "blender"]); s.add_argument("--seconds", type=float, default=30.0)
    s = sub.add_parser("prepare", help="story, voices, art, shot plan"); s.add_argument("--job"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("render", help="render (a shard of) the shots"); s.add_argument("--job"); s.add_argument("--shard", type=_shard); s.add_argument("--only")
    s.add_argument("--deadline-minutes", type=float, help="stop cleanly after this many minutes (exit code 3 = unfinished shots; run again to continue)")
    sub.add_parser("jobid", help="print the id of the job the next `prepare` will make (e01p3)")
    s = sub.add_parser("assemble", help="join shots, audio and subtitles"); s.add_argument("--job")
    s = sub.add_parser("short", help="all stages for the next daily short"); s.add_argument("--job")
    s = sub.add_parser("weekly", help="join the week's shorts into one episode"); s.add_argument("--parts-dir")
    s = sub.add_parser("publish", help="upload to YouTube (scheduled)"); s.add_argument("--job"); s.add_argument("--weekly", action="store_true")
    sub.add_parser("state", help="print the series state")
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        from . import doctor
        return doctor.print_report(doctor.run_checks(live=a.live, blender=a.blender))
    if a.cmd == "selftest":
        from .render import blender_runner
        b = blender_runner.find_blender()
        if not b:
            print("Blender not found. Install Blender 4.2 LTS+ and set BLENDER_BIN.")
            return 1
        r = blender_runner.selftest(b)
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0 if r.get("ok") else 1
    if a.cmd == "features":
        from .render import features
        print(features.report(job=_job(a) if a.job else None, as_json=a.json))
        return 0
    if a.cmd == "demo":
        from . import demo
        r = demo.run(engine=a.engine, seconds=a.seconds)
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0
    from . import pipeline
    if a.cmd == "jobid":
        job = pipeline.job_id(config.load_state())
        print(job)
        _gha(job=job)
        return 0
    if a.cmd == "prepare":
        r = pipeline.prepare(a.job, force=a.force)
        print(json.dumps(r, indent=1))
        n = max(1, min(int(os.environ.get("SHARDS", "8")), int(r["shots"])))
        _gha(job=r["job"], shards=json.dumps(list(range(n))), shard_count=n)
        return 0
    if a.cmd == "render":
        import time
        only = {int(x) for x in a.only.split(",")} if a.only else None
        deadline = time.time() + a.deadline_minutes * 60 if a.deadline_minutes else None
        r = pipeline.render_stage(_job(a), shard=a.shard, only=only, deadline=deadline)
        print(json.dumps({k: v["engine"] for k, v in r.items()}))
        left = sorted(k for k, v in r.items() if v.get("incomplete"))
        _gha(incomplete=str(bool(left)).lower())
        if left:
            print(f"UNFINISHED shots {left}: the time budget ended - run `render` again, finished drawings are reused", file=sys.stderr)
            return 3
        return 0
    if a.cmd == "assemble":
        job = _job(a)
        r = pipeline.assemble_stage(job)
        state = config.load_state()
        script = json.loads((config.work_dir(job) / "script.json").read_text("utf-8"))
        pipeline._advance(state, job, script, r)
        due = state["part"] > cfg.parts_per_episode
        _gha(video=r["video"], weekly_due=str(due).lower(), job=job)
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "short":
        print(json.dumps(pipeline.run_short(job=a.job), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "weekly":
        r = pipeline.run_weekly(parts_dir=a.parts_dir)
        _gha(episode_video=r["video"])
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "publish":
        from . import publish
        return publish.run(job=_job(a) if not a.weekly else None, weekly=a.weekly)
    if a.cmd == "state":
        print(json.dumps(config.load_state(), indent=1, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
