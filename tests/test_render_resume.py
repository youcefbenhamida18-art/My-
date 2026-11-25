"""The render stage must be safe to RESTART: frames and finished shots are reused only when they were produced from exactly the
same plan (this once silently reused 960x540 drawings of an older run for a 1920x1080 plan), and post copes with the real frame size."""
import json
import pathlib
import subprocess

import cv2
import pytest

from mangaforge import post
from mangaforge.art import layers
from mangaforge.config import cfg
from mangaforge.render import _claim_frames, _is_done, render_one, shot_signature
from tests.synth import key_art


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    d = tmp_path_factory.mktemp("resume_art")
    cv2.imwrite(str(d / "k.png"), key_art())
    return layers.build_layers(d / "k.png", d / "o", "k", target_w=640)


def make_shot(plate, d, *, size=(320, 180), frames=8, move="push_in", fx=("dust",), name="a"):
    return dict(idx=1, id=1, scene="s", type="wide", dur=frames / 24, frames=frames, fps=24, step=2, size=list(size), seed=5, wind=[.5, .05],
                plate=dict(image=plate["image"], backdrop=plate["backdrop"], depth=plate["depth"], sway=plate["sway"], rim=plate["rim"],
                           subject=plate["subject"], palette=plate["palette"], ground_y=plate["ground_y"], parallax=.35, sequence=None),
                framing={"zoom": 1.0, "cx": .5, "cy": .5}, grade={"preset": "night"},
                camera=dict(move=move, strength=.7, dof=True, shake=0.0, hit_at=.25, focal=35, rack=None),
                fx=[{"kind": k, "amount": .7, "at": 0.2, "dur": None} for k in fx],
                paths=dict(frames_dir=str(d / "frames" / name), final=str(d / "shots" / f"{name}.mp4")))


def probe(path, entries="width,height,nb_frames"):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={entries}", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out.split(",")


def test_post_scales_frames_that_do_not_match_the_plan_and_keeps_every_effect(plate, tmp_path):
    shot = make_shot(plate, tmp_path)
    fd = pathlib.Path(shot["paths"]["frames_dir"])
    fd.mkdir(parents=True)
    img = key_art()
    for i in range(1, 5):                                           # drawings at HALF the planned size
        cv2.imwrite(str(fd / f"f_{i:04d}.png"), cv2.resize(img, (160, 90)))
    final = post.finish_shot(shot, fd, vertical=False)
    rep = json.loads(pathlib.Path(final).with_suffix(".json").read_text())
    assert rep["degraded"] == []
    assert rep["features"]["post_bloom"] == "ok" and rep["features"]["post_grade"] == "ok"
    assert probe(final) == ["320", "180", "8"]


def test_claim_keeps_matching_frames_and_wipes_stale_ones(plate, tmp_path):
    shot = make_shot(plate, tmp_path)
    sig = _claim_frames(shot, "cv")
    fd = pathlib.Path(shot["paths"]["frames_dir"])
    (fd / "f_0001.png").write_bytes(b"x" * 2000)
    assert _claim_frames(shot, "cv") == sig and (fd / "f_0001.png").exists()          # same plan: resumable

    other = make_shot(plate, tmp_path, move="orbit_l")                                # the plan changed
    assert shot_signature(other, "cv") != sig
    _claim_frames(other, "cv")
    assert not (fd / "f_0001.png").exists() and (fd / ".signature").read_text() == shot_signature(other, "cv")

    (fd / "f_0001.png").write_bytes(b"x" * 2000)
    _claim_frames(other, "blender")                                                   # another engine never inherits frames
    assert not (fd / "f_0001.png").exists()


def test_signature_follows_file_content_and_draft_scale(plate, tmp_path, monkeypatch):
    shot = make_shot(plate, tmp_path)
    base = shot_signature(shot, "cv")
    assert shot_signature(json.loads(json.dumps(shot)), "cv") == base                  # deterministic
    monkeypatch.setattr(cfg, "render_scale", 0.5)
    assert shot_signature(shot, "cv") != base
    monkeypatch.setattr(cfg, "render_scale", 1.0)
    img = pathlib.Path(shot["plate"]["image"])
    old = img.read_bytes()
    try:
        im = cv2.imread(str(img))
        im[0:8, 0:8] = (1, 2, 3)
        cv2.imwrite(str(img), im)                                                      # new art under the same file name
        assert shot_signature(shot, "cv") != base
    finally:
        img.write_bytes(old)
    assert shot_signature(shot, "cv") == base


def test_a_finished_shot_is_reused_only_for_the_same_plan(plate, tmp_path):
    shot = make_shot(plate, tmp_path)
    assert not _is_done(shot, "cv")
    res = render_one(shot, "cv")
    assert res["engine"] == "cv" and _is_done(shot, "cv")
    assert not _is_done(make_shot(plate, tmp_path, move="orbit_r"), "cv")              # same file names, different plan
    assert not _is_done(shot, "blender")
    final = pathlib.Path(shot["paths"]["final"])
    final.with_suffix(".sig").unlink()                                                 # interrupted before the last write
    assert not _is_done(shot, "cv")


def test_restart_reuses_valid_drawings_and_never_mixes_plans(plate, tmp_path):
    shot = make_shot(plate, tmp_path, frames=10)
    render_one(shot, "cv")
    fd = pathlib.Path(shot["paths"]["frames_dir"])
    first = fd / "f_0002.png"
    stamp = first.stat().st_mtime_ns
    pathlib.Path(shot["paths"]["final"]).with_suffix(".sig").unlink()                  # as after a killed job: frames exist, no .sig
    render_one(shot, "cv")
    assert first.stat().st_mtime_ns == stamp                                           # drawings were reused, not redrawn

    changed = make_shot(plate, tmp_path, frames=10, fx=("dust", "petals"))
    render_one(changed, "cv")
    assert first.stat().st_mtime_ns != stamp                                           # a different plan redraws everything


def test_a_truncated_drawing_is_redrawn_not_trusted(plate, tmp_path):
    from mangaforge.render.cv_engine import png_complete
    shot = make_shot(plate, tmp_path, frames=8)
    render_one(shot, "cv")
    fd = pathlib.Path(shot["paths"]["frames_dir"])
    good = fd / "f_0003.png"
    assert png_complete(good)
    data = good.read_bytes()
    good.write_bytes(data[: len(data) // 2])                       # the job was killed in the middle of writing it
    assert not png_complete(good)
    pathlib.Path(shot["paths"]["final"]).with_suffix(".sig").unlink()
    render_one(shot, "cv")
    assert png_complete(good) and good.stat().st_size == len(data)


def test_deadline_stops_cleanly_and_a_second_run_finishes_the_rest(plate, tmp_path):
    import time
    from mangaforge.render import render_shots
    shots = [make_shot(plate, tmp_path, name=f"s{i}") | {"idx": i, "id": i} for i in range(3)]
    for i, s in enumerate(shots):
        s["paths"]["frames_dir"] = str(tmp_path / "frames" / f"s{i}")
        s["paths"]["final"] = str(tmp_path / "shots" / f"s{i}.mp4")
    res = render_shots(shots, engine="cv", deadline=time.time() + 5)                   # less than the 90 s safety margin: nothing starts
    assert all(r.get("incomplete") for r in res.values()) and not any(pathlib.Path(s["paths"]["final"]).exists() for s in shots)
    res = render_shots(shots, engine="cv", deadline=time.time() + 3600)
    assert not any(r.get("incomplete") for r in res.values()) and all(pathlib.Path(s["paths"]["final"]).exists() for s in shots)
    res = render_shots(shots, engine="cv")
    assert {r["engine"] for r in res.values()} == {"cached"}


def test_a_plan_written_on_another_machine_is_re_rooted_to_this_checkout(tmp_path, monkeypatch):
    from mangaforge import config
    from mangaforge.render import load_shots
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path / "work")
    import mangaforge.render as render_pkg
    job = "e07p3"
    d = config.work_dir(job)
    foreign = "/home/runner/work/studio/studio/work/e07p3/"
    plan = [{"idx": 1, "paths": {"frames_dir": foreign + "frames/shot_0001", "final": foreign + "shots/shot_0001.mp4"},
             "plate": {"image": foreign + "art/i01.png"}, "note": "/somewhere/else/keep.png"}]
    (d / "shots.json").write_text(json.dumps(plan), "utf-8")
    shots = load_shots(job)
    assert shots[0]["paths"]["frames_dir"] == str(d / "frames/shot_0001")
    assert shots[0]["plate"]["image"] == str(d / "art/i01.png")
    assert shots[0]["note"] == "/somewhere/else/keep.png"                    # paths outside the job are left alone
    assert render_pkg is not None
