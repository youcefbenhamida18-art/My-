import json
import pathlib

import cv2
import numpy as np
import pytest

from mangaforge import camera_paths as cp
from mangaforge.art import layers
from mangaforge.render import cv_engine, plate_geometry as pg
from tests.synth import key_art


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    d = tmp_path_factory.mktemp("cvart")
    cv2.imwrite(str(d / "k.png"), key_art())
    return layers.build_layers(d / "k.png", d / "o", "k", target_w=1280), d


def make_shot(art, d, move="push_in", fx=(), typ="wide", frames=12, zoom=1.0):
    m = art
    return dict(idx=3, id=3, scene="s", type=typ, dur=frames / 24, frames=frames, fps=24, step=2, size=[640, 360], seed=5, wind=[.5, .05],
                plate=dict(image=m["image"], backdrop=m["backdrop"], depth=m["depth"], sway=m["sway"], rim=m["rim"], subject=m["subject"],
                           palette=m["palette"], ground_y=m["ground_y"], parallax=.35, sequence=None),
                framing={"zoom": zoom, "cx": .5, "cy": .5},
                camera=dict(move=move, strength=.7, dof=True, shake=0.0, hit_at=.25, focal=35, rack=None),
                fx=[{"kind": k, "amount": .7, "at": 0.2, "dur": None} for k in fx], paths=dict(frames_dir=str(d / f"fr_{move}_{typ}_{len(fx)}")))


def test_zero_pose_reproduces_artwork(art):
    e = cv_engine.Engine(make_shot(*art), {})
    x, y, z = e.warp(0, 1, cp.pose("static", 0))
    u = e.grid_u * e.fw + e.cx
    assert np.abs(x - u).max() < 1e-4 and np.abs(z - pg.zeta(e.sample_depth(x, y), e.subject["depth"], .35)).max() < 1e-3


def test_parallax_near_moves_more_than_far(art):
    e = cv_engine.Engine(make_shot(*art), {})
    pose = dict(cp.pose("static", 0), dx=0.03)
    x, y, z = e.warp(0, 1, pose)
    base = e.grid_u * e.fw + e.cx
    shift = (x - base) / e.fw                                   # in frame widths
    near, far = z < 0.97, z > 1.03
    assert near.any() and far.any()
    assert shift[near].mean() > 0.03 > shift[far].mean() > 0
    sub = (np.abs(z - 1) < 0.01)
    assert abs(shift[sub].mean() - 0.03) < 0.004                # subject plane moves exactly with the camera


def test_dolly_scales_around_centre_and_dolly_zoom_keeps_subject_size(art):
    e = cv_engine.Engine(make_shot(*art), {})
    x, y, z = e.warp(0, 1, dict(cp.pose("static", 0), dz=0.2))
    c = (e.H // 2, e.W // 2)
    left = x[c[0], c[1] - 100] - x[c[0], c[1]]
    base = (e.grid_u[c[0], c[1] - 100] - e.grid_u[c[0], c[1]]) * e.fw
    assert left / base < 0.99                                   # window shrank -> magnification
    p = cp.pose("dolly_zoom", 1.0, 1.0)
    x2, y2, z2 = e.warp(0, 1, p)
    sub = np.abs(z2 - 1) < 0.01
    ratio = ((x2 - e.cx) / (e.grid_u * e.fw))[sub & (np.abs(e.grid_u) > .05)]
    assert abs(np.median(ratio) - 1.0) < 0.03                   # subject plane keeps its size


def test_render_writes_contiguous_drawings_and_report(art):
    sh = make_shot(*art, fx=("petals", "snow", "speed_lines", "slash", "shockwave", "wind_sway"), move="orbit_r", typ="action")
    out = cv_engine.render_shot(sh, {"resume": False})
    files = sorted(out.glob("f_*.png"))
    assert [f.name for f in files] == [f"f_{i:04d}.png" for i in range(1, 7)]
    im = cv2.imread(str(files[2]))
    assert im.shape == (360, 640, 3) and im.std() > 10
    rep = json.loads((out / "done.json").read_text())
    assert rep["engine"] == "cv" and rep["features"]["petals"] == "ok" and rep["features"]["slash"] in ("ok",)
    again = cv_engine.render_shot(sh, {"resume": True})          # resume keeps files
    assert len(list(again.glob("f_*.png"))) == 6


def test_frames_differ_over_time(art):
    out = cv_engine.render_shot(make_shot(*art, move="push_in", fx=("embers", "wind_sway")), {"resume": False})
    a, b = (cv2.imread(str(out / f"f_{i:04d}.png")).astype(float) for i in (1, 6))
    assert np.abs(a - b).mean() > 1.5


def test_rim_is_faded_for_heuristic_matte(art):
    sh = make_shot(*art)
    sh["plate"]["matte_method"] = "none"
    assert cv_engine.Engine(sh, {}).rim_scale == 0.0
    sh["plate"]["matte_method"] = "rembg"
    assert cv_engine.Engine(sh, {}).rim_scale == 1.0


@pytest.mark.parametrize("move", ["push_in", "pan_r", "tilt_dn", "orbit_l", "crane_up", "dolly_zoom", "whip_r", "handheld"])
def test_warp_inverts_the_shared_pinhole_projection(art, move):
    """Ray-casting through the exact camera (cv engine) and projecting a plate point forward (what Blender does) are inverses:
    forward-projecting the plate point the warp found for an output pixel returns that pixel (up to the depth fixed-point residual)."""
    e = cv_engine.Engine(make_shot(*art, move=move), {})
    pose = cp.total_pose(move, 0.7, 0.9, 5, 14, 24, 0.2, 6)
    x, y, z = e.warp(0, 1, pose)
    u0, v0 = (x - e.cx) / e.fw, (y - e.cy) / e.fw
    uu, vv = pg.project(u0, v0, z, pose, e.k * pg.SENSOR_W, e.subject_uv)
    ok = (x > 0.02) & (x < 0.98) & (y > 0.02) & (y < 0.98)                 # plate edge: the depth lookup is clamped there
    assert ok.mean() > 0.6
    assert np.abs(uu - e.grid_u)[ok].max() < 4e-3 and np.abs(vv - e.grid_v)[ok].max() < 4e-3
    assert np.median(np.abs(uu - e.grid_u)[ok]) < 2e-4
