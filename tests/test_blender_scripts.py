"""Executes the REAL Blender scripts (blender_scene + blender_fx + blender_extras) against tests/fake_bpy.py: every code path runs on a
synthetic shot, node sockets / enums are validated, and the camera + mesh maths are checked numerically against plate_geometry (the
model the cv engine uses). Blender itself is only available in CI (see the workflow's `selftest` job)."""
import json
import math
import os
import pathlib
import runpy
import sys

import cv2
import numpy as np
import pytest

from mangaforge import camera_paths as cp
from mangaforge import schema
from mangaforge.render import blender_runner as br
from mangaforge.render import plate_geometry as pg
from tests import fake_bpy

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "mangaforge" / "render" / "blender_scene.py"


def make_shot(d, **over):
    shot = br._selftest_shot(d)
    shot["fx"] = [{"kind": k, "amount": 0.7, "at": 0.1 if k in ("slash", "shockwave", "sparks", "debris", "lightning") else 0.0, "dur": None}
                  for k in schema.FX_BLENDER]
    shot.update(over)
    return shot


def run_scene(tmp_path, shot, **settings):
    bpy = fake_bpy.install()
    st = br._settings({"samples": 4, "physics": False, **settings})
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"shot": shot, "settings": st}), "utf-8")
    argv, env = sys.argv, os.environ.get("MANGAFORGE_ROOT")
    sys.argv = ["blender", "-b", "--", "--job", str(job)]
    os.environ["MANGAFORGE_ROOT"] = str(ROOT)
    code, g = 0, {}
    try:
        g = runpy.run_path(str(SCENE), run_name="__main__")
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv = argv
        if env is None:
            os.environ.pop("MANGAFORGE_ROOT", None)
        else:
            os.environ["MANGAFORGE_ROOT"] = env
    done = pathlib.Path(shot["paths"]["frames_dir"]) / "done.json"
    rep = json.loads(done.read_text("utf-8")) if done.exists() else None
    bpy.g = g
    return bpy, code, rep


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    fake_bpy.uninstall()


def test_full_shot_runs_every_feature_without_failures(tmp_path):
    bpy, code, rep = run_scene(tmp_path, make_shot(tmp_path))
    assert code == 0 and rep is not None
    bad = {k: v for k, v in rep["features"].items() if str(v).startswith("failed")}
    assert not bad, bad
    frames = sorted(pathlib.Path(tmp_path / "frames").glob("f_*.png"))
    assert len(frames) == 6                                                   # 12 frames on twos
    print(json.dumps(rep["features"], indent=1))


# ---------------------------------------------------------------- geometry: Blender camera == plate_geometry model
ASPECT = pg.ASPECT


def blender_project(bpy, verts, f, lens):
    """Pinhole projection through the keyframed fake camera -> (u in frame widths, v in frame HEIGHTS, down)."""
    cam = next(o for o in bpy.data.objects if o.name == "cam")
    loc = np.asarray(cam._keys[("location", f)], float)
    R = cam._keys[("rotation_euler", f)].to_matrix().m[:3, :3]
    pc = (np.asarray(verts, float) - loc) @ R                                  # R^T (P - loc)
    depth = -pc[:, 2]
    k = lens / pg.SENSOR_W
    return k * pc[:, 0] / depth, -k * pc[:, 1] / depth * ASPECT, depth


def plate_vertices(g):
    mg = g["MG"]
    X, Y, Z = mg["x"], mg["y"], mg["zeta"]
    u0, v0 = (X - g["CX"]) / g["FW"], (Y - g["CY"]) / g["FW"]
    verts = np.stack([u0 * g["WF0"] * Z, Z * g["D"], -v0 * g["HF0"] * Z], -1).reshape(-1, 3)
    return u0.ravel(), v0.ravel(), Z.ravel(), verts


MOVES = ["push_in", "pull_out", "drift", "crane_up", "crane_dn", "pan_l", "pan_r", "tilt_up", "tilt_dn", "orbit_l", "orbit_r",
         "dolly_zoom", "whip_l", "whip_r", "handheld", "punch_in", "static"]


@pytest.mark.parametrize("move", MOVES)
def test_blender_camera_matches_the_shared_pinhole_model(tmp_path, move):
    """The Blender camera (position + Euler angles, through the lens) must land every plate vertex exactly where
    plate_geometry.project says - the same model the cv engine ray-casts through. Includes shake / impact roll."""
    shot = make_shot(tmp_path, camera={"move": move, "strength": 0.9, "dof": False, "shake": 0.25, "hit_at": 0.2, "focal": 35,
                                       "rack": None}, fx=[], title3d=None, auto_dress=[], seed=7)
    bpy, code, rep = run_scene(tmp_path, shot)
    assert code == 0 and rep and not [k for k, v in rep["features"].items() if str(v).startswith("failed")]
    g = bpy.g
    u0, v0, Z, verts = plate_vertices(g)
    seed, frames, fps = shot["seed"], shot["frames"], shot["fps"]
    sub_uv = ((g["SUB"]["x"] - g["CX"]) / g["FW"], (g["SUB"]["y"] - g["CY"]) / g["FW"])
    for f in (1, 4, 8, 12):
        tn = (f - 1) / max(1, frames - 1)
        pose = cp.total_pose(move, tn, 0.9, seed, f, fps, 0.25, 1 + int(round(0.2 * fps)))
        lens = 35.0 * pose["focal_scale"]
        u, v, depth = blender_project(bpy, verts, f, lens)
        ur, vr = pg.project(u0, v0, Z, pose, 35.0, sub_uv)
        sel = (np.abs(ur) < 0.55) & (np.abs(vr) < 0.55)
        assert sel.sum() > 500
        err = max(np.abs(u - ur)[sel].max(), np.abs(v - vr)[sel].max())
        assert err < 1e-6, (move, f, err)


@pytest.mark.parametrize("move", ["orbit_l", "orbit_r"])
def test_orbit_is_an_exact_pinhole_orbit_around_the_subject(tmp_path, move):
    """Closed form for the camera sliding on a circle (radius D) around the subject plane while looking at it (frame 1: no truck/dolly):
        u = [u0 z cos p + k (z - 1) sin p] / [1 + (z - 1) cos p - (u0 z / k) sin p]       v = v0 z / [same denominator]"""
    shot = make_shot(tmp_path, camera={"move": move, "strength": 0.9, "dof": False, "shake": 0.0, "hit_at": None, "focal": 35,
                                       "rack": None}, fx=[], title3d=None, auto_dress=[], seed=7)
    bpy, code, rep = run_scene(tmp_path, shot)
    assert code == 0
    u0, v0, Z, verts = plate_vertices(bpy.g)
    phi = math.radians(cp.pose(move, 0.0, 0.9)["orbit_deg"])
    assert abs(phi) > math.radians(5)
    k = 35.0 / pg.SENSOR_W
    den = 1 + (Z - 1) * math.cos(phi) - (u0 * Z / k) * math.sin(phi)
    ue = (u0 * Z * math.cos(phi) + k * (Z - 1) * math.sin(phi)) / den
    ve = v0 * Z / den
    u, v, _ = blender_project(bpy, verts, 1, 35.0)
    assert np.abs(u - ue).max() < 1e-6 and np.abs(v - ve).max() < 1e-6
    # the subject point (z = 1, u0 = v0 = 0) stays exactly in the middle of the frame
    sub = bpy.g["S_W"]
    uu, vv, _ = blender_project(bpy, np.asarray([sub.v]), 1, 35.0)
    assert abs(uu[0]) < 1e-9


# ---------------------------------------------------------------- eye patches, physics tier, safe mode, every effect alone
def _patch_files(d, n, w=48, h=32):
    out = []
    for i in range(n):
        im = np.zeros((h, w, 4), np.uint8)
        im[..., :3] = (40 + 30 * i, 120, 200)
        im[8:24, 6:42, 3] = 255 if i % 2 else 120
        p = d / f"patch_{i}.png"
        cv2.imwrite(str(p), im)
        out.append(str(p))
    return out


def test_eye_patches_sit_on_the_plate_surface_and_follow_the_drawings(tmp_path):
    files = _patch_files(tmp_path, 3)
    seq = [files[0], files[0], files[1], files[1], files[2], files[2]]                      # one file per drawing (6)
    box = [0.42, 0.28, 0.58, 0.40]
    shot = make_shot(tmp_path, fx=[], title3d=None, auto_dress=[], eye_patches=[{"files": seq, "box": box, "side": "left", "face": 0,
                                                                              "report": {}}],
                     camera={"move": "orbit_r", "strength": 0.9, "dof": False, "shake": 0.0, "hit_at": None, "focal": 35, "rack": None})
    bpy, code, rep = run_scene(tmp_path, shot)
    assert code == 0 and rep["features"]["eye_patches"] == "ok"
    g = bpy.g
    ob = next(o for o in bpy.data.objects if o.name == "eye_patch_0")
    verts = np.asarray(ob.data._co, float)
    nx, ny = 12, 8
    xs, ys = np.linspace(box[0], box[2], nx + 1), np.linspace(box[1], box[3], ny + 1)
    XX, YY = np.meshgrid(xs, ys)
    u0, v0 = ((XX - g["CX"]) / g["FW"]).ravel(), ((YY - g["CY"]) / g["FW"]).ravel()
    Zs = g["zeta_at"](XX, YY).ravel()
    sub_uv = ((g["SUB"]["x"] - g["CX"]) / g["FW"], (g["SUB"]["y"] - g["CY"]) / g["FW"])
    for f in (1, 7, 12):
        u, v, _ = blender_project(bpy, verts, f, 35.0)
        pose = cp.total_pose("orbit_r", (f - 1) / 11, 0.9, shot["seed"], f, 24, 0.0, None)
        ur, vr = pg.project(u0, v0, Zs, pose, 35.0, sub_uv)
        assert np.abs(u - ur).max() < 4e-3 and np.abs(v - vr).max() < 4e-3                    # a hair in front of the plate, same place
    mat = ob.data.materials[0]
    tex = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage")
    assert pathlib.Path(tex.image.filepath).name == "patch_2.png"                             # last drawing -> last patch picture
    loaded = {pathlib.Path(i.filepath).name for i in bpy.data.images if getattr(i, "filepath", "").endswith(".png")}
    assert {"patch_0.png", "patch_1.png", "patch_2.png"} <= loaded


def test_physics_and_volumetric_tier(tmp_path):
    bpy, code, rep = run_scene(tmp_path, make_shot(tmp_path), physics=True, volumetrics=True, save_blend=True)
    assert code == 0
    f = rep["features"]
    bad = {k: v for k, v in f.items() if str(v).startswith("failed")}
    assert not bad, bad
    for k in ("mantaflow_smoke", "mantaflow_fire", "cloth_simulation", "volumetrics_principled"):
        assert f[k] == "ok", (k, f[k])
    names = [c[0] for c in bpy._calls]
    assert "object.quick_smoke" in names and "wm.save_as_mainfile" in names


def test_safe_mode_is_core_only(tmp_path):
    bpy, code, rep = run_scene(tmp_path, make_shot(tmp_path), safe_mode=True)
    assert code == 0 and rep["safe_mode"] is True
    f = rep["features"]
    assert f["fx_engine"].startswith("skipped") and f["extras_engine"].startswith("skipped")
    assert f["plate_depth_mesh"] == "ok" and f["camera_moves"] == "ok"
    assert not [k for k, v in f.items() if str(v).startswith("failed")]
    assert len(list((tmp_path / "frames").glob("f_*.png"))) == 6


@pytest.mark.parametrize("kind", schema.FX_BLENDER)
def test_each_effect_alone_runs_clean(tmp_path, kind):
    shot = make_shot(tmp_path, title3d=None, auto_dress=[], fx=[{"kind": kind, "amount": 0.9, "at": 0.0, "dur": None}])
    bpy, code, rep = run_scene(tmp_path, shot, physics=kind in ("smoke", "fire", "cloth_flag"), volumetrics=kind == "volume_fog")
    assert code == 0 and rep is not None
    bad = {k: v for k, v in rep["features"].items() if str(v).startswith("failed")}
    assert not bad, (kind, bad)
    assert kind in rep["features"] or kind in ("wind_sway", "hair_sway", "heat_haze", "water_ripple", "glow_pulse", "grass", "pebbles",
                                               "slash", "lightning", "swirl", "cloth_flag"), f"{kind} left no trace in the report"


def test_zoomed_framing_and_ai_animated_plate(tmp_path):
    shot = make_shot(tmp_path, framing={"zoom": 2.3, "cx": 0.58, "cy": 0.42})
    shot["plate"]["sequence"] = {"dir": str(tmp_path), "pattern": "f_%04d.png", "frames": 12, "fps": 24}
    bpy, code, rep = run_scene(tmp_path, shot)
    assert code == 0
    f = rep["features"]
    assert f["i2v_plate"] == "ok"
    assert not [k for k, v in f.items() if str(v).startswith("failed")]
