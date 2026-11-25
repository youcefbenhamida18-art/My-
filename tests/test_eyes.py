"""Eye system + title cards: pure logic (eye_anim), pixel analysis / layers on a synthetic anime face (detection is faked - the
real one is a vision LLM), and the cv-engine rig (blink, glow, gaze, title) - the engine that runs when Blender is unavailable."""
import json
import math

import cv2
import numpy as np
import pytest

from mangaforge import eye_anim, eyes, titles
from mangaforge.art import layers
from mangaforge.config import cfg
from mangaforge.render import cv_engine
from tests.synth_face import anime_face


# ---------------------------------------------------------------- eye_anim (pure)
def test_plan_is_deterministic_and_bounded():
    a = eye_anim.plan(6.0, seed=11, emotion="neutral")
    b = eye_anim.plan(6.0, seed=11, emotion="neutral")
    c = eye_anim.plan(6.0, seed=12, emotion="neutral")
    assert a == b and a != c
    assert json.loads(json.dumps(a)) == a                                   # JSON-able (travels inside shot.json)
    assert all(0 <= t0 < 6.0 and 0.1 < d < 0.4 for t0, d in a["blinks"])
    assert all(a["blinks"][i][0] < a["blinks"][i + 1][0] for i in range(len(a["blinks"]) - 1))
    for k in range(0, 60):
        st = eye_anim.state_at(a, k / 10.0)
        assert 0.0 <= st["lid"] <= 1.0 and -1.01 <= st["gx"] <= 1.01 and 0.5 < st["pupil"] < 1.3 and 0.0 <= st["glint"] <= 1.0


def test_blink_closes_and_reopens_and_lid_index_maps_to_drawings():
    p = eye_anim.plan(3.0, seed=2)
    p["blinks"] = [[1.0, 0.24]]
    p["droop"] = 0.0
    assert eye_anim.state_at(p, 0.5)["lid"] == 0.0
    assert eye_anim.state_at(p, 1.0 + 0.24 * 0.42)["lid"] == pytest.approx(1.0)
    assert eye_anim.state_at(p, 1.5)["lid"] == 0.0
    lids = [eye_anim.state_at(p, 1.0 + 0.24 * x / 20)["lid"] for x in range(21)]
    assert max(lids) == pytest.approx(1.0) and lids[0] == 0.0 and lids[-1] == 0.0
    assert [eye_anim.lid_index(v) for v in (0.0, 0.2, 0.45, 0.7, 0.85, 1.0)] == [0, 0, 1, 2, 2, 3]


def test_glow_ramps_and_hit_shrinks_the_pupil():
    p = eye_anim.plan(2.0, seed=4, mode="glow", glow_at=0.4)
    assert eye_anim.state_at(p, 0.3)["glow"] == 0.0
    assert 0.1 < eye_anim.state_at(p, 0.55)["glow"] < eye_anim.state_at(p, 0.75)["glow"]
    assert eye_anim.state_at(p, 1.2)["glow"] > 0.85
    q = eye_anim.plan(2.0, seed=4, hit_at=0.5)
    assert eye_anim.state_at(q, 0.62)["pupil"] < eye_anim.state_at(q, 0.3)["pupil"] * 0.85
    t = eye_anim.plan(2.0, seed=4, mode="tears")
    assert eye_anim.state_at(t, 0.1)["tear"] == 0.0 and eye_anim.state_at(t, 1.9)["tear"] > 0.5


def test_no_blink_right_at_the_hit():
    p = eye_anim.plan(8.0, seed=5, hit_at=3.0)
    assert not any(2.8 < t0 < 3.45 for t0, _ in p["blinks"])


# ---------------------------------------------------------------- analysis + layers on a synthetic face
@pytest.fixture(scope="module")
def face_art(tmp_path_factory):
    d = tmp_path_factory.mktemp("face")
    img, eboxes, fbox = anime_face()
    cv2.imwrite(str(d / "face.png"), img)
    m = layers.build_layers(d / "face.png", d / "o", "face", target_w=1600)
    H, W = img.shape[:2]
    return m, d, img, eboxes, fbox, (W, H)


def _fake_detect(eboxes, fbox, size):
    W, H = size

    def detect(src, hint=None):
        return {"faces": [{"box": (fbox[0] / W, fbox[1] / H, fbox[2] / W, fbox[3] / H), "facing": "front", "char": "ren", "emotion": "neutral",
                           "eyes": [{"box": (b[0] / W, b[1] / H, b[2] / W, b[3] / H)} for b in eboxes]}], "method": "fake"}
    return detect


@pytest.fixture(scope="module")
def eye_info(face_art):
    m, d, img, eboxes, fbox, size = face_art
    mp = pytest.MonkeyPatch()
    mp.setattr(eyes, "detect", _fake_detect(eboxes, fbox, size))
    mp.setattr(cfg, "eyes", True)
    try:
        info = eyes.process(m["image"], {"id": "i1", "characters": ["ren"], "no_humans": False}, {}, d / "o", "face")
    finally:
        mp.undo()
    return info


def test_process_finds_both_eyes_and_writes_layers(face_art, eye_info):
    m, d, img, eboxes, fbox, (W, H) = face_art
    assert eye_info and len(eye_info["eyes"]) == 2 and {e["side"] for e in eye_info["eyes"]} == {"left", "right"}
    assert (d / "o" / "face_eyes.json").exists()
    assert eyes.load(d / "o" / "face_eyes.json")["size"] == [W, H]
    for e in eye_info["eyes"]:
        cx, cy, rx, ry = e["iris"]
        ex = (eboxes[0] if e["side"] == "left" else eboxes[1])
        assert ex[0] - 20 < cx < ex[2] + 20 and ex[1] - 20 < cy < ex[3] + 20          # iris centre inside the real eye
        assert 0.15 * (ex[2] - ex[0]) < rx < 0.45 * (ex[2] - ex[0])
        L = e["layers"]
        for p in L["closed"] + [L["shine"], L["glow"]]:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            assert im is not None and im.ndim == 3 and im.shape[2] == 4 and im.shape[:2] == (L["size"][1], L["size"][0])
        closed = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in L["closed"]]
        cover = [float(c[..., 3].mean()) for c in closed]
        assert cover[0] < cover[1] < cover[2]                                           # lids close progressively
        lx, ly = int(cx - L["origin"][0]), int(cy - L["origin"][1])
        assert closed[2][ly, lx, 3] > 230                                              # fully closed lid covers the iris centre
        assert closed[0][ly, lx, 3] < closed[2][ly, lx, 3]
        skin = np.array([246, 206, 180])                                               # the synthetic skin, as cv2 reads it (BGR)
        assert np.abs(closed[2][ly, lx, :3].astype(int) - skin).max() < 40             # lid is skin coloured (no RGB/BGR mix-up)
    assert cv2.imread(eye_info["mask"], cv2.IMREAD_GRAYSCALE).max() > 200


def test_enhance_does_not_destroy_the_picture(face_art, eye_info):
    m, d, img, *_ = face_art
    after = cv2.cvtColor(cv2.imread(m["image"]), cv2.COLOR_BGR2RGB)
    assert after.shape[:2] == (img.shape[0], img.shape[1]) or after.shape[1] >= 1600


def test_process_is_off_without_humans_or_when_disabled(face_art, monkeypatch):
    m, d, img, eboxes, fbox, size = face_art
    monkeypatch.setattr(eyes, "detect", _fake_detect(eboxes, fbox, size))
    assert eyes.process(m["image"], {"id": "x", "characters": [], "no_humans": True}, {}, d / "o", "nohum") is None
    monkeypatch.setattr(cfg, "eyes", False)
    assert eyes.process(m["image"], {"id": "x", "characters": ["ren"]}, {}, d / "o", "off") is None


def test_detection_failure_degrades_to_no_eyes(face_art, monkeypatch):
    m, d, *_ = face_art
    monkeypatch.setattr(cfg, "eyes", True)
    monkeypatch.setattr(eyes, "detect", lambda src, hint=None: {"faces": [], "method": "unavailable", "error": "no key"})
    info = eyes.process(m["image"], {"id": "x", "characters": ["ren"]}, {}, d / "o", "fail")
    assert info["eyes"] == [] and info["error"] == "no key"


def test_closeup_framing_centres_on_the_eyes(eye_info):
    zoom, cx, cy = eyes.closeup_framing(eye_info)
    assert 2.0 <= zoom <= 3.6
    ys = [(e["box"][1] + e["box"][3]) / 2 for e in eye_info["eyes"]]
    assert abs(cy - sum(ys) / 2) < 0.02 and 0.4 < cx < 0.6
    assert eyes.closeup_framing({"eyes": []}) is None and eyes.closeup_framing(None) is None
    left = eyes.closeup_framing(eye_info, which="left")
    assert left[1] < cx and left[0] >= zoom


# ---------------------------------------------------------------- the cv-engine rig
def _shot(face_art, eye_info, plan, tmp, *, frames=16, move="static", zoom=1.0, title=None, size=(640, 360), name="s"):
    m = face_art[0]
    pl = dict(image=m["image"], backdrop=m["backdrop"], depth=m["depth"], sway=m["sway"], rim=m["rim"], subject=m["subject"],
              palette=m["palette"], ground_y=m["ground_y"], parallax=.3, sequence=None, eyes=eye_info, matte_method="none")
    sh = dict(idx=1, id=1, scene="s", type="closeup", dur=frames / 24, frames=frames, fps=24, step=1, size=list(size), seed=5, wind=[.0, .0],
              plate=pl, framing={"zoom": zoom, "cx": .5, "cy": .53},
              camera=dict(move=move, strength=0.0, dof=False, shake=0.0, hit_at=None, focal=50, rack=None),
              fx=[], eye_plan=plan, paths=dict(frames_dir=str(tmp / name)))
    if title:
        sh["title3d"] = title
    return sh


def _plan(**kw):
    p = eye_anim.plan(1.0, seed=3)
    p.update(blinks=[], gaze=[[0.0, 0.0, 0.0]], sparkles=[], droop=0.0, pupil=1.0)
    p.update(kw)
    return p


def _frames(out, n):
    return [cv2.imread(str(out / f"f_{i:04d}.png")) for i in range(1, n + 1)]


def _to_out(u, v, *, cx=.5, cy=.53, zoom=1.0, size=(640, 360)):
    """Normalised plate point -> output pixel for a static camera (frame covers PLATE_MARGIN/zoom of the plate; both are 16:9)."""
    fw = 0.9 / zoom
    return int(((u - cx) / fw + 0.5) * size[0]), int(((v - cy) / fw + 0.5) * size[1])


def test_blink_is_visible_only_in_the_eye_region(face_art, eye_info, tmp_path):
    plan = _plan(blinks=[[0.2, 0.3]])                                         # fully closed at ~0.31-0.35 s -> drawing 9
    out = cv_engine.render_shot(_shot(face_art, eye_info, plan, tmp_path), {"resume": False})
    rep = json.loads((out / "done.json").read_text())["features"]
    assert rep["eye_blink"] == "ok" and rep["eye_gaze"] == "ok" and rep["eye_pupil"] == "ok" and rep["eye_catchlight"] == "ok"
    fr = _frames(out, 16)
    d = np.abs(fr[8].astype(int) - fr[0].astype(int)).sum(-1)
    assert d.mean() < 12                                                       # only small parts of the image change
    ys, xs = np.nonzero(d > 60)
    assert len(xs) > 300                                                       # ... but the eyes really close
    top, bottom = d[: int(d.shape[0] * 0.25)].mean(), d[int(d.shape[0] * 0.35): int(d.shape[0] * 0.65)].mean()
    assert bottom > 5 * top + 1
    assert np.abs(fr[15].astype(int) - fr[0].astype(int)).mean() < 0.5         # the blink is over: eyes open again


def test_glow_keeps_the_iris_hue_and_brightens_it(face_art, eye_info, tmp_path):
    plan = _plan(mode="glow", glow={"on": True, "at": 0.0, "ramp": 0.3})
    out = cv_engine.render_shot(_shot(face_art, eye_info, plan, tmp_path, name="g"), {"resume": False})
    fr = _frames(out, 16)
    m, d, img, eboxes, fbox, (W, H) = face_art
    e = eye_info["eyes"][0]
    cx, cy, rx, ry = e["iris"]
    x, y = _to_out(cx / W, cy / H)
    patch0 = fr[0][y - 3:y + 4, x - 12:x - 6].reshape(-1, 3).mean(0)          # iris ring (not the pupil), BGR
    patch1 = fr[12][y - 3:y + 4, x - 12:x - 6].reshape(-1, 3).mean(0)
    assert patch1.sum() > patch0.sum() + 15                                    # brighter
    assert patch1[0] > patch1[2] + 40                                          # still BLUE (BGR: B > R) - regression: RGB/BGR swap


def test_gaze_moves_the_iris_but_not_the_lids(face_art, eye_info, tmp_path):
    still = _plan()
    look = _plan(gaze=[[0.0, 0.0, 0.0], [0.05, 1.0, 0.0]])
    a = cv_engine.render_shot(_shot(face_art, eye_info, still, tmp_path, name="a", frames=6), {"resume": False})
    b = cv_engine.render_shot(_shot(face_art, eye_info, look, tmp_path, name="b", frames=6), {"resume": False})
    fa, fb = _frames(a, 6)[5], _frames(b, 6)[5]
    d = np.abs(fa.astype(int) - fb.astype(int)).sum(-1)
    assert d.max() > 100 and d.mean() < 8
    assert np.abs(_frames(a, 6)[0].astype(int) - _frames(b, 6)[0].astype(int)).mean() < 0.3      # same before the saccade


def test_engine_without_eye_data_reports_nothing_and_still_renders(face_art, tmp_path):
    sh = _shot(face_art, None, None, tmp_path, name="n", frames=4)
    sh["plate"].pop("eyes")
    sh.pop("eye_plan")
    out = cv_engine.render_shot(sh, {"resume": False})
    rep = json.loads((out / "done.json").read_text())["features"]
    assert not any(k.startswith("eye_") for k in rep) and len(list(out.glob("f_*.png"))) == 4


# ---------------------------------------------------------------- title cards
@pytest.fixture(scope="module")
def title_layers(tmp_path_factory):
    d = tmp_path_factory.mktemp("title")
    t = titles.render_title("鬼滅", "سيف القمر", "series", d, "t", label="الحلقة الأولى")
    if not t:
        pytest.skip("no usable title font in this environment")
    return t


def test_title_layers_are_transparent_pngs_with_content(title_layers):
    assert title_layers["size"] == [1920, 1080] and title_layers["kind"] == "series"
    for k in ("face", "extrude", "glow"):
        im = cv2.imread(title_layers[k], cv2.IMREAD_UNCHANGED)
        assert im.shape == (1080, 1920, 4)
        cover = (im[..., 3] > 25).mean()
        assert 0.005 < cover < 0.6, (k, cover)
    face = cv2.imread(title_layers["face"], cv2.IMREAD_UNCHANGED)
    assert (face[..., 3] == 0).mean() > 0.7                                      # mostly transparent canvas
    y0, y1 = title_layers["bbox"][1], title_layers["bbox"][3]
    assert 0.2 < y0 < y1 < 0.9


def test_title_is_drawn_only_inside_its_window(face_art, title_layers, tmp_path):
    t3 = {"layers": {k: title_layers[k] for k in ("face", "extrude", "glow")}, "at": 0.3, "dur": 0.9}      # visible 0.3 s .. 1.2 s
    with_t = cv_engine.render_shot(_shot(face_art, None, None, tmp_path, name="t1", frames=36, title=t3), {"resume": False})
    no_t = cv_engine.render_shot(_shot(face_art, None, None, tmp_path, name="t0", frames=36), {"resume": False})
    assert json.loads((with_t / "done.json").read_text())["features"]["title3d"] == "ok"
    a, b = _frames(with_t, 36), _frames(no_t, 36)
    assert np.abs(a[0].astype(int) - b[0].astype(int)).mean() < 0.05            # before the window: identical
    assert np.abs(a[17].astype(int) - b[17].astype(int)).mean() > 2.0           # mid-window: title present
    assert np.abs(a[35].astype(int) - b[35].astype(int)).mean() < 0.05          # after the window: gone
    mid = np.abs(a[17].astype(int) - b[17].astype(int)).sum(-1)
    assert mid[int(360 * 0.3): int(360 * 0.7)].mean() > 5 * mid[: int(360 * 0.15)].mean() + 0.5  # in the title band, not at the top


def test_title_without_fonts_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(titles.subs, "find_cjk_font", lambda *a, **k: (None, None))
    monkeypatch.setattr(titles.subs, "find_arabic_font", lambda *a, **k: (None, None))
    assert titles.render_title("鬼滅", "سيف القمر", "series", tmp_path, "x") is None
    assert titles.render_title("", "", "episode", tmp_path, "y") is None


# ---------------------------------------------------------------- baked patches (what the Blender scene gets)
def test_bake_patches_blink_and_dedupe(face_art, eye_info, tmp_path):
    from mangaforge.render import eye_rig
    plan = _plan(blinks=[[0.2, 0.3]])
    sh = _shot(face_art, eye_info, plan, tmp_path, name="bk", frames=16)
    patches = eye_rig.bake_patches(sh, tmp_path / "eyes")
    assert len(patches) == 2 and {p["side"] for p in patches} == {"left", "right"}
    m, d, img, eboxes, fbox, (W, H) = face_art
    for p in patches:
        assert len(p["files"]) == 16
        x0, y0, x1, y1 = p["box"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert len(set(p["files"])) < 16                                    # identical open-eye drawings share a file
        closed = cv2.imread(p["files"][8], cv2.IMREAD_UNCHANGED)           # fully closed around 0.31-0.35 s
        opened = cv2.imread(p["files"][0], cv2.IMREAD_UNCHANGED)
        assert closed.shape[2] == 4 and closed.shape == opened.shape
        n_closed, n_open = (closed[..., 3] > 200).sum(), (opened[..., 3] > 200).sum()
        assert n_closed > 300 and n_closed > 3 * n_open                     # the lid appears (the open eye only carries its catch-light)
        assert (opened[..., 3] > 0).mean() < 0.4                            # everything else stays transparent
        last = cv2.imread(p["files"][15], cv2.IMREAD_UNCHANGED)
        assert np.abs(last.astype(int) - opened.astype(int)).mean() < 3     # blink over -> (almost) the picture from before it
        assert p["report"]["eye_blink"] == "ok"
    e = eye_info["eyes"][0]
    left = next(p for p in patches if p["side"] == e["side"])
    px0, py0 = int(left["box"][0] * W), int(left["box"][1] * H)
    closed = cv2.imread(left["files"][8], cv2.IMREAD_UNCHANGED)
    cx, cy = int(e["iris"][0]) - px0, int(e["iris"][1]) - py0
    assert closed[cy, cx, 3] > 230 and np.abs(closed[cy, cx, :3].astype(int) - np.array([246, 206, 180])).max() < 40


def test_bake_patches_skips_when_there_is_nothing_to_animate(face_art, eye_info, tmp_path):
    from mangaforge.render import eye_rig
    sh = _shot(face_art, eye_info, _plan(blinks=[[0.2, 0.3]]), tmp_path, name="sq", frames=4)
    sh["plate"]["sequence"] = {"dir": "/x", "pattern": "f_%04d.png", "frames": 4, "fps": 24}      # AI-animated plate: eyes already move
    assert eye_rig.bake_patches(sh, tmp_path / "e1") == []
    sh2 = _shot(face_art, None, None, tmp_path, name="ne", frames=4)
    sh2["plate"].pop("eyes")
    sh2.pop("eye_plan")
    assert eye_rig.bake_patches(sh2, tmp_path / "e2") == []


def test_bake_patches_tears_extend_the_box_downwards(face_art, eye_info, tmp_path):
    from mangaforge.render import eye_rig
    base = _shot(face_art, eye_info, _plan(), tmp_path, name="t0", frames=12)
    cry_plan = eye_anim.plan(1.0, seed=3, mode="tears")
    cry_plan.update(blinks=[], gaze=[[0.0, 0.0, 0.0]], sparkles=[], droop=0.0, pupil=1.0)
    assert cry_plan["tears"]["on"]
    cry = _shot(face_art, eye_info, cry_plan, tmp_path, name="t1", frames=12)
    a = eye_rig.bake_patches(base, tmp_path / "ea")
    b = eye_rig.bake_patches(cry, tmp_path / "eb")
    assert a and b and all(len(p["files"]) == 12 for p in b)
    assert (b[0]["box"][3] - b[0]["box"][1]) > (a[0]["box"][3] - a[0]["box"][1]) * 1.5      # room for the running streak
