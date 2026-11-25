"""Runs INSIDE Blender:   blender -b --factory-startup -noaudio -P blender_scene.py -- --job job.json

Builds one shot from the shot dict and renders its drawings ("on twos": every `step`-th frame) to f_0001.png ...

  * 2.5D plate: a depth mesh (faces never bridge a depth jump) with an emission-only, unlit material, so the artwork keeps its
    exact colours (Standard view transform) while the camera produces real parallax. Camera moves come from camera_paths.py,
    DoF/rack focus, motion blur, handheld/impact shake. A blurred backdrop plane means no edge is ever visible.
  * shader-driven wind/hair sway, heat haze, water ripple and rim light on the plate material (values updated per frame)
  * billboard "quad clouds" for particles and light effects (blender_fx.py), 3D dressing/toon/physics (blender_extras.py)
  * Every feature runs inside a guard and reports ok | off | skipped:<why> | failed:<error> into done.json.
Written against the Blender 4.2 LTS .. 5.2 LTS Python API; version-sensitive spots are introspected, never assumed.
"""
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("MANGAFORGE_ROOT", str(HERE.parent.parent)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load("mf_camera_paths", ROOT / "mangaforge" / "camera_paths.py")
pg = _load("mf_plate_geometry", HERE / "plate_geometry.py")

FEATURES, WARNINGS = {}, []


def say(*a):
    print("[mangaforge]", *a, flush=True)


class feat:
    """with feat("name"): ...   -> exceptions are reported as failed:<error> and never kill the shot."""

    def __init__(self, name, enabled=True, why="off"):
        self.name, self.enabled, self.why = name, enabled, why

    def __enter__(self):
        if not self.enabled:
            FEATURES.setdefault(self.name, self.why)
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            FEATURES[self.name] = f"failed:{et.__name__}: {str(ev)[:160]}"
            say("FEATURE FAILED", self.name, "".join(traceback.format_exception(et, ev, tb))[-900:])
            return True
        if self.enabled:
            FEATURES.setdefault(self.name, "ok")
        return False


def mark(name, status="ok"):
    FEATURES[name] = status


# ------------------------------------------------------------------ job
def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    job = json.loads(pathlib.Path(argv[argv.index("--job") + 1]).read_text("utf-8"))
    return job["shot"], job["settings"]


SHOT, SET = parse()
SAFE = bool(SET.get("safe_mode"))
FPS, STEP, FRAMES = int(SHOT["fps"]), int(SHOT["step"]), int(SHOT["frames"])
RW, RH = [int(x) for x in SHOT["size"]]
FDIR = pathlib.Path(SHOT["paths"]["frames_dir"])
PLATE = SHOT["plate"]
FR = SHOT["framing"]
CAM = SHOT["camera"]
SEED = int(SHOT.get("seed", 1))
FOCAL = float(CAM.get("focal", 35))
D = pg.D_REF
WF0 = pg.frame_width_at_ref(FOCAL)
HF0 = WF0 / pg.ASPECT
FW = cp.PLATE_MARGIN / max(1.0, FR["zoom"])
CX, CY = FR["cx"], FR["cy"]
SUB = PLATE.get("subject") or {"x": 0.5, "y": 0.5, "depth": 0.8}
PAR = float(PLATE.get("parallax", 0.35))
HIT = CAM.get("hit_at")
HIT_FRAME = None if HIT is None else 1 + int(round(HIT * FPS))
FXS = {f["kind"]: f for f in (SHOT.get("fx") or [])}
PAL = PLATE.get("palette") or {}
rng = np.random.default_rng(SEED)


def s2w(u, v, z):
    """screen-like (u frame widths right, v frame heights down, relative depth z) -> world, relative to the rest camera."""
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    return np.stack([u * WF0 * z, z * D, -v * HF0 * z], -1)


def lin(c, default=(0.3, 0.3, 0.3)):
    c = PAL.get(c) if isinstance(c, str) else c
    c = tuple(c) if c else default
    return (float(c[0]), float(c[1]), float(c[2]))


# ------------------------------------------------------------------ scene setup
def clean():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.curves, bpy.data.worlds):
        for b in list(coll):
            coll.remove(b)


def engine_id(name):
    ids = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    if name == "CYCLES":
        return "CYCLES" if "CYCLES" in ids else None
    for c in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):                 # 4.2-4.5 vs 5.x naming
        if c in ids:
            return c
    return None


def setup_render(scene):
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = RW, RH, 100
    r.fps, r.fps_base = FPS, 1.0
    scene.frame_start, scene.frame_end, scene.frame_current = 1, FRAMES, 1
    want = (SET.get("engine") or "auto").lower()
    use = "EEVEE" if want == "eevee" else "CYCLES"
    eng = engine_id(use) or engine_id("CYCLES")
    r.engine = eng
    mark("engine_" + ("cycles" if eng == "CYCLES" else "eevee"))
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.image_settings.compression = 12
    r.film_transparent = False
    r.use_persistent_data = True
    thr = int(SET.get("threads") or 0)
    if thr:
        r.threads_mode = "FIXED"
        r.threads = thr
    with feat("color_management"):
        vt = SET.get("view_transform") or "Standard"
        names = [e.identifier for e in scene.view_settings.bl_rna.properties["view_transform"].enum_items]
        scene.view_settings.view_transform = vt if vt in names else "Standard"
        try:
            scene.view_settings.look = "None"
        except TypeError:
            pass
        scene.display_settings.display_device = "sRGB"
    with feat("motion_blur", bool(SET.get("motion_blur")), "off"):
        r.use_motion_blur = True
        r.motion_blur_shutter = 0.5
    if eng == "CYCLES":
        c = scene.cycles
        c.samples = int(SET.get("samples") or 24)
        c.use_adaptive_sampling = True
        c.adaptive_threshold = 0.02
        c.max_bounces, c.diffuse_bounces, c.glossy_bounces, c.transmission_bounces, c.volume_bounces = 4, 2, 2, 4, 1
        with feat("denoise", bool(SET.get("denoise")), "off"):
            c.use_denoising = True
            try:
                c.denoiser = "OPENIMAGEDENOISE"
            except TypeError:
                pass
        dev = os.environ.get("CYCLES_DEVICE", "CPU").upper()
        c.device = "CPU"
        if dev == "GPU":
            with feat("cycles_gpu"):
                prefs = bpy.context.preferences.addons["cycles"].preferences
                for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        prefs.compute_device_type = kind
                        prefs.get_devices()
                        devs = [d for d in prefs.devices if d.type != "CPU"]
                        if devs:
                            for d in prefs.devices:
                                d.use = d.type != "CPU"
                            c.device = "GPU"
                            break
                    except Exception:  # noqa: BLE001
                        continue
                else:
                    raise RuntimeError("no GPU device found")
    else:
        e = scene.eevee
        for attr, val in (("taa_render_samples", int(SET.get("samples") or 24)), ("use_motion_blur_next", False)):
            try:
                setattr(e, attr, val)
            except Exception:  # noqa: BLE001
                pass


def make_world(scene):
    w = bpy.data.worlds.new("W")
    scene.world = w
    try:
        w.use_nodes = True
    except Exception:  # noqa: BLE001
        pass
    avg = lin("avg")
    try:
        bg = w.node_tree.nodes.get("Background") or w.node_tree.nodes.new("ShaderNodeBackground")
        bg.inputs[0].default_value = (avg[0] * 0.5, avg[1] * 0.5, avg[2] * 0.5, 1.0)
        bg.inputs[1].default_value = 0.6
    except Exception:  # noqa: BLE001
        w.color = avg


# ------------------------------------------------------------------ materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:  # noqa: BLE001
        pass
    m.node_tree.nodes.clear()
    return m


def load_image(path, noncolor=False):
    im = bpy.data.images.load(str(path), check_existing=True)
    im.colorspace_settings.name = "Non-Color" if noncolor else "sRGB"
    return im


def read_gray(path):
    """-> float32 HxW array, row 0 = top of the image (as in the file)."""
    im = bpy.data.images.load(str(path), check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    px = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(px)
    arr = px.reshape(h, w, 4)[::-1, :, 0].copy()
    bpy.data.images.remove(im)
    return arr


class PlateMat:
    """Emission-only plate material. Dynamic inputs (time, wind gust, pulse) are plain Value nodes updated per frame."""

    def __init__(self, image_path, name="plate", sway=None, rim=None, seq=None):
        self.m = new_mat(name)
        n, l = self.m.node_tree.nodes, self.m.node_tree.links
        self.n, self.l = n, l
        self.vals = {}
        uv = n.new("ShaderNodeTexCoord")
        vec = uv.outputs["UV"]
        want_wind = any(k in FXS for k in ("wind_sway", "hair_sway", "grass", "cloth_flag")) and sway and not SAFE
        want_haze = "heat_haze" in FXS and not SAFE
        want_water = "water_ripple" in FXS and not SAFE
        self.swaymask = None
        if want_wind or want_haze or want_water:
            vec = self._offsets(uv, vec, sway, want_wind, want_haze, want_water)
        tex = n.new("ShaderNodeTexImage")
        tex.image = load_image(image_path)
        tex.interpolation = "Cubic"
        tex.extension = "EXTEND"
        l.new(vec, tex.inputs["Vector"])
        self.tex = tex
        if seq:
            with feat("i2v_plate"):
                tex.image.source = "SEQUENCE"
                tex.image_user.frame_start = 1
                tex.image_user.frame_duration = int(seq["frames"])
                tex.image_user.frame_offset = 0
                tex.image_user.use_auto_refresh = True
                tex.image_user.use_cyclic = False
        # camera rays see exactly the artwork; GI rays see a tunable fraction so the plate lights 3D props without washing them
        lp = n.new("ShaderNodeLightPath")
        em_cam = n.new("ShaderNodeEmission")
        em_gi = n.new("ShaderNodeEmission")
        l.new(tex.outputs["Color"], em_cam.inputs["Color"])
        l.new(tex.outputs["Color"], em_gi.inputs["Color"])
        em_cam.inputs["Strength"].default_value = 1.0
        em_gi.inputs["Strength"].default_value = 0.55
        mix = n.new("ShaderNodeMixShader")
        l.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
        l.new(em_gi.outputs["Emission"], mix.inputs[1])
        l.new(em_cam.outputs["Emission"], mix.inputs[2])
        out_shader = mix.outputs["Shader"]
        mark("plate_unlit_emission")
        mark("gi_camera_ray_trick")
        rim_scale = {"rembg": 1.0, "grabcut": 0.3}.get(PLATE.get("matte_method", "rembg"), 0.0)
        if rim and rim_scale > 0 and not SAFE:
            try:
                rt = n.new("ShaderNodeTexImage")
                rt.image = load_image(rim, noncolor=True)
                rt.extension = "EXTEND"
                l.new(uv.outputs["UV"], rt.inputs["Vector"])
                self.rim_strength = self._value("rim")
                mul = n.new("ShaderNodeMath")
                mul.operation = "MULTIPLY"
                l.new(rt.outputs["Color"], mul.inputs[0])
                l.new(self.rim_strength.outputs[0], mul.inputs[1])
                em = n.new("ShaderNodeEmission")
                top = np.array(lin("top", (0.8, 0.85, 1.0)))
                col = np.clip(np.clip(top, 0, 1) ** 0.5 / (top.max() ** 0.5 + 1e-6) * 0.65 + 0.35, 0, 1)
                em.inputs["Color"].default_value = (col[0], col[1], col[2], 1)
                l.new(mul.outputs[0], em.inputs["Strength"])
                add = n.new("ShaderNodeAddShader")
                l.new(out_shader, add.inputs[0])
                l.new(em.outputs["Emission"], add.inputs[1])
                out_shader = add.outputs["Shader"]
                self.rim_rate = rim_scale
                mark("rim_glow")
            except Exception as e:  # noqa: BLE001
                mark("rim_glow", f"failed:{e}")
        else:
            self.rim_strength = None
            mark("rim_glow", "off" if not rim else "skipped:matte not accurate (install rembg)")
        out = n.new("ShaderNodeOutputMaterial")
        l.new(out_shader, out.inputs["Surface"])

    def _value(self, key):
        v = self.n.new("ShaderNodeValue")
        self.vals[key] = v
        return v

    def _noise(self, vec_in, scale):
        nz = self.n.new("ShaderNodeTexNoise")
        try:
            nz.noise_dimensions = "3D"
        except Exception:  # noqa: BLE001
            pass
        nz.inputs["Scale"].default_value = scale
        try:
            nz.inputs["Detail"].default_value = 1.0
        except Exception:  # noqa: BLE001
            pass
        self.l.new(vec_in, nz.inputs["Vector"])
        return nz

    def _math(self, op, a, b):
        m = self.n.new("ShaderNodeMath")
        m.operation = op
        for i, x in enumerate((a, b)):
            if hasattr(x, "node") or hasattr(x, "links"):
                self.l.new(x, m.inputs[i])
            else:
                m.inputs[i].default_value = float(x)
        return m.outputs[0]

    def _offsets(self, uv, vec, sway, want_wind, want_haze, want_water):
        n, l = self.n, self.l
        sep = n.new("ShaderNodeSeparateXYZ")
        l.new(vec, sep.inputs[0])
        t = self._value("time").outputs[0]
        gust = self._value("gust").outputs[0]
        wx = float((SHOT.get("wind") or [0.5, 0.05])[0])
        offx = offy = 0.0
        if want_wind:
            mask_tex = n.new("ShaderNodeTexImage")
            mask_tex.image = load_image(sway, noncolor=True)
            mask_tex.extension = "EXTEND"
            l.new(vec, mask_tex.inputs["Vector"])
            self.swaymask = mask_tex.outputs["Color"]
            comb = n.new("ShaderNodeCombineXYZ")
            l.new(self._math("SUBTRACT", self._math("MULTIPLY", sep.outputs["X"], 7.0), self._math("MULTIPLY", t, 0.9 * wx)), comb.inputs["X"])
            l.new(self._math("ADD", self._math("MULTIPLY", sep.outputs["Y"], 7.0), self._math("MULTIPLY", t, 0.3)), comb.inputs["Y"])
            l.new(self._math("MULTIPLY", t, 0.4), comb.inputs["Z"])
            nz = self._noise(comb.outputs[0], 1.0)
            sep_n = n.new("ShaderNodeSeparateXYZ")
            l.new(nz.outputs["Color"], sep_n.inputs[0])
            amp = self._math("MULTIPLY", self._math("MULTIPLY", self.swaymask, gust), 0.0085)
            offx = self._math("MULTIPLY", self._math("SUBTRACT", sep_n.outputs["X"], 0.5), amp)
            offy = self._math("MULTIPLY", self._math("SUBTRACT", sep_n.outputs["Y"], 0.5), self._math("MULTIPLY", amp, 0.55))
            for k in ("wind_sway", "hair_sway", "grass", "cloth_flag"):
                if k in FXS:
                    mark(k)
        if want_haze:
            c2 = n.new("ShaderNodeCombineXYZ")
            l.new(self._math("MULTIPLY", sep.outputs["X"], 1.0), c2.inputs["X"])
            l.new(self._math("SUBTRACT", self._math("MULTIPLY", sep.outputs["Y"], 1.0), self._math("MULTIPLY", t, 0.06)), c2.inputs["Y"])
            l.new(self._math("MULTIPLY", t, 0.5), c2.inputs["Z"])
            nz2 = self._noise(c2.outputs[0], 38.0)
            sp = n.new("ShaderNodeSeparateXYZ")
            l.new(nz2.outputs["Color"], sp.inputs[0])
            offx = self._math("ADD", offx, self._math("MULTIPLY", self._math("SUBTRACT", sp.outputs["X"], 0.5), 0.0035))
            offy = self._math("ADD", offy, self._math("MULTIPLY", self._math("SUBTRACT", sp.outputs["Y"], 0.5), 0.0022))
            mark("heat_haze")
        if want_water:
            band = self._math("MULTIPLY", self._math("SUBTRACT", 0.45, sep.outputs["Y"]), 4.0)             # lower part of the image
            band = self._math("MAXIMUM", band, 0.0)
            ph = self._math("ADD", self._math("MULTIPLY", sep.outputs["X"], 90.0), self._math("MULTIPLY", t, 3.0))
            ripple = self._math("MULTIPLY", self._math("SINE", ph, 0.0), self._math("MULTIPLY", self._math("MINIMUM", band, 1.0), 0.0014))
            offx = self._math("ADD", offx, ripple)
            mark("water_ripple")
        comb2 = n.new("ShaderNodeCombineXYZ")
        l.new(self._math("ADD", sep.outputs["X"], offx), comb2.inputs["X"])
        l.new(self._math("ADD", sep.outputs["Y"], offy), comb2.inputs["Y"])
        return comb2.outputs[0]

    def update(self, t, f):
        if "time" in self.vals:
            self.vals["time"].outputs[0].default_value = t
        if "gust" in self.vals:
            self.vals["gust"].outputs[0].default_value = 0.55 + 0.45 * math.sin(2 * math.pi * 0.33 * t + (SEED % 7))
        if "rim" in self.vals and self.rim_strength is not None:
            pulse = 0.5 + 0.12 * math.sin(2 * math.pi * 0.5 * t) + (0.9 if "aura" in FXS else 0.0)
            pulse += 0.5 * (1.0 if "glow_pulse" in FXS and math.sin(2 * math.pi * 0.9 * t) > 0 else 0.0)
            self.rim_strength.outputs[0].default_value = pulse * self.rim_rate * 1.4


def build_mesh(name, verts, faces, uvs):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    uvl = me.uv_layers.new(name="UV")
    li = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", li)
    uvl.data.foreach_set("uv", np.asarray(uvs, np.float32)[li].ravel())
    me.update()
    return me


MG = None


def zeta_at(X, Y):
    """Relative camera distance of the plate surface at plate coords (X right, Y down, both 0..1).
    Bilinear on the depth-mesh grid, so anything placed with it sits exactly on (or a hair in front of) the plate mesh."""
    Zg = MG["zeta"]
    ny, nx = Zg.shape[0] - 1, Zg.shape[1] - 1
    x = np.clip(np.asarray(X, np.float64), 0.0, 1.0) * nx
    y = np.clip(np.asarray(Y, np.float64), 0.0, 1.0) * ny
    i0 = np.clip(np.floor(x).astype(int), 0, nx - 1)
    j0 = np.clip(np.floor(y).astype(int), 0, ny - 1)
    fx, fy = x - i0, y - j0
    return (Zg[j0, i0] * (1 - fx) * (1 - fy) + Zg[j0, i0 + 1] * fx * (1 - fy)
            + Zg[j0 + 1, i0] * (1 - fx) * fy + Zg[j0 + 1, i0 + 1] * fx * fy)


def build_plate(scene):
    global MG
    depth = read_gray(PLATE["depth"])
    d8 = (np.clip(depth, 0, 1) * 255).round().astype(np.uint8)
    mg = pg.mesh_grid(d8, SUB["depth"], PAR, nx=192 if not SAFE else 96, ny=108 if not SAFE else 54)
    MG = mg
    X, Y, Z = mg["x"], mg["y"], mg["zeta"]
    u0, v0 = (X - CX) / FW, (Y - CY) / FW
    verts = np.stack([u0 * WF0 * Z, Z * D, -v0 * HF0 * Z], -1).reshape(-1, 3)
    uvs = np.stack([X, 1 - Y], -1).reshape(-1, 2)
    me = build_mesh("plate", verts, mg["faces"], uvs)
    ob = bpy.data.objects.new("plate", me)
    scene.collection.objects.link(ob)
    pm = PlateMat(PLATE["image"], "plate", sway=PLATE.get("sway"), rim=PLATE.get("rim"), seq=PLATE.get("sequence"))
    me.materials.append(pm.m)
    mark("plate_depth_mesh")
    mark("depth_jump_face_drop")
    # blurred backdrop: same projective mapping, flat at the far side, covers 1.5x the plate
    zb = 1.0 + PAR * 1.25
    bs = float(PLATE.get("backdrop_scale", 1.5))
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    bv, buv = [], []
    for (xb, yb) in corners:
        X_, Y_ = 0.5 + (xb - 0.5) * bs, 0.5 + (yb - 0.5) * bs
        bv.append(((X_ - CX) / FW * WF0 * zb, zb * D, -((Y_ - CY) / FW) * HF0 * zb))
        buv.append((xb, 1 - yb))
    bme = build_mesh("backdrop", bv, [(0, 1, 2, 3)], buv)
    bob = bpy.data.objects.new("backdrop", bme)
    scene.collection.objects.link(bob)
    bm = new_mat("backdrop")
    n, l = bm.node_tree.nodes, bm.node_tree.links
    t = n.new("ShaderNodeTexImage")
    t.image = load_image(PLATE["backdrop"])
    t.extension = "EXTEND"
    em = n.new("ShaderNodeEmission")
    out = n.new("ShaderNodeOutputMaterial")
    uv = n.new("ShaderNodeTexCoord")
    l.new(uv.outputs["UV"], t.inputs["Vector"])
    l.new(t.outputs["Color"], em.inputs["Color"])
    l.new(em.outputs["Emission"], out.inputs["Surface"])
    bme.materials.append(bm)
    mark("backdrop_plane")
    return ob, pm


# ------------------------------------------------------------------ camera
def subject_world():
    return Vector(pg.subject_world(((SUB["x"] - CX) / FW, (SUB["y"] - CY) / FW), FOCAL).tolist())


S_W = None
_CAM_CACHE = {}


def cam_matrix(f):
    """4x4 world matrix of the camera at frame f. The pose -> camera maths lives in plate_geometry.camera_pose_world, the one
    model the cv engine ray-casts through as well, so both engines put every pixel in the same place."""
    if f in _CAM_CACHE:
        return _CAM_CACHE[f]
    tn = (f - 1) / max(1, FRAMES - 1)
    p = cp.total_pose(CAM["move"], tn, CAM["strength"], SEED, f, FPS, CAM.get("shake", 0.0), HIT_FRAME)
    R, C = pg.camera_pose_world(p, np.array(S_W, dtype=np.float64), FOCAL)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R, C
    M = Matrix(A.tolist())
    _CAM_CACHE[f] = (M, p)
    return _CAM_CACHE[f]


def build_camera(scene):
    global S_W
    S_W = subject_world()
    cd = bpy.data.cameras.new("cam")
    cd.sensor_width, cd.sensor_fit = 36.0, "HORIZONTAL"
    cd.lens, cd.clip_start, cd.clip_end = FOCAL, 0.05, 2000.0
    co = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(co)
    scene.camera = co
    dof_on = bool(CAM.get("dof")) and not SAFE
    rack = CAM.get("rack")
    if dof_on:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = {"closeup": 0.45, "reaction": 0.5, "insert": 0.55, "medium": 0.8}.get(SHOT.get("type"), 1.2)
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:  # noqa: BLE001
        pass
    prev = None
    zooming = any(abs(cam_matrix(f)[1]["focal_scale"] - 1.0) > 1e-4 for f in range(1, FRAMES + 1))
    for f in range(1, FRAMES + 1):
        M, p = cam_matrix(f)
        co.location = M.to_translation()
        co.rotation_euler = M.to_euler("XYZ", prev) if prev is not None else M.to_euler("XYZ")
        prev = co.rotation_euler.copy()
        co.keyframe_insert("location", frame=f)
        co.keyframe_insert("rotation_euler", frame=f)
        cd.lens = FOCAL * p["focal_scale"]
        if zooming:                                                   # a key on every frame: no stray extrapolation at the ends
            cd.keyframe_insert("lens", frame=f)
        if dof_on:
            tn = (f - 1) / max(1, FRAMES - 1)
            fd = (rack["from"] + (rack["to"] - rack["from"]) * cp.ease_in_out(tn)) if rack else SUB["depth"]
            zf = 1.0 - PAR * (fd - SUB["depth"])
            cd.dof.focus_distance = max(0.3, zf * D - p["dz"] * D)
            cd.dof.keyframe_insert("focus_distance", frame=f)
    mark("camera_moves")
    mark("camera_shake" if (CAM.get("shake", 0) > 0 or CAM["move"] in ("handheld", "punch_in") or HIT is not None) else "camera_shake", "ok")
    mark("dof" if dof_on else "dof", "ok" if dof_on else "off")
    mark("rack_focus", "ok" if (dof_on and rack) else "off")
    mark("dolly_zoom", "ok" if CAM["move"] == "dolly_zoom" else "off")
    return co


def build_lights(scene):
    sun = bpy.data.lights.new("key", "SUN")
    sun.energy = 2.2
    sun.color = lin("top", (1, 0.95, 0.9))
    try:
        sun.angle = math.radians(6)
    except Exception:  # noqa: BLE001
        pass
    so = bpy.data.objects.new("key", sun)
    scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(55), 0, math.radians(35))
    mark("lighting_key_sun")


def png_complete(path):
    """A drawing counts as done only if its PNG is fully written (IEND chunk): a killed job can leave a truncated file."""
    try:
        with open(path, "rb") as fh:
            fh.seek(-12, 2)
            return fh.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
    except OSError:
        return False


def enable_addons(names):
    """BLENDER_ADDONS=a,b : extra add-ons (Node Wrangler, Extra Objects, your own ...) enabled before the scene is built.
    A missing add-on is reported in done.json (addon_<name> = failed:...) and never stops the shot."""
    for name in names:
        with feat("addon_" + str(name)):
            bpy.ops.preferences.addon_enable(module=str(name))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    FDIR.mkdir(parents=True, exist_ok=True)
    clean()
    scene = bpy.context.scene
    setup_render(scene)
    enable_addons(SET.get("addons") or [])
    make_world(scene)
    plate_obj, pm = build_plate(scene)
    cam = build_camera(scene)
    build_lights(scene)

    ctx = {"scene": scene, "shot": SHOT, "S": SET, "feat": feat, "mark": mark, "s2w": s2w, "D": D, "WF0": WF0, "HF0": HF0, "FW": FW,
           "CX": CX, "CY": CY, "SUB": SUB, "PAR": PAR, "rng": rng, "FXS": FXS, "FPS": FPS, "FRAMES": FRAMES, "HIT": HIT, "cam": cam,
           "cam_matrix": cam_matrix, "plate_obj": plate_obj, "pal": lin, "new_mat": new_mat, "build_mesh": build_mesh,
           "pg": pg, "cp": cp, "read_gray": read_gray, "load_image": load_image, "plate": PLATE, "seed": SEED, "safe": SAFE,
           "say": say, "FDIR": FDIR, "FOCAL": FOCAL, "zeta_at": zeta_at, "MG": MG, "S_W": S_W, "pm": pm,
           "fx_disabled": set(), "RW": RW, "RH": RH}
    dyn = []
    if not SAFE:
        with feat("fx_engine"):
            fx = _load("mf_blender_fx", HERE / "blender_fx.py")
            dyn += fx.setup(ctx)
        with feat("extras_engine"):
            ex = _load("mf_blender_extras", HERE / "blender_extras.py")
            dyn += ex.setup(ctx)
    else:
        mark("fx_engine", "skipped:safe_mode")
        mark("extras_engine", "skipped:safe_mode")

    need_sim = bool(SET.get("physics")) and not SAFE
    drawings = int(math.ceil(FRAMES / STEP))
    done = 0
    for f in range(1, FRAMES + 1):
        is_drawing = (f - 1) % STEP == 0
        i = (f - 1) // STEP + 1
        out = FDIR / f"f_{i:04d}.png"
        if not is_drawing and not need_sim:
            continue
        if is_drawing and png_complete(out) and not need_sim:
            done += 1
            continue
        scene.frame_set(f)
        t = (f - 1) / FPS
        pm.update(t, f)
        for d in dyn:
            try:
                d(t, f)
            except Exception as e:  # noqa: BLE001
                key = "dyn_" + getattr(d, "__name__", "fx")
                if not str(FEATURES.get(key, "")).startswith("failed"):
                    say("dynamic update failed", key, e)
                    FEATURES[key] = f"failed:{e}"
        if is_drawing:
            scene.render.filepath = str(out)
            bpy.ops.render.render(write_still=True)
            done += 1
            if done % 6 == 0 or done == drawings:
                say(f"drawing {done}/{drawings} ({time.time() - t0:.0f}s)")
    if SET.get("save_blend"):
        with feat("save_blend"):
            bpy.ops.file.make_paths_relative()
            bpy.ops.wm.save_as_mainfile(filepath=str(FDIR / "shot.blend"), compress=True, relative_remap=True)
    if SET.get("export_usd"):
        with feat("export_usd"):
            bpy.ops.wm.usd_export(filepath=str(FDIR / "shot.usd"))
    rep = {"drawings": drawings, "engine": "blender", "blender": ".".join(map(str, bpy.app.version)), "warnings": WARNINGS,
           "features": FEATURES, "seconds": round(time.time() - t0, 1), "safe_mode": SAFE}
    (FDIR / "done.json").write_text(json.dumps(rep, indent=1), "utf-8")
    say(f"shot done: {drawings} drawings in {time.time() - t0:.0f}s; features ok={sum(v == 'ok' for v in FEATURES.values())} "
        f"failed={sum(str(v).startswith('failed') for v in FEATURES.values())}")


try:
    main()
except SystemExit:
    raise
except BaseException:  # noqa: BLE001
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)
