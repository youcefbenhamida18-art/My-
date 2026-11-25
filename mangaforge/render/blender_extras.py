"""Runs INSIDE Blender (loaded by blender_scene.py): everything that is not a billboard sprite.

    setup(ctx) -> [callable(t, f)]

  * 3D title card      face / extrusion / glow layers parented to the camera with a slight yaw/pitch (the extrusion shows depth),
                       scale-in, fade, moving light sweep.  Text is drawn with Pillow (Arabic + Japanese) - Blender text objects
                       cannot shape Arabic.
  * eye patches        per-drawing eye images baked by eye_rig.bake_patches (blink, gaze, pupil, catch-light, glow, tears - the same
                       rig the cv engine uses) shown on quads that CONFORM to the plate's depth surface, so they keep perfect
                       registration while the camera moves.
  * ground dressing    pebbles (toon-shaded low-poly rocks + contact shadows), grass blades swaying in the wind, fallen leaves:
                       real meshes with parallax, coloured from the plate palette, optional Freestyle ink lines.
  * curve strokes      slash / lightning / swirl as tapered glowing bevelled curves drawn on with bevel_factor.
  * compositor         bloom (Glare) on HDR emission, optional Z / mist EXR passes.
  * physics tier       (PHYSICS=1 / VOLUMETRICS=1) Mantaflow smoke / fire, Principled-Volume fog, cloth banner.
Each piece is guarded: a failure is reported in done.json and the shot still renders.
"""
import math
import pathlib
import zlib

import bpy
import numpy as np

TAU = 2.0 * math.pi
ASPECT = 16.0 / 9.0
SENSOR = 36.0


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def to_linear(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgba_lin(c, a=1.0):
    r = to_linear(c)
    return (float(r[0]), float(r[1]), float(r[2]), float(a))


def named(fn, name):
    def w(t, f):
        return fn(t, f)
    w.__name__ = name
    return w


def render_method_blended(mat):
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND"), ("show_transparent_back", False)):
        try:
            setattr(mat, attr, val)
        except Exception:  # noqa: BLE001
            pass


def no_gi(ob):
    for attr in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter", "visible_shadow"):
        try:
            setattr(ob, attr, False)
        except Exception:  # noqa: BLE001
            pass


class Cam:
    """Camera basis at frame f (same camera the scene script keyframes)."""
    def __init__(self, ctx, f):
        M, p = ctx["cam_matrix"](f)
        A = np.array(M, dtype=np.float64)
        self.R, self.pos = A[:3, :3], A[:3, 3]
        self.right, self.up, self.fwd = self.R[:, 0], self.R[:, 1], -self.R[:, 2]
        self.focal = float(ctx["FOCAL"]) * float(p["focal_scale"])

    def wl(self, L):
        return L * SENSOR / self.focal


def phys_to_world(ctx, x, y, z):
    """Physical frame coordinates (x right, y UP, both in frame widths; origin = frame centre) at relative depth z -> world."""
    return ctx["s2w"](np.asarray(x, float), -np.asarray(y, float) * ASPECT, np.asarray(z, float))


# ------------------------------------------------------------------ shared materials
def glow_material(ctx, name, color, strength, additive):
    """Emission with a 'fade' Value node (0..1) the caller drives per frame."""
    m = ctx["new_mat"](name)
    n, l = m.node_tree.nodes, m.node_tree.links
    em = n.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba_lin(color)
    fade = n.new("ShaderNodeValue")
    fade.outputs[0].default_value = 1.0
    tr = n.new("ShaderNodeBsdfTransparent")
    out = n.new("ShaderNodeOutputMaterial")
    if additive:
        mul = n.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        l.new(fade.outputs[0], mul.inputs[0])
        mul.inputs[1].default_value = float(strength)
        l.new(mul.outputs[0], em.inputs["Strength"])
        add = n.new("ShaderNodeAddShader")
        l.new(tr.outputs["BSDF"], add.inputs[0])
        l.new(em.outputs["Emission"], add.inputs[1])
        l.new(add.outputs["Shader"], out.inputs["Surface"])
    else:
        em.inputs["Strength"].default_value = float(strength)
        mix = n.new("ShaderNodeMixShader")
        l.new(fade.outputs[0], mix.inputs[0])
        l.new(tr.outputs["BSDF"], mix.inputs[1])
        l.new(em.outputs["Emission"], mix.inputs[2])
        l.new(mix.outputs["Shader"], out.inputs["Surface"])
    render_method_blended(m)
    return m, fade


def math_node(m, op, a, b=None, c=None):
    """Small helper for node graphs: a / b / c are sockets or floats; returns the output socket."""
    n, l = m.node_tree.nodes, m.node_tree.links
    node = n.new("ShaderNodeMath")
    node.operation = op
    for i, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float, np.floating)):
            node.inputs[i].default_value = float(x)
        else:
            l.new(x, node.inputs[i])
    return node.outputs[0]


# ================================================================== 3D title card
class TitleRig:
    LAYERS = (("glow", 0.30, True, 1.5), ("extrude", 0.14, False, 1.0), ("face", 0.0, False, 1.0))      # name, depth, additive, strength

    def __init__(self, ctx, t3):
        self.ctx = ctx
        lay = t3.get("layers") or {}
        self.at, self.dur = float(t3.get("at", 0.35)), float(t3.get("dur", 3.0))
        self.items = []
        for name, dz, additive, strength in self.LAYERS:
            path = lay.get(name)
            if not path or not pathlib.Path(path).exists():
                continue
            self.items.append(self._quad(name, path, dz, additive, strength))
        if not self.items:
            raise RuntimeError("title layers missing")
        ctx["mark"]("title_3d_layers", "ok")

    def _quad(self, name, path, dz, additive, strength):
        ctx = self.ctx
        me = bpy.data.meshes.new("title_" + name)
        me.from_pydata([(0.0, 0.0, 0.0)] * 4, [], [(0, 1, 2, 3)])
        uvl = me.uv_layers.new(name="UV")
        uvl.data.foreach_set("uv", np.array([0, 0, 1, 0, 1, 1, 0, 1], np.float32))
        ob = bpy.data.objects.new("title_" + name, me)
        ctx["scene"].collection.objects.link(ob)
        no_gi(ob)
        m = ctx["new_mat"]("title_" + name)
        n, l = m.node_tree.nodes, m.node_tree.links
        tex = n.new("ShaderNodeTexImage")
        tex.image = ctx["load_image"](path)
        tex.extension, tex.interpolation = "CLIP", "Linear"
        uv = n.new("ShaderNodeTexCoord")
        l.new(uv.outputs["UV"], tex.inputs["Vector"])
        fade = n.new("ShaderNodeValue")
        fade.outputs[0].default_value = 0.0
        sweep = n.new("ShaderNodeValue")
        sweep.outputs[0].default_value = -1.0
        col = tex.outputs["Color"]
        if name == "face":                                                   # moving diagonal highlight band across the lettering
            sep = n.new("ShaderNodeSeparateXYZ")
            l.new(uv.outputs["UV"], sep.inputs[0])
            diag = math_node(m, "ADD", sep.outputs["X"], math_node(m, "MULTIPLY", sep.outputs["Y"], 0.35))
            dist = math_node(m, "ABSOLUTE", math_node(m, "SUBTRACT", diag, sweep.outputs[0]))
            band = math_node(m, "MAXIMUM", math_node(m, "SUBTRACT", 1.0, math_node(m, "DIVIDE", dist, 0.07)), 0.0)
            band = math_node(m, "MULTIPLY", band, 0.85)
            comb = n.new("ShaderNodeCombineXYZ")
            for k in range(3):
                l.new(band, comb.inputs[k])
            add = n.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            l.new(tex.outputs["Color"], add.inputs[0])
            l.new(comb.outputs[0], add.inputs[1])
            col = add.outputs["Vector"]
        alpha = math_node(m, "MULTIPLY", tex.outputs["Alpha"], fade.outputs[0])
        em = n.new("ShaderNodeEmission")
        l.new(col, em.inputs["Color"])
        tr = n.new("ShaderNodeBsdfTransparent")
        out = n.new("ShaderNodeOutputMaterial")
        if additive:
            l.new(math_node(m, "MULTIPLY", alpha, float(strength)), em.inputs["Strength"])
            add_s = n.new("ShaderNodeAddShader")
            l.new(tr.outputs["BSDF"], add_s.inputs[0])
            l.new(em.outputs["Emission"], add_s.inputs[1])
            l.new(add_s.outputs["Shader"], out.inputs["Surface"])
        else:
            em.inputs["Strength"].default_value = float(strength)
            mix = n.new("ShaderNodeMixShader")
            l.new(alpha, mix.inputs[0])
            l.new(tr.outputs["BSDF"], mix.inputs[1])
            l.new(em.outputs["Emission"], mix.inputs[2])
            l.new(mix.outputs["Shader"], out.inputs["Surface"])
        render_method_blended(m)
        me.materials.append(m)
        ob.hide_render = True
        return {"ob": ob, "me": me, "dz": dz, "fade": fade, "sweep": sweep, "name": name}

    def update(self, t, f):
        k = t - self.at
        vis = 0.0 <= k <= self.dur
        for it in self.items:
            it["ob"].hide_render = not vis
        if not vis:
            return
        ctx = self.ctx
        a = float(smooth(k / 0.22) * (1.0 - smooth((k - (self.dur - 0.45)) / 0.45)))
        ease = 1.0 - (1.0 - min(1.0, k / 0.5)) ** 3
        sc = 1.0 + 0.10 * (1.0 - ease)
        yaw, pitch = math.radians(6.0) * math.sin(0.9 * t), math.radians(3.0) * math.sin(0.7 * t + 1.0)
        cam = Cam(ctx, f)
        L0 = 0.35 * ctx["D"]
        cy_, sy_, cp_, sp_ = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        for it in self.items:
            Li = L0 + it["dz"] * ctx["D"] * 0.25                     # farther layers are scaled up so they still fill the frame
            wl = cam.wl(Li) * sc
            x, y = corners[:, 0] * wl, corners[:, 1] * wl / ASPECT
            d = np.full(4, Li - L0)                                    # depth of the layer relative to the face plane
            x1, d1 = x * cy_ + d * sy_, -x * sy_ + d * cy_            # yaw about the up axis
            y2, d2 = y * cp_ - d1 * sp_, y * sp_ + d1 * cp_           # pitch about the right axis
            verts = cam.pos + cam.fwd * (L0 + d2)[:, None] + cam.right * x1[:, None] + cam.up * y2[:, None]
            it["me"].vertices.foreach_set("co", verts.astype(np.float32).ravel())
            it["me"].update()
            it["fade"].outputs[0].default_value = a
            it["sweep"].outputs[0].default_value = -0.3 + 1.6 * ((k - 0.45) / 0.8)


# ================================================================== eye patches (baked by eye_rig.bake_patches)
class EyePatches:
    Z_OFFSET = 0.004

    def __init__(self, ctx, patches):
        self.ctx, self.items = ctx, []
        CX, CY, FW, D, WF0, HF0 = ctx["CX"], ctx["CY"], ctx["FW"], ctx["D"], ctx["WF0"], ctx["HF0"]
        for k, p in enumerate(patches):
            X0, Y0, X1, Y1 = p["box"]
            nx, ny = 12, 8
            xs, ys = np.linspace(X0, X1, nx + 1), np.linspace(Y0, Y1, ny + 1)
            XX, YY = np.meshgrid(xs, ys)
            Z = ctx["zeta_at"](XX, YY) - self.Z_OFFSET                 # same surface as the plate mesh, a hair in front
            u0, v0 = (XX - CX) / FW, (YY - CY) / FW
            verts = np.stack([u0 * WF0 * Z, Z * D, -v0 * HF0 * Z], -1).reshape(-1, 3)
            uvs = np.stack([(XX - X0) / (X1 - X0), 1.0 - (YY - Y0) / (Y1 - Y0)], -1).reshape(-1, 2)
            faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i) for j in range(ny) for i in range(nx)]
            me = ctx["build_mesh"](f"eye_patch_{k}", verts, faces, uvs)
            ob = bpy.data.objects.new(f"eye_patch_{k}", me)
            ctx["scene"].collection.objects.link(ob)
            no_gi(ob)
            m = ctx["new_mat"](f"eye_patch_{k}")
            n, l = m.node_tree.nodes, m.node_tree.links
            tex = n.new("ShaderNodeTexImage")
            tex.extension, tex.interpolation = "EXTEND", "Linear"
            uv = n.new("ShaderNodeTexCoord")
            l.new(uv.outputs["UV"], tex.inputs["Vector"])
            em = n.new("ShaderNodeEmission")
            l.new(tex.outputs["Color"], em.inputs["Color"])
            em.inputs["Strength"].default_value = 1.0
            tr = n.new("ShaderNodeBsdfTransparent")
            mix = n.new("ShaderNodeMixShader")
            l.new(tex.outputs["Alpha"], mix.inputs[0])
            l.new(tr.outputs["BSDF"], mix.inputs[1])
            l.new(em.outputs["Emission"], mix.inputs[2])
            out = n.new("ShaderNodeOutputMaterial")
            l.new(mix.outputs["Shader"], out.inputs["Surface"])
            render_method_blended(m)
            me.materials.append(m)
            self.items.append({"tex": tex, "files": p["files"], "cache": {}, "ob": ob})
        self.step = max(1, int(ctx["shot"].get("step", 2)))

    def update(self, t, f):
        i = (f - 1) // self.step
        for it in self.items:
            files = it["files"]
            path = files[min(i, len(files) - 1)]
            img = it["cache"].get(path)
            if img is None:
                img = it["cache"][path] = self.ctx["load_image"](path)
            if it["tex"].image is not img:
                it["tex"].image = img


# ================================================================== ground dressing
def toon_material(ctx, name, light=(-0.45, -0.6, 0.65), tones=(0.50, 0.80, 1.05), cuts=(0.40, 0.68)):
    """Unlit 'cel' shading that works in Cycles AND EEVEE (no Shader-to-RGB): N.L through a constant ColorRamp x vertex colour."""
    m = ctx["new_mat"](name)
    n, l = m.node_tree.nodes, m.node_tree.links
    geo = n.new("ShaderNodeNewGeometry")
    dot = n.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    lv = np.asarray(light, float)
    lv = lv / np.linalg.norm(lv)
    l.new(geo.outputs["Normal"], dot.inputs[0])
    dot.inputs[1].default_value = (float(lv[0]), float(lv[1]), float(lv[2]))
    half = math_node(m, "MULTIPLY_ADD", dot.outputs["Value"], 0.5, 0.5)
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (tones[0], tones[0], tones[0], 1.0)
    els[1].position, els[1].color = cuts[0], (tones[1], tones[1], tones[1], 1.0)
    e3 = els.new(cuts[1])
    e3.color = (tones[2], tones[2], tones[2], 1.0)
    l.new(half, ramp.inputs["Fac"])
    attr = n.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    mul = n.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    l.new(attr.outputs["Color"], mul.inputs[0])
    l.new(ramp.outputs["Color"], mul.inputs[1])
    em = n.new("ShaderNodeEmission")
    l.new(mul.outputs["Vector"], em.inputs["Color"])
    out = n.new("ShaderNodeOutputMaterial")
    l.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def flat_vertex_material(ctx, name, alpha=False):
    m = ctx["new_mat"](name)
    n, l = m.node_tree.nodes, m.node_tree.links
    attr = n.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    em = n.new("ShaderNodeEmission")
    l.new(attr.outputs["Color"], em.inputs["Color"])
    out = n.new("ShaderNodeOutputMaterial")
    if alpha:
        tr = n.new("ShaderNodeBsdfTransparent")
        mix = n.new("ShaderNodeMixShader")
        l.new(attr.outputs["Alpha"], mix.inputs[0])
        l.new(tr.outputs["BSDF"], mix.inputs[1])
        l.new(em.outputs["Emission"], mix.inputs[2])
        l.new(mix.outputs["Shader"], out.inputs["Surface"])
        render_method_blended(m)
    else:
        l.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def add_color_attr(me, cols):
    ca = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    ca.data.foreach_set("color", np.asarray(cols, np.float32).ravel())
    return ca


def new_mesh_object(ctx, name, verts, faces, cols, material, collection=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    ca = add_color_attr(me, cols)
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    (collection or ctx["scene"].collection).objects.link(ob)
    no_gi(ob)
    return ob, me, ca


class Dressing:
    """Pebbles / fallen leaves (static, toon shaded) and swaying grass blades, placed on the plate's ground band."""

    def __init__(self, ctx, kinds, amounts):
        self.ctx, self.kinds = ctx, kinds
        self.rs = np.random.default_rng(int(ctx["seed"]) * 31 + 17)
        CX, CY, FW = ctx["CX"], ctx["CY"], ctx["FW"]
        self.xlo, self.xhi = max(0.01, CX - FW / 2 * 1.04), min(0.99, CX + FW / 2 * 1.04)
        self.ybot = min(0.985, CY + FW / 2)
        self.ytop = max(float(ctx["plate"].get("ground_y") or 0.85) + 0.01, self.ybot - 0.30 * FW)
        pal = ctx["pal"]
        self.ground = np.clip(np.asarray(pal("bottom", (0.30, 0.27, 0.22)), float), 0, 1)
        self.props_coll = bpy.data.collections.new("mf_props")
        ctx["scene"].collection.children.link(self.props_coll)
        self.grass = None
        if "pebbles" in kinds:
            self.build_pebbles(amounts.get("pebbles", 0.6))
            ctx["mark"]("pebbles", "ok")
        if "leaves" in kinds:
            self.build_leaves(amounts.get("leaves", 0.6))
            ctx["mark"]("leaves_ground", "ok")
        if "grass" in kinds:
            self.build_grass(amounts.get("grass", 0.6))
            ctx["mark"]("grass", "ok")

    # --- placement
    def spots(self, n, bias=1.5):
        ctx, rs = self.ctx, self.rs
        X = rs.uniform(self.xlo, self.xhi, n)
        Y = self.ybot - (rs.random(n) ** bias) * (self.ybot - self.ytop)
        Z = ctx["zeta_at"](X, Y) - 0.010
        P = np.stack([(X - ctx["CX"]) / ctx["FW"] * ctx["WF0"] * Z, Z * ctx["D"], -((Y - ctx["CY"]) / ctx["FW"]) * ctx["HF0"] * Z], 1)
        depth01 = (Y - self.ytop) / max(1e-6, self.ybot - self.ytop)            # 0 far .. 1 near (bottom of the frame)
        return X, Y, Z, P, depth01

    def tint(self, n, lo=0.55, hi=1.25):
        rs = self.rs
        base = self.ground[None, :] * rs.uniform(lo, hi, (n, 1))
        return np.clip(base + rs.uniform(-0.03, 0.05, (n, 3)), 0, 1)

    # --- pebbles
    def build_pebbles(self, amount):
        ctx, rs = self.ctx, self.rs
        n = int(10 + 30 * amount)
        X, Y, Z, P, near = self.spots(n)
        seg_u, rings = 8, 4
        base = [(0.0, 0.0, 1.0)]
        for r in range(1, rings):
            ph = math.pi * r / rings
            for u in range(seg_u):
                th = TAU * u / seg_u
                base.append((math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)))
        base.append((0.0, 0.0, -1.0))
        base = np.array(base)
        nv = len(base)
        tri = []
        for u in range(seg_u):                                                     # cap, rings, bottom
            tri.append((0, 1 + u, 1 + (u + 1) % seg_u))
        for r in range(rings - 2):
            for u in range(seg_u):
                a, b = 1 + r * seg_u + u, 1 + r * seg_u + (u + 1) % seg_u
                tri.append((a, a + seg_u, b + seg_u, b))
        last = 1 + (rings - 2) * seg_u
        for u in range(seg_u):
            tri.append((nv - 1, last + (u + 1) % seg_u, last + u))
        verts, faces, cols, shadow_v, shadow_f, shadow_c = [], [], [], [], [], []
        for i in range(n):
            size = rs.uniform(0.0045, 0.014) * (0.55 + 1.5 * near[i])             # nearer to the bottom of the frame = bigger
            sx, sy, sz = size * rs.uniform(0.8, 1.3), size * rs.uniform(0.7, 1.1), size * rs.uniform(0.45, 0.75)
            rot = rs.uniform(0, TAU)
            noise = 1.0 + rs.uniform(-0.18, 0.18, (nv, 1))
            pts = base * noise * np.array([sx, sy, sz])
            c, s = math.cos(rot), math.sin(rot)
            pts = np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c, pts[:, 2]], 1)
            wsz = ctx["WF0"] * Z[i]
            pts = pts * wsz
            pts[:, 2] += sz * wsz * 0.55                                           # sit on the ground
            verts.append(P[i] + pts)
            off = i * nv
            faces += [tuple(off + k for k in f) for f in tri]
            cols.append(np.tile(self.tint(1, 0.6, 1.2), (nv, 1)))
            # contact shadow: a camera-facing flattened fan with an alpha falloff
            k0 = len(shadow_v)
            ew, eh = sx * wsz * 1.9, sz * wsz * 0.55
            ctr = P[i] + np.array([0.0, 0.002, -0.001])
            shadow_v.append(ctr)
            shadow_c.append((0.03, 0.025, 0.03, 0.55))
            for a in range(10):
                th = TAU * a / 10
                shadow_v.append(ctr + np.array([math.cos(th) * ew, 0.0, math.sin(th) * eh]))
                shadow_c.append((0.03, 0.025, 0.03, 0.0))
            shadow_f += [(k0, k0 + 1 + a, k0 + 1 + (a + 1) % 10) for a in range(10)]
        verts = np.concatenate(verts)
        cols = np.concatenate(cols)
        rgba = np.concatenate([cols, np.ones((len(cols), 1))], 1)
        mat = toon_material(ctx, "pebbles_toon")
        self.pebbles, _, _ = new_mesh_object(ctx, "pebbles", verts, faces, rgba, mat, self.props_coll)
        smat = flat_vertex_material(ctx, "contact_shadow", alpha=True)
        new_mesh_object(ctx, "contact_shadows", np.array(shadow_v), shadow_f, np.array(shadow_c), smat)

    # --- fallen leaves
    def build_leaves(self, amount):
        ctx, rs = self.ctx, self.rs
        n = int(8 + 24 * amount)
        X, Y, Z, P, near = self.spots(n, bias=1.2)
        palette = np.array([(0.86, 0.46, 0.12), (0.74, 0.27, 0.10), (0.58, 0.52, 0.12), (0.80, 0.62, 0.15)])
        verts, faces, cols = [], [], []
        shape = np.array([[0.0, -1.0], [0.52, -0.1], [0.0, 1.0], [-0.52, -0.1]])
        for i in range(n):
            size = rs.uniform(0.007, 0.017) * (0.6 + 1.3 * near[i]) * ctx["WF0"] * Z[i]
            rot = rs.uniform(0, TAU)
            c, s = math.cos(rot), math.sin(rot)
            tilt = rs.uniform(0.15, 0.5)                                            # lies on the ground, seen at a grazing angle
            pts = []
            for (x, y) in shape:
                xr, yr = x * c - y * s, x * s + y * c
                pts.append((xr * size, 0.002 * i + yr * size * tilt * 0.35, yr * size * tilt))
            verts.append(P[i] + np.array(pts))
            faces.append(tuple(4 * i + k for k in range(4)))
            cols.append(np.tile(palette[rs.integers(0, len(palette))] * rs.uniform(0.8, 1.05), (4, 1)))
        verts, cols = np.concatenate(verts), np.concatenate(cols)
        rgba = np.concatenate([cols, np.ones((len(cols), 1))], 1)
        self.leaves, _, _ = new_mesh_object(ctx, "ground_leaves", verts, faces, rgba, toon_material(ctx, "leaves_toon"), self.props_coll)

    # --- grass
    def build_grass(self, amount):
        ctx, rs = self.ctx, self.rs
        n = int(160 + 420 * amount)
        FW = ctx["FW"]
        X = rs.uniform(self.xlo, self.xhi, n)
        Y = np.minimum(0.99, self.ybot + 0.02 - (rs.random(n) ** 0.9) * 0.16 * FW)
        Z = ctx["zeta_at"](X, Y) - 0.012
        self.gP = np.stack([(X - ctx["CX"]) / FW * ctx["WF0"] * Z, Z * ctx["D"], -((Y - ctx["CY"]) / FW) * ctx["HF0"] * Z], 1)
        wsz = ctx["WF0"] * Z
        self.gh = rs.uniform(0.035, 0.115, n) * wsz * (0.8 + 0.6 * (Y - self.ytop) / max(1e-6, self.ybot - self.ytop))
        self.gw = rs.uniform(0.0028, 0.0055, n) * wsz
        self.glean = rs.uniform(-0.28, 0.28, n)
        self.gph, self.gf = rs.uniform(0, TAU, n), rs.uniform(0.5, 1.1, n)
        self.n_grass = n
        levels = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        faces = []
        for i in range(n):
            o = i * 8
            for lv in range(3):
                faces.append((o + lv * 2, o + lv * 2 + 1, o + lv * 2 + 3, o + lv * 2 + 2))
        g = self.ground
        green = np.clip(0.55 * g + 0.45 * np.array([0.28, 0.5, 0.2]), 0, 1)
        shade = np.array([0.50, 0.78, 1.0, 1.25])
        cols = np.zeros((n, 4, 2, 3))
        base_var = rs.uniform(0.85, 1.15, (n, 1))
        for lv in range(4):
            cols[:, lv, :, :] = np.clip(green[None, None, :] * shade[lv] * base_var[:, :, None].transpose(0, 2, 1)[:, :1, :] if False else
                                        green[None, :] * shade[lv] * base_var, 0, 1)[:, None, :]
        rgba = np.concatenate([cols.reshape(-1, 3), np.ones((n * 8, 1))], 1)
        self.grass_ob, self.grass_me, _ = new_mesh_object(ctx, "grass", np.zeros((n * 8, 3)), faces, rgba, flat_vertex_material(ctx, "grass_mat"))
        self.levels = levels
        self.grass_update(0.0)

    def grass_update(self, t):
        ctx = self.ctx
        n = self.n_grass
        wind = float((ctx["shot"].get("wind") or [0.5, 0.05])[0])
        gust = 0.55 + 0.45 * math.sin(TAU * 0.33 * t + (int(ctx["seed"]) % 7))
        bend = self.glean + wind * gust * (0.34 + 0.18 * np.sin(TAU * self.gf * t + self.gph))
        verts = np.zeros((n, 4, 2, 3))
        for lv, s in enumerate(self.levels):
            xo = self.gh * np.sin(bend * s * s * 1.2)
            zo = self.gh * s * (1.0 - 0.12 * s * s)
            w = self.gw * (1.0 - s) ** 0.8
            verts[:, lv, 0, :] = self.gP + np.stack([xo - w, np.zeros(n), zo], 1)
            verts[:, lv, 1, :] = self.gP + np.stack([xo + w, np.zeros(n), zo], 1)
        self.grass_me.vertices.foreach_set("co", verts.reshape(-1, 3).astype(np.float32).ravel())
        self.grass_me.update()

    def update(self, t, f):
        if self.grass is not False and "grass" in self.kinds:
            self.grass_update(t)


# ================================================================== curve strokes (slash / lightning / swirl)
class Stroke:
    """A tapered, glowing, bevelled curve (core + wider additive glow) drawn on with bevel_factor."""

    def __init__(self, ctx, name, npts, width, core_color, glow_color, core_strength=9.0, glow_strength=2.4):
        self.ctx, self.n = ctx, npts
        self.cus, self.obs, self.fades = [], [], []
        for suffix, scale, color, strength, additive in (("", 1.0, core_color, core_strength, False), ("_glow", 3.6, glow_color, glow_strength, True)):
            cu = bpy.data.curves.new(name + suffix, "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = width * scale
            cu.bevel_resolution = 2
            cu.resolution_u = 2
            try:
                cu.use_fill_caps = False
            except Exception:  # noqa: BLE001
                pass
            sp = cu.splines.new("POLY")
            sp.points.add(npts - 1)
            m, fade = glow_material(ctx, name + suffix + "_mat", color, strength, additive)
            cu.materials.append(m)
            ob = bpy.data.objects.new(name + suffix, cu)
            ctx["scene"].collection.objects.link(ob)
            no_gi(ob)
            ob.hide_render = True
            self.cus.append((cu, sp))
            self.obs.append(ob)
            self.fades.append(fade)

    def set_points(self, pts, radius):
        co = np.concatenate([np.asarray(pts, float), np.ones((self.n, 1))], 1).astype(np.float32).ravel()
        for cu, sp in self.cus:
            sp.points.foreach_set("co", co)
            sp.points.foreach_set("radius", np.asarray(radius, np.float32))

    def show(self, vis, head=1.0, tail=0.0, alpha=1.0):
        for (cu, sp), ob, fade in zip(self.cus, self.obs, self.fades):
            ob.hide_render = not vis
            if vis:
                cu.bevel_factor_end = float(np.clip(head, 0.0, 1.0))
                cu.bevel_factor_start = float(np.clip(tail, 0.0, 1.0))
                fade.outputs[0].default_value = float(alpha)


class Strokes:
    def __init__(self, ctx):
        self.ctx = ctx
        FXS = ctx["FXS"]
        self.su = (ctx["SUB"]["x"] - ctx["CX"]) / ctx["FW"]
        self.sv = (ctx["SUB"]["y"] - ctx["CY"]) / ctx["FW"]
        self.px, self.py = self.su, -self.sv / ASPECT                              # subject in physical coords (frame widths, y up)
        self.D, self.WF0 = ctx["D"], ctx["WF0"]
        acc = np.clip(np.asarray(ctx["pal"]("accent", (0.5, 0.8, 1.0)), float), 0, 1)
        self.accent = np.clip(0.7 * acc + 0.3, 0, 1)
        self.step = max(1, int(ctx["shot"].get("step", 2)))
        self.slashes, self.bolts, self.swirl = [], [], None
        fx = FXS.get("slash")
        if fx:
            self.s_at, self.s_amount = float(fx.get("at") or 0.0), float(fx.get("amount") or 0.6)
            rs = np.random.default_rng(int(ctx["seed"]) * 5 + 1)
            k = 1 + int(round(2 * self.s_amount))
            for i in range(k):
                st = Stroke(ctx, f"slash{i}", 28, 0.010 * self.WF0, (1.0, 1.0, 1.0), self.accent)
                ang = math.radians(rs.uniform(-42, 42)) + (math.pi if i % 2 else 0.0)
                half = rs.uniform(0.26, 0.42)
                ctr = (self.px + rs.uniform(-0.05, 0.05), self.py + 0.01 + rs.uniform(-0.04, 0.04))
                s = np.linspace(-1, 1, 28)
                bulge = rs.uniform(0.04, 0.09) * (1 - s * s) * (1 if i % 2 else -1)
                x = ctr[0] + math.cos(ang) * s * half - math.sin(ang) * bulge
                y = ctr[1] + math.sin(ang) * s * half + math.cos(ang) * bulge
                st.set_points(phys_to_world(ctx, x, y, 0.97 + 0.03 * (s * 0.5 + 0.5)), np.sin(np.pi * (s * 0.5 + 0.5) ** 0.8) ** 0.7 + 0.04)
                self.slashes.append((st, 0.06 * i))
            ctx["mark"]("slash", "ok")
        fx = FXS.get("lightning")
        if fx:
            self.l_at, self.l_dur = float(fx.get("at") or 0.0), float(fx.get("dur") or 0.6)
            for i, npts in enumerate((33, 17, 17)):
                self.bolts.append(Stroke(ctx, f"bolt{i}", npts, (0.010 if i == 0 else 0.006) * self.WF0, (1.0, 1.0, 1.0), (0.55, 0.75, 1.0), 12.0, 3.0))
            ctx["mark"]("lightning", "ok")
        fx = FXS.get("swirl")
        if fx:
            self.w_at, self.w_dur, self.w_amount = float(fx.get("at") or 0.0), float(fx.get("dur") or 2.4), float(fx.get("amount") or 0.6)
            self.swirl = Stroke(ctx, "swirl", 56, 0.0085 * self.WF0, (1.0, 1.0, 1.0), self.accent, 7.0, 2.2)
            ctx["mark"]("swirl", "ok")

    @staticmethod
    def bolt_path(rs, a, b, levels, amp):
        pts = [np.asarray(a, float), np.asarray(b, float)]
        for lv in range(levels):
            nxt = []
            for p, q in zip(pts[:-1], pts[1:]):
                mid = (p + q) / 2
                d = q - p
                nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
                nxt += [p, mid + nrm * rs.uniform(-1, 1) * amp / (1.6 ** lv)]
            nxt.append(pts[-1])
            pts = nxt
        return np.array(pts)

    def update(self, t, f):
        ctx = self.ctx
        for st, delay in self.slashes:
            k = t - self.s_at - delay
            if k < 0 or k > 0.7:
                st.show(False)
                continue
            head = float(smooth(k / 0.08))
            tail = float(smooth((k - 0.10) / 0.20)) * head
            st.show(True, head, tail, 1.0 - float(smooth((k - 0.34) / 0.25)))
        if self.bolts:
            k = t - self.l_at
            if k < 0 or k > self.l_dur:
                for b in self.bolts:
                    b.show(False)
            else:
                di = (int(f) - 1) // self.step
                rs = np.random.default_rng(int(ctx["seed"]) * 97 + di * 7919)
                flick = (1.0, 0.25, 0.9, 0.05, 0.7, 0.3, 0.55, 0.12)[di % 8] * (1.0 - k / self.l_dur) ** 0.6
                top = np.array([self.px + rs.uniform(-0.18, 0.18), 0.5 / ASPECT + 0.08])
                end = np.array([self.px + rs.uniform(-0.04, 0.04), self.py + 0.02])
                main = self.bolt_path(rs, top, end, 5, 0.07)                        # 2^5 + 1 = 33 points
                self.bolts[0].set_points(phys_to_world(ctx, main[:, 0], main[:, 1], np.full(33, 0.99)), np.ones(33))
                self.bolts[0].show(True, 1.0, 0.0, flick)
                for j in (1, 2):
                    i0 = int(rs.integers(6, 22))
                    br_end = main[i0] + np.array([rs.uniform(-0.18, 0.18), -rs.uniform(0.06, 0.16)])
                    br = self.bolt_path(rs, main[i0], br_end, 4, 0.04)              # 2^4 + 1 = 17 points
                    self.bolts[j].set_points(phys_to_world(ctx, br[:, 0], br[:, 1], np.full(17, 0.99)), np.linspace(1.0, 0.25, 17))
                    self.bolts[j].show(True, 1.0, 0.0, flick * 0.8)
        if self.swirl is not None:
            k = t - self.w_at
            if k < 0 or k > self.w_dur:
                self.swirl.show(False)
            else:
                n = 56
                s = np.linspace(0, 1, n)
                th = s * TAU * 2.3 + 2.4 * k
                r = 0.05 + 0.24 * s * (0.8 + 0.4 * self.w_amount)
                x = self.px + r * np.cos(th)
                y = self.py + 0.02 + r * np.sin(th) * 0.62
                z = 1.0 - 0.07 * np.sin(th)                                          # wraps around the subject: in front on one half, behind on the other
                self.swirl.set_points(phys_to_world(ctx, x, y, z), np.sin(np.pi * s) ** 0.6 + 0.05)
                head = float(smooth(k / 0.7))
                tail = max(0.0, head - 0.55) if k > 0.7 else 0.0
                self.swirl.show(True, head, tail, 1.0 - float(smooth((k - (self.w_dur - 0.5)) / 0.5)))


# ================================================================== compositor / passes / freestyle
def comp_tree(scene):
    """-> (tree, output_node, new_api).  Blender <= 4.5: scene.node_tree + Composite;  5.x: compositing_node_group + Group Output."""
    try:
        scene.use_nodes = True
        tree = scene.node_tree
        if tree is not None:
            tree.nodes.clear()
            return tree, tree.nodes.new("CompositorNodeComposite"), False
    except Exception:  # noqa: BLE001
        pass
    ng = bpy.data.node_groups.new("mf_comp", "CompositorNodeTree")
    scene.compositing_node_group = ng
    try:
        ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    except Exception:  # noqa: BLE001
        pass
    return ng, ng.nodes.new("NodeGroupOutput"), True


def set_any(node, names_values):
    ok = False
    for name, val in names_values:
        try:
            setattr(node, name, val)
            ok = True
        except Exception:  # noqa: BLE001
            try:
                node.inputs[name].default_value = val
                ok = True
            except Exception:  # noqa: BLE001
                pass
    return ok


def build_compositor(ctx, passes):
    scene = ctx["scene"]
    tree, out, new_api = comp_tree(scene)
    rl = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    for gt in ("BLOOM", "FOG_GLOW"):
        try:
            glare.glare_type = gt
            break
        except Exception:  # noqa: BLE001
            continue
    set_any(glare, [("threshold", 1.0), ("Threshold", 1.0), ("size", 7), ("Size", 0.5), ("mix", -0.55), ("Strength", 0.35), ("quality", "HIGH")])
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], out.inputs[0] if not new_api else out.inputs["Image"])
    ctx["mark"]("compositor_glare_bloom", "ok")
    if passes:
        vl = scene.view_layers[0]
        vl.use_pass_z = True
        vl.use_pass_mist = True
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.base_path = str(ctx["FDIR"] / "passes")
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "16"
        try:
            fo.file_slots[0].path = "depth_"
            tree.links.new(rl.outputs["Depth"], fo.inputs[0])
        except Exception:  # noqa: BLE001
            fo.file_output_items.new("FLOAT", "depth_")
            tree.links.new(rl.outputs["Depth"], fo.inputs[0])
        ctx["mark"]("passes_exr_depth", "ok")


def build_freestyle(ctx, coll):
    scene = ctx["scene"]
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = "ABSOLUTE"
    scene.render.line_thickness = max(0.8, ctx["scene"].render.resolution_x / 1280.0)
    vl = scene.view_layers[0]
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new("ink")
    ls.select_by_visibility = True
    ls.select_by_collection = True
    ls.collection = coll
    ls.select_silhouette, ls.select_border, ls.select_crease = True, True, True
    ls.linestyle.color = (0.05, 0.03, 0.04)
    ls.linestyle.thickness = 1.6


# ================================================================== physics / volumetrics tier
def mantaflow(ctx, kind):
    scene = ctx["scene"]
    loc = ctx["s2w"](np.array(ctx["SUB"]["x"] - ctx["CX"]) / ctx["FW"], np.array(0.30), np.array(0.9))
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.35, location=tuple(float(v) for v in loc))
    ob = bpy.context.active_object
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.quick_smoke(style="FIRE" if kind == "fire" else "SMOKE", show_flows=False)
    dom = bpy.context.active_object
    ds = dom.modifiers["Fluid"].domain_settings
    ds.resolution_max = 48
    try:
        ds.use_noise = False
    except Exception:  # noqa: BLE001
        pass
    dom.location = (float(loc[0]), float(loc[1]), float(loc[2]) + 1.6)
    dom.scale = (2.4, 2.4, 3.0)
    no_gi(dom)
    ob.hide_render = True
    ctx["fx_disabled"].add(kind)
    ctx["mark"](kind, "ok")
    ctx["mark"]("mantaflow_" + kind, "ok")


def volume_fog(ctx):
    scene = ctx["scene"]
    z = 0.86
    loc = ctx["s2w"](np.array(0.0), np.array(max(-0.1, min(0.4, (float(ctx["plate"].get("ground_y") or 0.85) - ctx["CY"]) / ctx["FW"] - 0.05))), np.array(z))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(float(v) for v in loc))
    ob = bpy.context.active_object
    ob.scale = (ctx["WF0"] * z * 1.7, 3.2, ctx["HF0"] * z * 0.42)
    m = ctx["new_mat"]("fog_volume")
    n, l = m.node_tree.nodes, m.node_tree.links
    pv = n.new("ShaderNodeVolumePrincipled")
    avg = np.clip(np.asarray(ctx["pal"]("avg", (0.5, 0.5, 0.55)), float), 0, 1)
    pv.inputs["Color"].default_value = rgba_lin(0.5 * avg + 0.5 * np.array([0.85, 0.88, 0.95]))
    pv.inputs["Density"].default_value = 0.0
    try:
        pv.inputs["Emission Strength"].default_value = 0.25
        pv.inputs["Emission Color"].default_value = rgba_lin(0.6 * avg + 0.4)
    except Exception:  # noqa: BLE001
        pass
    tc = n.new("ShaderNodeTexCoord")
    nz = n.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1.1
    w = n.new("ShaderNodeValue")
    l.new(tc.outputs["Object"], nz.inputs["Vector"])
    try:
        nz.noise_dimensions = "4D"
        l.new(w.outputs[0], nz.inputs["W"])
    except Exception:  # noqa: BLE001
        pass
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.42, 0.78
    l.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    dens = math_node(m, "MULTIPLY", ramp.outputs["Color"], 0.9)
    l.new(dens, pv.inputs["Density"])
    out = n.new("ShaderNodeOutputMaterial")
    l.new(pv.outputs["Volume"], out.inputs["Volume"])
    ob.data.materials.append(m)
    ctx["fx_disabled"].add("volume_fog")
    ctx["mark"]("volume_fog", "ok")
    ctx["mark"]("volumetrics_principled", "ok")

    def drift(t, f):
        w.outputs[0].default_value = t * 0.07
    return named(drift, "volume_fog_drift")


def cloth_banner(ctx):
    S = ctx["S"]
    loc = ctx["s2w"](np.array(0.30), np.array(-0.30), np.array(0.85))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=16, y_subdivisions=10, size=1.0, location=tuple(float(v) for v in loc))
    ob = bpy.context.active_object
    ob.rotation_euler = (math.radians(90), 0.0, 0.0)
    ob.scale = (ctx["WF0"] * 0.20, ctx["HF0"] * 0.28, 1.0)
    me = ob.data
    pin = ob.vertex_groups.new(name="pin")
    xs = np.array([v.co.x for v in me.vertices])
    pin.add([i for i, x in enumerate(xs) if x <= xs.min() + 1e-4], 1.0, "REPLACE")
    cl = ob.modifiers.new("Cloth", "CLOTH")
    cl.settings.vertex_group_mass = "pin"
    cl.settings.quality = 6
    cl.collision_settings.use_collision = False
    bpy.ops.object.effector_add(type="WIND", location=(float(loc[0]) - 2.0, float(loc[1]), float(loc[2])))
    wind = bpy.context.active_object
    wind.rotation_euler = (0.0, math.radians(90), 0.0)
    wind.field.strength = 6.0 * (1 if float((ctx["shot"].get("wind") or [1.0])[0]) >= 0 else -1)
    wind.field.noise = 1.5
    acc = np.clip(np.asarray(ctx["pal"]("accent", (0.8, 0.3, 0.2)), float), 0, 1)
    m, _ = glow_material(ctx, "banner", acc, 1.0, False)
    me.materials.append(m)
    ctx["mark"]("cloth_flag", "ok")
    ctx["mark"]("cloth_simulation", "ok")


# ================================================================== entry point
def setup(ctx):
    dyn = []
    feat, mark, SHOT, S, FXS = ctx["feat"], ctx["mark"], ctx["shot"], ctx["S"], ctx["FXS"]
    ctx.setdefault("fx_disabled", set())

    t3 = SHOT.get("title3d")
    with feat("title_3d", bool(t3), "off"):
        if t3:
            dyn.append(named(TitleRig(ctx, t3).update, "title_3d"))

    patches = SHOT.get("eye_patches")
    with feat("eye_patches", bool(patches), "off"):
        if patches:
            dyn.append(named(EyePatches(ctx, patches).update, "eye_patches"))
            for p in patches[:1]:
                for k, v in (p.get("report") or {}).items():
                    mark(k, v)

    kinds = [k for k in ("pebbles", "grass", "leaves") if k in (SHOT.get("auto_dress") or []) or (k in FXS and k != "leaves")]
    amounts = {k: float((FXS.get(k) or {}).get("amount") or 0.6) for k in kinds}
    dress = None
    with feat("ground_dressing", bool(kinds), "off"):
        if kinds:
            dress = Dressing(ctx, kinds, amounts)
            if "grass" in kinds:
                dyn.append(named(dress.update, "grass_wind"))
    with feat("toon_shading", bool(dress and ("pebbles" in kinds or "leaves" in kinds)), "off"):
        if dress and ("pebbles" in kinds or "leaves" in kinds):
            mark("toon_shading", "ok")

    stroke_kinds = [k for k in ("slash", "lightning", "swirl") if k in FXS]
    with feat("curve_strokes", bool(stroke_kinds), "off"):
        if stroke_kinds:
            dyn.append(named(Strokes(ctx).update, "curve_strokes"))

    ink = SHOT.get("ink", S.get("ink", True)) and dress is not None and ("pebbles" in kinds or "leaves" in kinds)
    with feat("freestyle_ink", bool(ink), "off"):
        if ink:
            build_freestyle(ctx, dress.props_coll)
            mark("freestyle_ink", "ok")

    with feat("compositor", bool(S.get("compositor", True)), "off"):
        if S.get("compositor", True):
            build_compositor(ctx, bool(S.get("passes")))
            mark("compositor", "ok")

    phys = bool(S.get("physics"))
    for kind in ("smoke", "fire"):
        with feat("mantaflow_" + kind, kind in FXS and phys, "off" if kind not in FXS else "skipped:PHYSICS=0 (sprite version used)"):
            if kind in FXS and phys:
                mantaflow(ctx, kind)
    with feat("cloth_simulation", "cloth_flag" in FXS and phys, "off" if "cloth_flag" not in FXS else "skipped:PHYSICS=0"):
        if "cloth_flag" in FXS and phys:
            cloth_banner(ctx)
    with feat("volumetrics_principled", "volume_fog" in FXS and bool(S.get("volumetrics")), "off" if "volume_fog" not in FXS else "skipped:VOLUMETRICS=0 (sprite fog used)"):
        if "volume_fog" in FXS and S.get("volumetrics"):
            dyn.append(volume_fog(ctx))
    return dyn
