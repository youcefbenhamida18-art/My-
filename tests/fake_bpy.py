"""A stand-in for `bpy` / `mathutils` so the Blender scripts (blender_scene / blender_fx / blender_extras) can be EXECUTED and checked
without Blender (CI has the real thing; a laptop or a review sandbox may not).

What is real (so geometry can be asserted):   Vector / Matrix / Euler (numpy), meshes (vertices, loops, UVs, colour attributes),
                                              images (pixels read from disk, written back), object transforms + keyframes.
What is validated against Blender's real API: node-socket names + link direction for every shader node the scripts use,
                                              Math / VectorMath operations, texture enums, `from_pydata` / `foreach_set` sizes.
Everything else is permissive and recorded (`bpy._calls`), because inventing the whole API would only re-state our own guesses.
"""
from __future__ import annotations

import copy
import math
import sys
import types

import numpy as np


# ====================================================================== mathutils
class Vector:
    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self.v = np.array([float(x) for x in seq], dtype=np.float64)

    def __array__(self, dtype=None, copy=None):
        return self.v if dtype is None else self.v.astype(dtype)

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        return iter(self.v.tolist())

    def __getitem__(self, i):
        return float(self.v[i])

    def __setitem__(self, i, x):
        self.v[i] = float(x)

    x = property(lambda s: float(s.v[0]), lambda s, x: s.v.__setitem__(0, float(x)))
    y = property(lambda s: float(s.v[1]), lambda s, x: s.v.__setitem__(1, float(x)))
    z = property(lambda s: float(s.v[2]), lambda s, x: s.v.__setitem__(2, float(x)))

    def __add__(self, o):
        return Vector(self.v + np.asarray(o, float))

    __radd__ = __add__

    def __sub__(self, o):
        return Vector(self.v - np.asarray(o, float))

    def __rsub__(self, o):
        return Vector(np.asarray(o, float) - self.v)

    def __mul__(self, k):
        return Vector(self.v * float(k))

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vector(self.v / float(k))

    def __neg__(self):
        return Vector(-self.v)

    def __matmul__(self, o):
        return float(np.dot(self.v, np.asarray(o, float)))

    def tolist(self):
        return self.v.tolist()

    def copy(self):
        return Vector(self.v)

    @property
    def length(self):
        return float(np.linalg.norm(self.v))

    def normalized(self):
        n = np.linalg.norm(self.v)
        return Vector(self.v / n if n else self.v)

    def dot(self, o):
        return float(np.dot(self.v, np.asarray(o, float)))

    def cross(self, o):
        return Vector(np.cross(self.v, np.asarray(o, float)))

    def __repr__(self):
        return f"Vector({self.v.tolist()})"


class Euler:
    def __init__(self, seq=(0.0, 0.0, 0.0), order="XYZ"):
        self.v = np.array([float(x) for x in seq], dtype=np.float64)
        self.order = order

    x = property(lambda s: float(s.v[0]), lambda s, x: s.v.__setitem__(0, float(x)))
    y = property(lambda s: float(s.v[1]), lambda s, x: s.v.__setitem__(1, float(x)))
    z = property(lambda s: float(s.v[2]), lambda s, x: s.v.__setitem__(2, float(x)))

    def __array__(self, dtype=None, copy=None):
        return self.v

    def __iter__(self):
        return iter(self.v.tolist())

    def __getitem__(self, i):
        return float(self.v[i])

    def __len__(self):
        return 3

    def copy(self):
        return Euler(self.v, self.order)

    def to_matrix(self):
        a, b, c = self.v
        return Matrix(_rot("Z", c) @ _rot("Y", b) @ _rot("X", a))


def _rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


class Matrix:
    def __init__(self, rows=None):
        a = np.eye(4) if rows is None else np.array(rows, dtype=np.float64)
        if a.shape == (3, 3):
            b = np.eye(4)
            b[:3, :3] = a
            a = b
        assert a.shape == (4, 4), a.shape
        self.m = a

    def __array__(self, dtype=None, copy=None):
        return self.m if dtype is None else self.m.astype(dtype)

    def __matmul__(self, o):
        if isinstance(o, Matrix):
            return Matrix(self.m @ o.m)
        o = np.asarray(o, float)
        if o.shape == (3,):
            return Vector((self.m @ np.append(o, 1.0))[:3])
        return Vector(self.m @ o)

    def __getitem__(self, i):
        return self.m[i].tolist()

    def copy(self):
        return Matrix(self.m)

    def to_translation(self):
        return Vector(self.m[:3, 3])

    def to_3x3(self):
        return Matrix(self.m[:3, :3])

    def inverted(self):
        return Matrix(np.linalg.inv(self.m))

    def to_euler(self, order="XYZ", compat=None):
        assert order == "XYZ", "fake supports XYZ only"
        R = self.m[:3, :3]
        sy = math.hypot(R[0, 0], R[1, 0])
        sols = []
        b = math.atan2(-R[2, 0], sy)
        sols.append((math.atan2(R[2, 1], R[2, 2]), b, math.atan2(R[1, 0], R[0, 0])))
        sols.append((sols[0][0] + math.pi, math.pi - b, sols[0][2] + math.pi))

        def wrap(x, ref):
            return x + 2 * math.pi * round((ref - x) / (2 * math.pi))

        if compat is not None:
            ref = np.asarray(compat, float)
            sols = [tuple(wrap(v, r) for v, r in zip(s, ref)) for s in sols]
            best = min(sols, key=lambda s: sum(abs(v - r) for v, r in zip(s, ref)))
        else:
            best = min(sols, key=lambda s: sum(abs(v) for v in s))
        return Euler(best, order)

    @staticmethod
    def Rotation(angle, size=4, axis="Z"):
        r = _rot(axis.upper(), float(angle))
        return Matrix(r)

    @staticmethod
    def Translation(vec):
        m = np.eye(4)
        m[:3, 3] = np.asarray(vec, float)
        return Matrix(m)

    @staticmethod
    def Scale(f, size=4, axis=None):
        m = np.eye(4)
        if axis is None:
            m[:3, :3] *= float(f)
        return Matrix(m)

    def __repr__(self):
        return f"Matrix({self.m.round(4).tolist()})"


# ====================================================================== permissive recorder
CALLS: list = []


class Perm:
    """Any attribute exists (an auto-created Perm), any call is recorded and returns a Perm."""

    def __init__(self, _name="?"):
        object.__setattr__(self, "_name", _name)
        object.__setattr__(self, "_kids", {})

    def __getattr__(self, k):
        if k.startswith("__") or k in ("_name", "_kids"):
            raise AttributeError(k)
        kids = object.__getattribute__(self, "_kids")
        if k not in kids:
            kids[k] = Perm(f"{self._name}.{k}")
        return kids[k]

    def __call__(self, *a, **kw):
        CALLS.append((self._name, a, kw))
        return Perm(self._name + "()")

    def __getitem__(self, k):
        kids = self._kids
        if k not in kids:
            kids[k] = Perm(f"{self._name}[{k!r}]")
        return kids[k]

    def __setitem__(self, k, v):
        self._kids[k] = v

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def keyframe_insert(self, data_path, frame=None, index=-1, **kw):
        keys = self.__dict__.setdefault("_keys", {})
        keys[(data_path, frame)] = copy.deepcopy(getattr(self, data_path))
        return True


# ====================================================================== node trees (validated)
def _io(ins, outs):
    return list(ins), list(outs)


NODES = {
    "ShaderNodeTexCoord": _io([], ["Generated", "Normal", "UV", "Object", "Camera", "Window", "Reflection"]),
    "ShaderNodeTexImage": _io(["Vector"], ["Color", "Alpha"]),
    "ShaderNodeEmission": _io(["Color", "Strength", "Weight"], ["Emission"]),
    "ShaderNodeBsdfTransparent": _io(["Color", "Weight"], ["BSDF"]),
    "ShaderNodeMixShader": _io(["Fac", "Shader", "Shader"], ["Shader"]),
    "ShaderNodeAddShader": _io(["Shader", "Shader"], ["Shader"]),
    "ShaderNodeOutputMaterial": _io(["Surface", "Volume", "Displacement", "Thickness"], []),
    "ShaderNodeLightPath": _io([], ["Is Camera Ray", "Is Shadow Ray", "Is Diffuse Ray", "Is Glossy Ray", "Is Singular Ray",
                                    "Is Reflection Ray", "Is Transmission Ray", "Is Volume Scatter Ray", "Ray Length", "Ray Depth",
                                    "Diffuse Depth", "Glossy Depth", "Transparent Depth", "Transmission Depth"]),
    "ShaderNodeValue": _io([], ["Value"]),
    "ShaderNodeMath": _io(["Value", "Value", "Value"], ["Value"]),
    "ShaderNodeSeparateXYZ": _io(["Vector"], ["X", "Y", "Z"]),
    "ShaderNodeCombineXYZ": _io(["X", "Y", "Z"], ["Vector"]),
    "ShaderNodeTexNoise": _io(["Vector", "W", "Scale", "Detail", "Roughness", "Lacunarity", "Offset", "Gain", "Distortion"], ["Fac", "Color"]),
    "ShaderNodeValToRGB": _io(["Fac"], ["Color", "Alpha"]),
    "ShaderNodeVectorMath": _io(["Vector", "Vector", "Vector", "Scale"], ["Vector", "Value"]),
    "ShaderNodeAttribute": _io([], ["Color", "Vector", "Fac", "Alpha"]),
    "ShaderNodeNewGeometry": _io([], ["Position", "Normal", "Tangent", "True Normal", "Incoming", "Parametric", "Backfacing",
                                      "Pointiness", "Random Per Island"]),
    "ShaderNodeBackground": _io(["Color", "Strength", "Weight"], ["Background"]),
}
MATH_OPS = set("""ADD SUBTRACT MULTIPLY DIVIDE MULTIPLY_ADD POWER LOGARITHM SQRT INVERSE_SQRT ABSOLUTE EXPONENT MINIMUM MAXIMUM LESS_THAN
GREATER_THAN SIGN COMPARE SMOOTH_MIN SMOOTH_MAX ROUND FLOOR CEIL TRUNCATE FRACT MODULO WRAP SNAP PINGPONG SINE COSINE TANGENT ARCSINE
ARCCOSINE ARCTANGENT ARCTAN2 SINH COSH TANH RADIANS DEGREES""".split())
VMATH_OPS = set("""ADD SUBTRACT MULTIPLY DIVIDE MULTIPLY_ADD CROSS_PRODUCT PROJECT REFLECT REFRACT FACEFORWARD DOT_PRODUCT DISTANCE LENGTH
SCALE NORMALIZE ABSOLUTE POWER SIGN MINIMUM MAXIMUM FLOOR CEIL FRACTION MODULO WRAP SNAP SINE COSINE TANGENT""".split())
ENUMS = {("ShaderNodeTexImage", "interpolation"): {"Linear", "Closest", "Cubic", "Smart"},
         ("ShaderNodeTexImage", "extension"): {"REPEAT", "EXTEND", "CLIP", "MIRROR"},
         ("ShaderNodeMath", "operation"): MATH_OPS, ("ShaderNodeVectorMath", "operation"): VMATH_OPS,
         ("ShaderNodeTexNoise", "noise_dimensions"): {"1D", "2D", "3D", "4D"}}


class Socket:
    def __init__(self, node, name, is_output, index):
        self.node, self.name, self.is_output, self.index = node, name, is_output, index
        self.default_value = 0.0
        self.links = []

    def __repr__(self):
        return f"<Socket {self.node.bl_idname}.{'out' if self.is_output else 'in'}[{self.name}]>"


class Sockets:
    def __init__(self, node, names, is_output):
        self.node = node
        self.items = [Socket(node, n, is_output, i) for i, n in enumerate(names)]

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.items[k]
        for s in self.items:
            if s.name == k:
                return s
        raise KeyError(f"bpy_prop_collection[key]: key \"{k}\" not found on {self.node.bl_idname} "
                       f"(has {[s.name for s in self.items]})")

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class Node(Perm):
    def __init__(self, tree, idname):
        Perm.__init__(self, idname)
        d = self.__dict__
        d["tree"], d["bl_idname"] = tree, idname
        if idname in NODES:
            ins, outs = NODES[idname]
            d["inputs"], d["outputs"] = Sockets(self, ins, False), Sockets(self, outs, True)
        else:                                                                    # compositor / volume / group nodes: any socket
            d["inputs"], d["outputs"] = PermSockets(self, False), PermSockets(self, True)

    def __setattr__(self, k, v):
        allowed = ENUMS.get((self.bl_idname, k))
        if allowed is not None and v not in allowed:
            raise TypeError(f"{self.bl_idname}.{k}: enum \"{v}\" not found in {sorted(allowed)[:6]}...")
        object.__setattr__(self, k, v)


class PermSockets:
    def __init__(self, node, is_output):
        self.node, self.is_output, self.cache = node, is_output, {}

    def __getitem__(self, k):
        if k not in self.cache:
            self.cache[k] = Socket(self.node, str(k), self.is_output, len(self.cache))
        return self.cache[k]

    def __iter__(self):
        return iter(self.cache.values())

    def __len__(self):
        return len(self.cache)


class Links:
    def __init__(self, tree):
        self.tree, self.items = tree, []

    def new(self, a, b, **kw):
        if not (isinstance(a, Socket) and isinstance(b, Socket)):
            raise TypeError(f"links.new expects sockets, got {type(a).__name__}, {type(b).__name__}")
        if not a.is_output or b.is_output:
            raise RuntimeError(f"links.new direction: first must be an OUTPUT socket, second an INPUT socket "
                               f"({a!r} -> {b!r})")
        if a.node.tree is not self.tree or b.node.tree is not self.tree:
            raise RuntimeError("links.new: sockets of different node trees")
        a.links.append(b)
        b.links.append(a)
        self.items.append((a, b))
        return Perm("link")

    def __iter__(self):
        return iter(self.items)


class Nodes:
    def __init__(self, tree):
        self.tree, self.items = tree, []

    def new(self, idname):
        n = Node(self.tree, idname)
        self.items.append(n)
        return n

    def clear(self):
        self.items.clear()

    def get(self, name):
        return next((n for n in self.items if getattr(n, "name", None) == name), None)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class NodeTree(Perm):
    def __init__(self, name="tree"):
        Perm.__init__(self, name)
        d = self.__dict__
        d["nodes"], d["links"] = Nodes(self), Links(self)
        d["interface"] = Perm("interface")


# ====================================================================== meshes / images / objects
class _Arr:
    """foreach_get / foreach_set over a numpy array with Blender's size checking."""

    def __init__(self, owner, count, attrs):
        self.owner, self.count, self.attrs = owner, count, attrs

    def __len__(self):
        return self.count()

    def foreach_set(self, attr, flat):
        arr = np.asarray(flat)
        per = self.attrs[attr][0]
        if arr.size != per * self.count():
            raise RuntimeError(f"foreach_set({attr!r}): got {arr.size} values, expected {per}*{self.count()}")
        self.attrs[attr][1](arr.reshape(-1, per).astype(np.float64))

    def foreach_get(self, attr, out):
        per = self.attrs[attr][0]
        if out.size != per * self.count():
            raise RuntimeError(f"foreach_get({attr!r}): buffer {out.size}, expected {per}*{self.count()}")
        out[...] = self.attrs[attr][2]().reshape(-1)


class _Verts(_Arr):
    """me.vertices: foreach_set/get plus iteration / indexing of vertex proxies (`v.co`)."""

    def __iter__(self):
        return iter([self[i] for i in range(len(self))])

    def __getitem__(self, i):
        owner = self.owner
        return types.SimpleNamespace(co=Vector(owner._co[i]), index=int(i))


class UVLayer:
    def __init__(self, mesh, name):
        self.name, self.mesh = name, mesh
        self.uv = np.zeros((len(mesh._loop_vi), 2))
        self.data = _Arr(self, lambda: len(self.mesh._loop_vi), {"uv": (2, lambda a: setattr(self, "uv", a), lambda: self.uv)})


class UVLayers:
    def __init__(self, mesh):
        self.mesh, self.items = mesh, []

    def new(self, name="UV"):
        u = UVLayer(self.mesh, name)
        self.items.append(u)
        return u

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, k):
        return self.items[k] if isinstance(k, int) else next(u for u in self.items if u.name == k)


class ColAttr:
    def __init__(self, mesh, name, kind, domain):
        self.name, self.mesh, self.domain, self.kind = name, mesh, domain, kind
        n = lambda: len(mesh._co) if domain == "POINT" else len(mesh._loop_vi)  # noqa: E731
        self.col = np.zeros((n(), 4))
        self.data = _Arr(self, n, {"color": (4, lambda a: setattr(self, "col", a), lambda: self.col)})


class ColAttrs:
    def __init__(self, mesh):
        self.mesh, self.items = mesh, {}

    def new(self, name, type, domain):
        assert type in ("FLOAT_COLOR", "BYTE_COLOR"), type
        assert domain in ("POINT", "CORNER"), domain
        c = ColAttr(self.mesh, name, type, domain)
        self.items[name] = c
        return c

    def __getitem__(self, k):
        return self.items[k]


class Mesh(Perm):
    def __init__(self, name):
        Perm.__init__(self, name)
        d = self.__dict__
        d["name"], d["_co"], d["_faces"], d["_loop_vi"] = name, np.zeros((0, 3)), [], np.zeros(0, np.int32)
        d["materials"], d["uv_layers"], d["color_attributes"] = [], UVLayers(self), ColAttrs(self)
        d["vertices"] = _Verts(self, lambda: len(self._co), {"co": (3, lambda a: self.__dict__.__setitem__("_co", a), lambda: self._co)})
        d["loops"] = _Arr(self, lambda: len(self._loop_vi), {"vertex_index": (1, None, lambda: self._loop_vi)})
        d["polygons"] = Perm("polygons")

    def from_pydata(self, verts, edges, faces):
        co = np.array(verts, np.float64).reshape(-1, 3)
        fs = [tuple(int(i) for i in f) for f in faces]
        for f in fs:
            assert len(f) >= 3 and all(0 <= i < len(co) for i in f), f"bad face {f}"
        self.__dict__["_co"], self.__dict__["_faces"] = co, fs
        self.__dict__["_loop_vi"] = np.array([i for f in fs for i in f], np.int32)

    def update(self, *a, **k):
        return None

    def validate(self, *a, **k):
        return False

    def calc_normals(self):
        return None


class Image(Perm):
    def __init__(self, name, w, h, rgba=None):
        Perm.__init__(self, name)
        d = self.__dict__
        d["name"], d["size"] = name, (int(w), int(h))
        d["_px"] = np.zeros(w * h * 4, np.float32) if rgba is None else rgba
        d["colorspace_settings"] = Perm("cs")
        d["pixels"] = _Pixels(self)

    def update(self):
        return None


class _Pixels:
    def __init__(self, img):
        self.img = img

    def __len__(self):
        return self.img._px.size

    def foreach_get(self, out):
        if out.size != self.img._px.size:
            raise RuntimeError("pixels.foreach_get: size mismatch")
        out[...] = self.img._px

    def foreach_set(self, arr):
        arr = np.asarray(arr, np.float32).ravel()
        if arr.size != self.img._px.size:
            raise RuntimeError(f"pixels.foreach_set: got {arr.size}, expected {self.img._px.size}")
        self.img.__dict__["_px"] = arr.copy()


class Material(Perm):
    def __init__(self, name):
        Perm.__init__(self, name)
        self.__dict__["name"] = name
        self.__dict__["node_tree"] = NodeTree(name)
        self.__dict__["_use_nodes"] = False

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)


class DataBlock(Perm):
    def __init__(self, name, kind=None):
        Perm.__init__(self, name)
        self.__dict__["name"], self.__dict__["kind"] = name, kind


class Spline:
    def __init__(self):
        self.pts = np.zeros((1, 4))
        self.rad = np.ones(1)
        n = lambda: len(self.pts)  # noqa: E731
        self.points = _Points(self)


class _Points(_Arr):
    def __init__(self, sp):
        self.sp = sp
        _Arr.__init__(self, sp, lambda: len(sp.pts), {
            "co": (4, lambda a: setattr(sp, "pts", a), lambda: sp.pts),
            "radius": (1, lambda a: setattr(sp, "rad", a.ravel()), lambda: sp.rad)})

    def add(self, n):
        self.sp.pts = np.vstack([self.sp.pts, np.zeros((int(n), 4))])
        self.sp.rad = np.concatenate([self.sp.rad, np.ones(int(n))])

    def __getitem__(self, i):
        sp = self.sp
        o = Perm("pt")
        return o


class Curve(Perm):
    def __init__(self, name, kind="CURVE"):
        Perm.__init__(self, name)
        self.__dict__["name"], self.__dict__["splines"] = name, _Splines()
        self.__dict__["materials"] = []


class _Splines:
    def __init__(self):
        self.items = []

    def new(self, kind):
        assert kind in ("POLY", "BEZIER", "NURBS")
        s = Spline()
        self.items.append(s)
        return s

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class Object(Perm):
    def __init__(self, name, data=None):
        Perm.__init__(self, name)
        d = self.__dict__
        d["name"], d["data"] = name, data
        d["location"], d["rotation_euler"], d["scale"] = Vector(), Euler(), Vector((1, 1, 1))
        d["parent"] = None
        d["modifiers"] = _Modifiers()
        d["_children"] = []

    def __setattr__(self, k, v):
        if k == "location" or k == "scale":
            v = Vector(v)
        elif k == "rotation_euler" and not isinstance(v, Euler):
            v = Euler(v)
        elif k == "parent" and v is not None:
            v._children.append(self)
        object.__setattr__(self, k, v)

    @property
    def matrix_world(self):
        loc = Matrix.Translation(self.location)
        rot = self.rotation_euler.to_matrix()
        sc = Matrix(np.diag([*self.scale.v, 1.0]))
        m = loc @ rot @ sc
        return self.parent.matrix_world @ m if self.parent is not None else m

    def keyframe_insert(self, data_path, frame=None, index=-1, **kw):
        keys = self.__dict__.setdefault("_keys", {})
        keys[(data_path, frame)] = copy.deepcopy(getattr(self, data_path))
        return True


class _Modifiers:
    def __init__(self):
        self.items = []

    def new(self, name, kind):
        m = Perm(f"mod:{kind}")
        object.__setattr__(m, "kind", kind)
        object.__setattr__(m, "name", name)
        self.items.append(m)
        return m

    def __getitem__(self, k):
        for m in self.items:
            if m.name == k:
                return m
        raise KeyError(f'bpy_prop_collection[key]: key "{k}" not found')

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class Collection(Perm):
    def __init__(self, name="Collection"):
        Perm.__init__(self, name)
        self.__dict__["name"], self.__dict__["objects"], self.__dict__["children"] = name, _Linker(), _Linker()


class _Linker:
    def __init__(self):
        self.items = []

    def link(self, x):
        assert x not in self.items, "already linked"
        self.items.append(x)

    def unlink(self, x):
        self.items.remove(x)

    def __iter__(self):
        return iter(list(self.items))

    def __len__(self):
        return len(self.items)


class Registry:
    """bpy.data.<collection>"""

    def __init__(self, factory):
        self.factory, self.items = factory, []

    def new(self, name, *a, **kw):
        x = self.factory(name, *a, **kw)
        self.items.append(x)
        return x

    def remove(self, x, do_unlink=True):
        self.items.remove(x)

    def __iter__(self):
        return iter(list(self.items))

    def __len__(self):
        return len(self.items)

    def get(self, name):
        return next((i for i in self.items if i.name == name), None)


class ImagesRegistry(Registry):
    def __init__(self):
        Registry.__init__(self, lambda n, w, h, **kw: Image(n, w, h))

    def load(self, path, check_existing=False):
        import cv2
        if check_existing:
            for i in self.items:
                if getattr(i, "filepath", None) == str(path):
                    return i
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Error: Cannot read '{path}'")
        if im.ndim == 2:
            im = np.dstack([im, im, im])
        if im.shape[2] == 3:
            im = np.dstack([im, np.full(im.shape[:2], 255 if im.dtype == np.uint8 else 65535, im.dtype)])
        rgba = im[..., [2, 1, 0, 3]].astype(np.float32) / (255.0 if im.dtype == np.uint8 else 65535.0)
        h, w = rgba.shape[:2]
        img = Image(str(path), w, h, rgba[::-1].copy().reshape(-1))               # Blender: first row = BOTTOM of the picture
        img.__dict__["filepath"] = str(path)
        self.items.append(img)
        return img


class Scene(Perm):
    def __init__(self):
        Perm.__init__(self, "scene")
        d = self.__dict__
        d["collection"] = Collection("Scene Collection")
        d["render"], d["cycles"], d["eevee"] = Perm("render"), Perm("cycles"), Perm("eevee")
        vs = Perm("view_settings")
        vs.bl_rna.properties["view_transform"] = types.SimpleNamespace(enum_items=[types.SimpleNamespace(identifier=i) for i in
                                                                                  ("Standard", "Khronos PBR Neutral", "AgX", "Filmic", "Raw")])
        d["view_settings"], d["display_settings"] = vs, Perm("display_settings")
        d["frame_current"], d["frame_start"], d["frame_end"] = 1, 1, 1
        d["camera"], d["world"] = None, None
        d["view_layers"] = [Perm("view_layer")]
        d["use_nodes"], d["node_tree"], d["compositing_node_group"] = False, None, None
        d["_frame_log"] = []

    def __setattr__(self, k, v):
        if k == "use_nodes" and v:
            object.__setattr__(self, "node_tree", NodeTree("compositor"))
        object.__setattr__(self, k, v)

    def frame_set(self, f, subframe=0.0):
        self.__dict__["frame_current"] = int(f)
        self._frame_log.append(int(f))

    @property
    def all_objects(self):
        out, stack = [], [self.collection]
        while stack:
            c = stack.pop()
            out += list(c.objects)
            stack += list(c.children)
        return out


# ====================================================================== bpy module
def install(monkeypatch=None):
    """Create fresh `bpy` + `mathutils` modules and register them in sys.modules. Returns the bpy module."""
    CALLS.clear()
    bpy = types.ModuleType("bpy")
    bpy._calls = CALLS
    bpy.app = types.SimpleNamespace(version=(4, 2, 0), version_string="4.2.0")
    scene = Scene()
    ctx = types.SimpleNamespace(scene=scene, active_object=None, view_layer=Perm("view_layer"), preferences=Perm("prefs"))
    ctx.view_layer.objects.active = None
    bpy.context = ctx
    data = types.SimpleNamespace(
        objects=Registry(Object), meshes=Registry(Mesh), materials=Registry(Material), cameras=Registry(DataBlock),
        lights=Registry(DataBlock), curves=Registry(Curve), worlds=Registry(lambda n: _World(n)), images=ImagesRegistry(),
        node_groups=Registry(lambda n, kind="ShaderNodeTree": NodeTree(n)), collections=Registry(Collection))
    data.objects.items.clear()
    bpy.data = data
    engine_items = [types.SimpleNamespace(identifier=i) for i in ("BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH", "CYCLES")]
    bpy.types = types.SimpleNamespace(RenderSettings=types.SimpleNamespace(
        bl_rna=types.SimpleNamespace(properties={"engine": types.SimpleNamespace(enum_items=engine_items)})))
    bpy.ops = _Ops(bpy)
    mu = types.ModuleType("mathutils")
    mu.Vector, mu.Matrix, mu.Euler = Vector, Matrix, Euler
    sys.modules["bpy"], sys.modules["mathutils"] = bpy, mu
    return bpy


class _World(Perm):
    def __init__(self, name):
        Perm.__init__(self, name)
        self.__dict__["name"] = name
        self.__dict__["node_tree"] = NodeTree(name)
        self.__dict__["node_tree"].nodes.new("ShaderNodeBackground").name = "Background"
        self.__dict__["color"] = (0, 0, 0)


class _Ops:
    """bpy.ops.*: primitives create an active object; render writes a PNG so the scene script's resume logic sees a frame."""

    def __init__(self, bpy):
        self.bpy = bpy
        self.mesh = types.SimpleNamespace(**{n: self._prim(n) for n in ("primitive_ico_sphere_add", "primitive_cube_add",
                                                                          "primitive_grid_add", "primitive_plane_add",
                                                                          "primitive_uv_sphere_add")})
        self.render = types.SimpleNamespace(render=self._render)
        self.object = types.SimpleNamespace(select_all=self._noop("object.select_all"), quick_smoke=self._smoke,
                                            effector_add=self._effector)
        self.wm = types.SimpleNamespace(save_as_mainfile=self._noop("wm.save_as_mainfile"), usd_export=self._noop("wm.usd_export"))
        self.file = types.SimpleNamespace(make_paths_relative=self._noop("file.make_paths_relative"))
        self.rendered = []

    def _noop(self, name):
        def f(*a, **kw):
            CALLS.append((name, a, kw))
            return {"FINISHED"}
        return f

    def _make(self, name, loc=(0, 0, 0)):
        me = self.bpy.data.meshes.new(name)
        me.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
        ob = self.bpy.data.objects.new(name, me)
        ob.location = loc
        self.bpy.context.scene.collection.objects.link(ob)
        self.bpy.context.active_object = ob
        return ob

    def _prim(self, name):
        def f(*a, location=(0, 0, 0), **kw):
            CALLS.append(("mesh." + name, a, dict(kw, location=location)))
            self._make(name, location)
            return {"FINISHED"}
        return f

    def _smoke(self, *a, **kw):
        CALLS.append(("object.quick_smoke", a, kw))
        dom = self._make("Smoke Domain")
        dom.modifiers.new("Fluid", "FLUID")
        return {"FINISHED"}

    def _effector(self, *a, location=(0, 0, 0), **kw):
        CALLS.append(("object.effector_add", a, kw))
        self._make("Wind", location)
        return {"FINISHED"}

    def _render(self, *a, write_still=False, **kw):
        import cv2
        r = self.bpy.context.scene.render
        path = str(r.filepath)
        self.rendered.append(path)
        if write_still:
            rng = np.random.default_rng(len(self.rendered))
            cv2.imwrite(path, (rng.random((36, 64, 3)) * 255).astype(np.uint8))
        return {"FINISHED"}


def uninstall():
    for k in ("bpy", "mathutils", "mf_blender_fx", "mf_blender_extras", "mf_camera_paths", "mf_plate_geometry"):
        sys.modules.pop(k, None)
