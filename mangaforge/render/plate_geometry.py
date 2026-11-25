"""One camera/plate model for BOTH engines (Blender mesh and the cv2 fallback) - pure numpy, no bpy.

Frame coordinates: u right, v down, origin at the frame centre, u in frame WIDTHS, v in frame HEIGHTS (both within +-0.5).
A plate pixel at depth value d (0 = far .. 1 = near) sits at relative distance  zeta = 1 - parallax * (d - d_subject)
(1.0 on the subject plane). It is placed so that, from the un-moved camera, it projects exactly where the flat image would
put it - so with no camera move the render equals the artwork and every move produces genuine parallax:

    u' = (u0 * zeta - dx) / (zeta - dz)        v' = (v0 * zeta + dy_h) / (zeta - dz)           (dy_h = dy * 16/9, +y = camera up)

dx, dy are fractions of the frame width at the subject plane, dz a fraction of the subject distance (camera_paths units).
"""
from __future__ import annotations

import math

import numpy as np

ASPECT = 16 / 9
SENSOR_W = 36.0                       # mm, full-frame sensor width (Blender's default sensor fit is horizontal)
D_REF = 10.0                          # world distance from the camera to the subject plane (matches SPEC ctx.D_REF)


def zeta(depth01, subject_depth, parallax=0.35):
    """Relative camera distance of a pixel (array or float)."""
    return 1.0 - parallax * (np.asarray(depth01, dtype=np.float64) - subject_depth)


def frame_width_at_ref(focal_mm):
    """World width of the visible frame at the subject plane (D_REF) for a focal length."""
    return D_REF * SENSOR_W / focal_mm


def orbit_to_dx(orbit_deg, focal_mm=35.0):
    """Lateral offset (frame widths at the subject plane) of orbiting the subject by `orbit_deg`."""
    return math.sin(math.radians(orbit_deg)) * D_REF / frame_width_at_ref(focal_mm)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def subject_world(subject_uv, focal_mm=35.0):
    """World position of the subject point: subject_uv = (u0, v0) in frame coords at the rest camera (u widths right, v heights down)."""
    wf0 = frame_width_at_ref(focal_mm)
    return np.array([subject_uv[0] * wf0, D_REF, -subject_uv[1] * wf0 / ASPECT])


def camera_pose_world(pose, subject_w, focal_mm=35.0):
    """THE camera model (Blender builds its camera from this; the cv engine ray-casts through it).

    World axes: x right, y away from the camera at rest, z up. A camera_paths pose becomes a position C (truck / pedestal / dolly,
    then an orbit = rotation of that position about the vertical axis through the subject) and a rotation R whose columns are the
    camera's right / up / back axes (Blender convention: it looks along -R[:,2]): orbit and yaw about the vertical axis, pitch about
    the camera's right axis, roll about its view axis. Returns (R 3x3, C 3).
    """
    wf0 = frame_width_at_ref(focal_mm)
    T = np.array([pose.get("dx", 0.0) * wf0, pose.get("dz", 0.0) * D_REF, pose.get("dy", 0.0) * wf0])
    phi = math.radians(pose.get("orbit_deg", 0.0))
    Rz = _rz(phi)
    S = np.asarray(subject_w, float)
    C = S + Rz @ (T - S) if abs(phi) > 1e-9 else T
    R = (Rz @ _rz(math.radians(pose.get("yaw", 0.0))) @ _rx(math.radians(90.0 + pose.get("pitch", 0.0)))
         @ _rz(math.radians(pose.get("roll", 0.0))))
    return R, C


def project(u0, v0, z, pose, focal_mm=35.0, subject_uv=(0.0, 0.0)):
    """Exact pinhole projection of plate points (frame coords u0, v0 at the rest camera, relative depth z) through a camera_paths pose.
    -> (u, v) in frame units (u widths right, v heights down). `focal_scale` is a lens change, as in the Blender scene."""
    wf0 = frame_width_at_ref(focal_mm)
    R, C = camera_pose_world(pose, subject_world(subject_uv, focal_mm), focal_mm)
    u0, v0, z = np.broadcast_arrays(np.asarray(u0, float), np.asarray(v0, float), np.asarray(z, float))
    P = np.stack([u0 * wf0 * z, z * D_REF, -v0 * wf0 / ASPECT * z], -1)
    pc = (P - C) @ R                                                 # rows: R^T (P - C)
    k = focal_mm * pose.get("focal_scale", 1.0) / SENSOR_W
    depth = np.maximum(-pc[..., 2], 1e-3)
    return k * pc[..., 0] / depth, -k * pc[..., 1] / depth * ASPECT


def hit_plate(u, v, z, R, C, focal_mm, focal_scale=1.0):
    """Inverse of `project` for the cv engine: which plate point (u0, v0) shows at output pixel (u, v) if it sits at relative depth z
    (z may be an array: the plate point at depth z lies on the world plane y = z * D_REF, so it is a ray / plane intersection)."""
    wf0 = frame_width_at_ref(focal_mm)
    k = focal_mm * focal_scale / SENSOR_W
    xc, yc = u / k, -v / (k * ASPECT)
    dx = R[0, 0] * xc + R[0, 1] * yc - R[0, 2]
    dy = R[1, 0] * xc + R[1, 1] * yc - R[1, 2]
    dz = R[2, 0] * xc + R[2, 1] * yc - R[2, 2]
    t = (z * D_REF - C[1]) / np.maximum(dy, 1e-3)
    return (C[0] + t * dx) / (wf0 * z), -(C[2] + t * dz) / (wf0 / ASPECT * z)


def mesh_grid(depth_u8, subject_depth, parallax=0.35, nx=192, ny=108, jump=0.18):
    """Vertex/face data for the depth mesh (consumed by blender_scene; testable without bpy).

    Returns dict(u0, v0, zeta, uv, faces) where u0/v0 are the reference frame coords of the plate at dz=dx=0 and zeta the
    relative distance per vertex; faces are quads that do NOT bridge a depth discontinuity (> `jump` in zeta).
    """
    h, w = depth_u8.shape
    xs = np.linspace(0, 1, nx + 1)
    ys = np.linspace(0, 1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    sx = np.clip((X * (w - 1)).round().astype(int), 0, w - 1)
    sy = np.clip((Y * (h - 1)).round().astype(int), 0, h - 1)
    d = depth_u8[sy, sx].astype(np.float64) / 255.0
    z = zeta(d, subject_depth, parallax)
    faces = []
    idx = lambda j, i: j * (nx + 1) + i  # noqa: E731
    for j in range(ny):
        for i in range(nx):
            q = (z[j, i], z[j, i + 1], z[j + 1, i + 1], z[j + 1, i])
            if max(q) - min(q) > jump:
                continue
            faces.append((idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)))
    return {"x": X, "y": Y, "zeta": z, "faces": faces, "nx": nx, "ny": ny}
