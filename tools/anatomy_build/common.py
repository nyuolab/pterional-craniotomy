"""Shared helpers for the anatomy build steps.

Records are the dicts produced by ``tools/anatomy.py unpack``. Each record's
``data`` is base64 little-endian float32, six floats per vertex (XYZ position,
XYZ normal), three vertices per triangle, in display coordinates:

* +X is the patient's left (the right side is negative X),
* +Y is superior,
* +Z is anterior.

One display unit is 2 cm (native metres x 50).

Normals must point to the outer side of a surface (out of a solid, or toward
the surgeon for an open sheet). Build steps write unit normals; the final
ambient-occlusion step (``ao.py``) stores occlusion in the normal length.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"
CHUNKS = [ASSETS / f"anatomy-{i:02}.bin" for i in range(1, 7)]
UNIT_CM = 2.0  # one display unit in centimetres


# ---------------------------------------------------------------- records I/O
def load_records(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_records(records: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")


def load_packed(chunks: list[Path] = CHUNKS) -> list[dict]:
    return json.loads(gzip.decompress(b"".join(c.read_bytes() for c in chunks)))


def pack(records: list[dict], chunks: list[Path] = CHUNKS) -> int:
    """Write records as gzip JSON split evenly over the chunk files. Returns bytes written."""
    data = gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    for i, chunk in enumerate(chunks):
        chunk.write_bytes(data[len(data) * i // len(chunks): len(data) * (i + 1) // len(chunks)])
    return len(data)


def find(records: list[dict], *, id: str | None = None, name: str | None = None,
         native: str | None = None) -> dict:
    for m in records:
        if (id is not None and m["id"] == id) or (name is not None and m["name"] == name) or (
                native is not None and m.get("nativeName") == native):
            return m
    raise KeyError(id or name or native)


def decode(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, normals) as float64 arrays of shape (count, 3)."""
    f = np.frombuffer(base64.b64decode(m["data"]), dtype="<f4").reshape(-1, 6).astype(np.float64)
    return f[:, :3].copy(), f[:, 3:].copy()


def encode(m: dict, pos: np.ndarray, nrm: np.ndarray) -> None:
    """Store unindexed triangle vertices and refresh count, bounds and anchor."""
    pos = np.asarray(pos, dtype=np.float64)
    nrm = np.asarray(nrm, dtype=np.float64)
    assert pos.shape == nrm.shape and pos.ndim == 2 and pos.shape[1] == 3 and len(pos) % 3 == 0
    assert np.isfinite(pos).all() and np.isfinite(nrm).all()
    arr = np.concatenate([pos, nrm], axis=1).astype("<f4")
    m["data"] = base64.b64encode(arr.tobytes()).decode("ascii")
    m["count"] = int(len(pos))
    lo, hi = pos.min(0), pos.max(0)
    m["bounds"] = [lo.tolist(), hi.tolist()]
    m["anchor"] = ((lo + hi) / 2).tolist()


# ------------------------------------------------------------- mesh topology
def weld(pos: np.ndarray, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices. Returns (V, F) with F of shape (T, 3)."""
    key = np.round(pos / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return pos[first].copy(), inv.reshape(-1, 3)


def face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Unnormalised (area-weighted) face normals."""
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = face_normals(V, F)
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return normalize(n)


def normalize(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(length, 1e-12)


def unweld(V: np.ndarray, F: np.ndarray, N: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if N is None:
        N = vertex_normals(V, F)
    idx = F.reshape(-1)
    return V[idx].copy(), N[idx].copy()


def orient_outward(V: np.ndarray, F: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Flip faces whose normal points toward ``centre`` (for convex-ish shells/sheets)."""
    fn = face_normals(V, F)
    c = V[F].mean(1)
    flip = np.einsum("ij,ij->i", fn, c - centre) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def mirror_x(pos: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror across the midsagittal plane, keeping triangle winding consistent with normals."""
    s = np.array([-1.0, 1.0, 1.0])
    p = (pos * s).reshape(-1, 3, 3)[:, [0, 2, 1]].reshape(-1, 3)
    n = (nrm * s).reshape(-1, 3, 3)[:, [0, 2, 1]].reshape(-1, 3)
    return p, n


# ---------------------------------------------------------------- ray casting
def ray_nearest(orig: np.ndarray, dirs: np.ndarray, V: np.ndarray, F: np.ndarray,
                tmin: float = 1e-6, tmax: float = np.inf, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Nearest ray-triangle hit (Moller-Trumbore, both faces).

    Returns (t, tri) where t is inf and tri is -1 for misses. Cost is rays x
    triangles, so restrict F to the region of interest for large meshes.
    """
    orig = np.atleast_2d(orig).astype(np.float64)
    dirs = np.atleast_2d(dirs).astype(np.float64)
    if len(orig) == 1 and len(dirs) > 1:
        orig = np.repeat(orig, len(dirs), 0)
    v0 = V[F[:, 0]]
    e1 = V[F[:, 1]] - v0
    e2 = V[F[:, 2]] - v0
    best_t = np.full(len(orig), np.inf)
    best_i = np.full(len(orig), -1, dtype=np.int64)
    for s in range(0, len(orig), chunk):
        o = orig[s:s + chunk, None, :]
        d = dirs[s:s + chunk, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum("rtk,tk->rt", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[None]
        u = np.einsum("rtk,rtk->rt", tv, p) * inv
        q = np.cross(tv, e1[None])
        v = np.einsum("rtk,rtk->rt", d, q) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin) & (t < tmax)
        t = np.where(hit, t, np.inf)
        i = t.argmin(1)
        best_t[s:s + chunk] = t[np.arange(len(i)), i]
        best_i[s:s + chunk] = np.where(np.isfinite(best_t[s:s + chunk]), i, -1)
    return best_t, best_i


def triangles_near(V: np.ndarray, F: np.ndarray, lo, hi, pad: float = 0.0) -> np.ndarray:
    """Subset of F whose bounding boxes intersect the box [lo-pad, hi+pad]."""
    lo = np.asarray(lo) - pad
    hi = np.asarray(hi) + pad
    tri = V[F]
    keep = (tri.max(1) >= lo).all(1) & (tri.min(1) <= hi).all(1)
    return F[keep]


# ------------------------------------------------------------ generated shapes
def grid_to_tris(P: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate an (H, W, 3) grid of points. Quads with any masked-out corner are skipped."""
    H, W, _ = P.shape
    if mask is None:
        mask = np.isfinite(P).all(-1)
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    ok = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    F = np.concatenate([np.stack([a, c, b], -1)[ok], np.stack([b, c, d], -1)[ok]])
    V = np.where(mask[..., None], P, 0.0).reshape(-1, 3)
    used = np.unique(F)
    remap = np.full(H * W, -1)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def tube(points: np.ndarray, radius, sides: int = 12, caps: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Unindexed triangles for a tube along a polyline (parallel-transport frames).

    ``radius`` may be a scalar or one value per point (for tapering).
    """
    P = np.asarray(points, dtype=np.float64)
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(P),))
    T = np.gradient(P, axis=0)
    T = normalize(T)
    ref = np.array([0.0, 1.0, 0.0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = [normalize(np.cross(T[0], ref))]
    for i in range(1, len(P)):
        n = N[-1] - T[i] * np.dot(N[-1], T[i])
        N.append(normalize(n))
    N = np.array(N)
    B = np.cross(T, N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring_n = np.cos(ang)[None, :, None] * N[:, None] + np.sin(ang)[None, :, None] * B[:, None]
    ring_p = P[:, None] + r[:, None, None] * ring_n
    V = ring_p.reshape(-1, 3)
    Nv = ring_n.reshape(-1, 3)
    faces = []
    for i in range(len(P) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            faces += [[a, b, c], [b, d, c]]
    F = np.array(faces)
    # orient so the geometric normal agrees with the radial normal
    if np.einsum("ij,ij->i", face_normals(V, F), Nv[F].sum(1)).mean() < 0:
        F = F[:, [0, 2, 1]]
    pos, nrm = V[F.reshape(-1)], Nv[F.reshape(-1)]
    if caps:
        extra_p, extra_n = [], []
        for end, sign in ((0, -1.0), (len(P) - 1, 1.0)):
            ring = ring_p[end]
            cn = T[end] * sign
            for j in range(sides):
                tri = [P[end], ring[j], ring[(j + 1) % sides]]
                if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), cn) < 0:
                    tri = [tri[0], tri[2], tri[1]]
                extra_p += tri
                extra_n += [cn] * 3
        pos = np.concatenate([pos, np.array(extra_p)])
        nrm = np.concatenate([nrm, np.array(extra_n)])
    return pos, nrm


def catmull_rom(points: np.ndarray, samples_per_segment: int = 12) -> np.ndarray:
    """Smooth centripetal-ish Catmull-Rom resampling through control points."""
    P = np.asarray(points, dtype=np.float64)
    P = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def structure_entry(m: dict) -> dict:
    """Manifest ``structures`` entry for a record (everything except vertex data)."""
    return {k: v for k, v in m.items() if k != "data"}
