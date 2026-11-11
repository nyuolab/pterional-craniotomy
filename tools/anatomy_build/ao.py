#!/usr/bin/env python3
"""Bake per-vertex ambient occlusion into the length of each vertex normal.

This is the LAST build step. It works on any records generically: for every
vertex it estimates how much of the outward hemisphere is blocked by nearby
geometry and stores ``ao`` in [AO_MIN, 1] as the normal length (1 = open).
The renderer reads ``ao = length(aNormal)``; rotations preserve length.

Method
------
* Occluders for each receiving group are voxelised (surface rasterisation at
  VOXEL = 0.025 units = 0.5 mm) inside the set's bounds, plus a 3x3x3-dilated
  copy used for the far part of each ray so thin shells cannot be skipped.
* For each unique (position, normal) pair -- vertices are heavily duplicated in
  the unindexed triangle data -- RAYS cosine-weighted hemisphere directions
  (Hammersley set, rotated about the normal by a per-vertex hash so banding
  becomes fine noise) are marched from an origin pushed 2 voxels along the
  normal (no self-shadowing acne on flat surfaces) out to the set's reach R.
  A first hit at distance t blocks ``1 - t/R`` of that direction.
* AO is sampled at every vertex AND at four points inside every triangle
  larger than two voxels. Per-vertex values are then fitted by least squares
  so that their linear interpolation across each triangle matches the samples
  (this keeps a crease at one corner of a big triangle from smearing grey over
  the whole triangle, and averages out sampling noise). Finally
  ao = clip(fit ** GAMMA, AO_MIN, 1).

Occluder sets (receiver <- occluders)
-------------------------------------
* bone <- bone; brain <- brain; muscle <- muscle.
* arteries, veins, nerves <- brain + arteries + veins + nerves (vessels lying in
  sulci and cisterns darken; bone is NOT an occluder because bone is cut away
  in the shader during the procedure and the vessels must not keep a shadow of
  a clinoid or flap that has been removed).
* dura, fascia, deepfascia, fat, arachnoid, skin, galea, pericranium <- the
  same RECORD only. These are thin sheets that the renderer displaces,
  reflects, cuts or separates (``sep=1``), so occlusion between different
  sheets or by the bone/muscle under them would be wrong in some step. The
  scalp layers (``scalp.py``) are the clearest case: the galea's outer face and
  the pericranium and bone beneath it are only ever seen once the layer above
  has been peeled back or the layers are separated, and the reflected flaps
  show their deep faces, so a rest-pose shadow cast by the layer above (or by
  the bone onto the pericranium's deep face) would darken exactly the surfaces
  the guide exposes. While the scalp is closed only its outer skin surface is
  visible, which nothing occludes. Self-occlusion still darkens each slab's own
  creases and cut walls. Unknown future groups get the same per-record
  treatment.
* annotations, instruments: no AO (exact unit normals).

Orientation: occlusion is gathered on the side the normal points to, so closed
records whose normals point INTO the solid (found by casting rays along +n and
-n against the record itself) get their normals and winding flipped first.
In the base data that is one side of most mirrored vessels/nerves and the old
illustrative discs; the summary lists every flipped id. Zero-length normals
(degenerate triangles) are repaired. Records that already carry AO (non-unit
normals) make the step raise, so it cannot be applied twice.

contextOnly records are included as occluders. They are only drawn in the
full-head context, but in practice they lie either beneath the visible surface
(deep brain nuclei, the medial surfaces of the parcels, the mandible under the
zygomatic arch) or continue the visible surface (neighbouring gyri, other
skull bones), so they add the correct sulcal/sutural darkening without
shadowing anything that is exposed in the default view.

Usage:  python3 tools/anatomy_build/ao.py IN.json OUT.json
"""
from __future__ import annotations

import base64
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import encode, load_records, save_records  # noqa: E402

VOXEL = 0.025          # occupancy grid spacing (display units; 0.5 mm)
RAYS = 40              # hemisphere directions per vertex
OFFSET = 2.0 * VOXEL   # ray origin pushed along the normal (>= 1.5 voxels)
NEAR = 6.0 * VOXEL     # beyond this distance rays test the dilated grid
AO_MIN = 0.3
GAMMA = 1.0
WORKERS = int(os.environ.get("AO_WORKERS", min(16, os.cpu_count() or 1)))  # 1 = no fork pool

VESSEL_OCCLUDERS = ("brain", "arteries", "veins", "nerves")
GROUP_SETS = {  # receiving group -> (occluder groups, reach R in display units)
    "bone": (("bone",), 0.60),
    "brain": (("brain",), 0.50),
    "arteries": (VESSEL_OCCLUDERS, 0.40),
    "veins": (VESSEL_OCCLUDERS, 0.40),
    "nerves": (VESSEL_OCCLUDERS, 0.40),
    "muscle": (("muscle",), 0.45),
}
SELF_REACH = 0.40      # per-record sets (sheets, including the scalp layers, and unknown groups)
NO_AO = {"annotations", "instruments"}


# ------------------------------------------------------------------ helpers
def _decode32(m: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(m["data"]), dtype="<f4").reshape(-1, 6)


def check_and_repair(records: list[dict], arrays: list[np.ndarray]) -> dict:
    """Raise if AO was already applied; repair zero/NaN normals of degenerate vertices."""
    already, repaired = 0, 0
    for m, a in zip(records, arrays):
        n = a[:, 3:].astype(np.float64)
        length = np.linalg.norm(n, axis=1)
        bad = ~np.isfinite(length) | (length < 1e-4)
        already += int(((length > 0.25) & (np.abs(length - 1.0) > 1e-3)).sum())
        if bad.any():
            # use the mean of the triangle's other normals, else its face normal, else +Y
            tri = n.reshape(-1, 3, 3)
            tb = bad.reshape(-1, 3)
            good = np.where(tb[..., None], 0.0, np.nan_to_num(tri))
            fill = good.sum(1)
            p = a[:, :3].astype(np.float64).reshape(-1, 3, 3)
            fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            use_fn = np.linalg.norm(fill, axis=1) < 1e-9
            fill[use_fn] = fn[use_fn]
            still = np.linalg.norm(fill, axis=1) < 1e-12
            fill[still] = (0.0, 1.0, 0.0)
            fill /= np.linalg.norm(fill, axis=1, keepdims=True)
            rows = np.nonzero(bad)[0]
            a[rows, 3:] = fill[rows // 3]
            repaired += len(rows)
    if already:
        raise RuntimeError(f"{already} vertex normals are already non-unit: ambient occlusion "
                           "seems to have been applied; run ao.py on unit-normal records only")
    return {"degenerateNormalsRepaired": repaired}


ORIENT_SAMPLES = 48


def _orientation(k: int) -> str:
    """'in' if the record is a closed solid whose normals point into it, else 'ok'.

    Rays are cast from sample vertices along +n and -n against the record's own
    triangles. For a solid with outward normals almost every -n ray hits the far
    wall and few +n rays do; inward normals give the reverse; open sheets and
    tubes without caps hit little either way and are left alone.
    """
    from common import ray_nearest, weld
    a = _G["arrays"][k]
    pos = a[:, :3].astype(np.float64)
    nrm = a[:, 3:].astype(np.float64)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    V, F = weld(pos)
    rng = np.random.default_rng(zlib.crc32(_G["ids"][k].encode()))
    idx = rng.choice(len(pos), min(ORIENT_SAMPLES, len(pos)), replace=False)
    eps = 1e-4
    tp, _ = ray_nearest(pos[idx] + eps * nrm[idx], nrm[idx], V, F, chunk=8)
    tn, _ = ray_nearest(pos[idx] - eps * nrm[idx], -nrm[idx], V, F, chunk=8)
    hp, hn = np.isfinite(tp).mean(), np.isfinite(tn).mean()
    return "in" if hp > 0.8 and hn < 0.25 else "ok"


def fix_orientation(records: list[dict], arrays: list[np.ndarray]) -> list[str]:
    """Flip normals and winding of closed records whose normals point inward.

    The atlas data has this for many mirrored vessels/nerves and for the old
    illustrative discs. The previous renderers lit both faces alike so it was
    invisible, but occlusion is gathered on the side the normal points to, so
    an inward normal would read as almost fully occluded. Air-sinus cavities in
    the bone group are skipped: their normals point into the air, i.e. out of
    the bone, which is already correct.
    """
    todo = [k for k, m in enumerate(records) if m["group"] not in NO_AO
            and not (m["group"] == "bone" and "sinus" in m["name"].lower())]
    _G["arrays"] = arrays
    _G["ids"] = [m["id"] for m in records]
    if WORKERS > 1:
        with mp.get_context("fork").Pool(WORKERS) as pool:
            res = pool.map(_orientation, todo, chunksize=1)
    else:
        res = [_orientation(k) for k in todo]
    _G.clear()
    flipped = []
    for k, r in zip(todo, res):
        if r == "in":
            a = arrays[k]
            a[:, 3:] *= -1.0
            arrays[k] = a.reshape(-1, 3, 6)[:, [0, 2, 1]].reshape(-1, 6).copy()
            flipped.append(records[k]["id"])
    return flipped


def voxelize(tri_sets: list[np.ndarray], lo: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Rasterise triangle soups (each (T,3,3)) into a flat uint8 occupancy grid."""
    grid = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    sy, sz = shape[1] * shape[2], shape[2]
    spacing = 0.5 * VOXEL
    for tris in tri_sets:
        tris = ((tris - lo) / VOXEL).astype(np.float32)  # voxel units
        e = np.stack([tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1], tris[:, 0] - tris[:, 2]], 1)
        longest = np.sqrt((e * e).sum(-1)).max(1) * VOXEL
        nsub = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
        for k in np.unique(nsub):
            sel = tris[nsub == k]
            i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
            keep = (i + j) <= k
            bu = (i[keep] / k).astype(np.float32)
            bv = (j[keep] / k).astype(np.float32)
            step = max(1, 4_000_000 // len(bu))
            for s in range(0, len(sel), step):
                t = sel[s:s + step]
                v0, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
                pts = v0[:, None] + bu[None, :, None] * e1[:, None] + bv[None, :, None] * e2[:, None]
                ijk = pts.reshape(-1, 3).astype(np.int32)
                grid[ijk[:, 0] * sy + ijk[:, 1] * sz + ijk[:, 2]] = 1
    return grid


def dilate(grid: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    from scipy.ndimage import maximum_filter
    return maximum_filter(grid.reshape(shape), size=3, mode="constant").reshape(-1)


def hemisphere() -> np.ndarray:
    """Cosine-weighted Hammersley directions in the local frame (z = normal)."""
    i = np.arange(RAYS)
    u1 = (i + 0.5) / RAYS
    u2 = np.array([int(bin(k)[2:][::-1], 2) / 2 ** len(bin(k)[2:]) if k else 0.0 for k in i])
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], 1)


def schedule(reach: float) -> np.ndarray:
    ts, t = [], 0.0
    while t < reach:
        ts.append(t)
        t += 0.75 * VOXEL if t < NEAR else 1.25 * VOXEL
    return np.array(ts, dtype=np.float32)


def pos_hash(p: np.ndarray) -> np.ndarray:
    q = np.round(p * 2000.0).astype(np.int64)
    h = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    h = (h ^ (h >> 13)) * 0x5BD1E995
    return ((h ^ (h >> 15)) & 0xFFFFFF).astype(np.float64) / float(0x1000000)


# ------------------------------------------------------------ worker process
_G: dict = {}


def _march(span: tuple[int, int]) -> np.ndarray:
    s, e = span
    P = _G["P"][s:e]
    N = _G["N"][s:e]
    lo, shape, reach = _G["lo"], _G["shape"], _G["reach"]
    raw, dil, ts, local = _G["raw"], _G["dil"], _G["ts"], _G["dirs"]
    nv = len(P)
    # tangent frame rotated about the normal by a per-vertex angle
    a = np.where(np.abs(N[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    T = np.cross(a, N)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.cross(N, T)
    h = pos_hash(P)
    ang = 2 * np.pi * h
    c, si = np.cos(ang)[:, None], np.sin(ang)[:, None]
    lx = c * local[None, :, 0] - si * local[None, :, 1]
    ly = si * local[None, :, 0] + c * local[None, :, 1]
    lz = np.broadcast_to(local[None, :, 2], lx.shape)
    D = (lx[..., None] * T[:, None] + ly[..., None] * B[:, None] + lz[..., None] * N[:, None])
    D = (D / VOXEL).reshape(-1, 3).astype(np.float32)                    # voxel units per unit t
    O = ((P + OFFSET * N - lo) / VOXEL).astype(np.float32)
    O = np.repeat(O, RAYS, axis=0)
    jit = np.repeat(np.modf(h * 7919.0)[0], RAYS).astype(np.float32)   # per-vertex start jitter
    sy, sz = np.int32(shape[1] * shape[2]), np.int32(shape[2])
    first = np.full(len(O), np.inf, dtype=np.float32)
    alive = np.arange(len(O))
    for j, t0 in enumerate(ts):
        step = (ts[j + 1] - t0) if j + 1 < len(ts) else (reach - t0)
        t = t0 + jit[alive] * step
        q = O[alive] + D[alive] * t[:, None]
        ijk = q.astype(np.int32)
        idx = ijk[:, 0] * sy + ijk[:, 1] * sz + ijk[:, 2]
        hit = (raw if t0 < NEAR else dil)[idx].astype(bool)
        if hit.any():
            first[alive[hit]] = t[hit]
            alive = alive[~hit]
            if not len(alive):
                break
    blocked = np.clip(1.0 - first / reach, 0.0, 1.0)  # inf -> 0
    return (1.0 - blocked.reshape(nv, RAYS).mean(1)).astype(np.float32)


# --------------------------------------------------------------- AO for a set
def bake(occ_tris: list[np.ndarray], P: np.ndarray, N: np.ndarray, reach: float,
         pool_ok: bool = True) -> tuple[np.ndarray, dict]:
    """AO for receiver vertices P/N (unique) against occluder triangle soups."""
    pad = OFFSET + reach + 3 * VOXEL
    lo = np.minimum(np.min([t.reshape(-1, 3).min(0) for t in occ_tris], 0), P.min(0)) - pad
    hi = np.maximum(np.max([t.reshape(-1, 3).max(0) for t in occ_tris], 0), P.max(0)) + pad
    shape = tuple(int(x) for x in np.ceil((hi - lo) / VOXEL) + 1)
    raw = voxelize(occ_tris, lo, shape)
    dil = dilate(raw, shape)
    order = np.lexsort(((P[:, 2] / 0.2).astype(int), (P[:, 1] / 0.2).astype(int), (P[:, 0] / 0.2).astype(int)))
    _G.update(P=P[order], N=N[order], lo=lo, shape=shape, reach=reach, raw=raw, dil=dil,
              ts=schedule(reach), dirs=hemisphere())
    chunk = 2048
    spans = [(s, min(s + chunk, len(P))) for s in range(0, len(P), chunk)]
    if pool_ok and len(spans) > 4 and WORKERS > 1:
        with mp.get_context("fork").Pool(WORKERS) as pool:
            parts = pool.map(_march, spans, chunksize=4)
    else:
        parts = [_march(sp) for sp in spans]
    ao = np.empty(len(P), dtype=np.float32)
    ao[order] = np.concatenate(parts) if parts else np.empty(0, np.float32)
    _G.clear()
    return ao, {"grid": list(shape), "voxels": int(raw.sum())}


INTERIOR = np.array([[4, 1, 1], [1, 4, 1], [1, 1, 4], [2, 2, 2]], dtype=np.float64) / 6.0
VERTEX_WEIGHT = 0.5    # weight of the point sample at the vertex itself in the fit
MIN_EDGE = 2 * VOXEL   # triangles smaller than this need no interior samples


def interior_samples(U: np.ndarray, Nu: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points inside the larger triangles (positions, interpolated normals, face ids)."""
    tri = U[F]
    edge = np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2).max(1)
    big = np.nonzero(edge >= MIN_EDGE)[0]
    P = np.einsum("sk,tkd->tsd", INTERIOR, tri[big]).reshape(-1, 3)
    N = np.einsum("sk,tkd->tsd", INTERIOR, Nu[F[big]]).reshape(-1, 3)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return P, N, big


def fit(ao_v: np.ndarray, F: np.ndarray, big: np.ndarray, ao_s: np.ndarray) -> np.ndarray:
    """Per-vertex values whose linear interpolation best matches all samples.

    Point-sampling AO only at the vertices smears a crease that lies at one
    vertex of a large triangle across the whole triangle (e.g. a broad grey
    band beside a suture on the coarse skull). Fitting to interior samples as
    well keeps the darkening where the triangle is actually occluded.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg
    nv = len(ao_v)
    if len(big) == 0:
        return ao_v
    rows = np.repeat(np.arange(len(ao_s)), 3)
    cols = np.repeat(F[big], len(INTERIOR), axis=0).reshape(-1)
    vals = np.tile(INTERIOR, (len(big), 1)).reshape(-1)
    A = coo_matrix((vals, (rows, cols)), shape=(len(ao_s), nv)).tocsr()
    M = (A.T @ A + diags(np.full(nv, VERTEX_WEIGHT))).tocsr()
    rhs = A.T @ ao_s.astype(np.float64) + VERTEX_WEIGHT * ao_v.astype(np.float64)
    pre = diags(1.0 / M.diagonal())
    x, _ = cg(M, rhs, x0=ao_v.astype(np.float64), M=pre, rtol=1e-6, maxiter=400)
    return x.astype(np.float32)


def unique_vertices(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of unique (pos, normal) rows and the inverse map."""
    v = np.ascontiguousarray(a).view(np.dtype((np.void, 24))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return first, inv.ravel()


# --------------------------------------------------------------------- apply
def apply(records: list[dict]) -> dict:
    t_start = time.time()
    arrays = [_decode32(m).copy() for m in records]
    summary: dict = check_and_repair(records, arrays)
    flipped = fix_orientation(records, arrays)
    summary["normalsFlippedOutward"] = {"count": len(flipped), "ids": flipped}
    ao_out: list[np.ndarray | None] = [None] * len(records)

    def receivers_of(group: str) -> list[int]:
        return [k for k, m in enumerate(records) if m["group"] == group]

    jobs: list[tuple[str, list[int], list[int], float]] = []
    by_key: dict[tuple, list[int]] = {}
    for g, (occ, reach) in GROUP_SETS.items():
        by_key.setdefault((occ, reach), []).extend(receivers_of(g))
    for (occ, reach), recv in by_key.items():
        if recv:
            occ_idx = [k for k, m in enumerate(records) if m["group"] in occ]
            jobs.append(("+".join(occ), recv, occ_idx, reach))
    for k, m in enumerate(records):
        if m["group"] not in GROUP_SETS and m["group"] not in NO_AO:
            jobs.append((f"self:{m['id']}", [k], [k], SELF_REACH))

    timings = {}
    for name, recv, occ_idx, reach in jobs:
        t0 = time.time()
        recv = [k for k in recv if len(arrays[k])]
        occ_idx = [k for k in occ_idx if len(arrays[k])]
        if not recv:
            continue
        occ_tris = [arrays[k][:, :3].astype(np.float64).reshape(-1, 3, 3) for k in occ_idx]
        parts_p, parts_n, layout = [], [], []
        for k in recv:
            first, inv = unique_vertices(arrays[k])
            U = arrays[k][first, :3].astype(np.float64)
            Nu = arrays[k][first, 3:].astype(np.float64)
            Nu /= np.linalg.norm(Nu, axis=1, keepdims=True)
            F = inv.reshape(-1, 3)
            Ps, Ns, big = interior_samples(U, Nu, F)
            parts_p += [U, Ps]
            parts_n += [Nu, Ns]
            layout.append((k, inv, F, big, len(U), len(Ps)))
        P, N = np.concatenate(parts_p), np.concatenate(parts_n)
        ao, info = bake(occ_tris, P, N, reach, pool_ok=not name.startswith("self:") or len(P) > 20000)
        off = 0
        for k, inv, F, big, nu, ns in layout:
            a_v, a_s = ao[off:off + nu], ao[off + nu:off + nu + ns]
            off += nu + ns
            a_u = fit(a_v, F, big, a_s)
            a_u = np.clip(np.clip(a_u, 0.0, 1.0) ** GAMMA, AO_MIN, 1.0)
            ao_out[k] = a_u[inv]
        if not name.startswith("self:"):
            timings[name] = {"seconds": round(time.time() - t0, 1), "rayOrigins": int(len(P)),
                             "receivers": len(recv), "occluders": len(occ_idx), **info}
    summary["sets"] = timings
    summary["selfSets"] = sum(1 for j in jobs if j[0].startswith("self:"))

    hist_edges = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    per_group: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for k, (m, a) in enumerate(zip(records, arrays)):
        n = a[:, 3:].astype(np.float64)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        if ao_out[k] is not None:
            n *= ao_out[k][:, None].astype(np.float64)
            t = a[:, :3].astype(np.float64).reshape(-1, 3, 3)
            area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
            per_group.setdefault(m["group"], []).append((ao_out[k], np.repeat(area / 3, 3)))
        # positions are unchanged, so keep the anchor chosen by earlier steps (encode() would reset
        # it to the bounding-box centre, which for a curved sheet such as the dura lies off the sheet)
        anchor = m.get("anchor")
        encode(m, a[:, :3].astype(np.float64), n)
        if anchor is not None:
            m["anchor"] = anchor
    hist = {"edges": hist_edges, "weighting": "surface area"}
    for g, parts in sorted(per_group.items()):
        v = np.concatenate([p[0] for p in parts]).astype(np.float64)
        w = np.concatenate([p[1] for p in parts])
        h = np.histogram(v, bins=hist_edges, weights=w)[0] / w.sum()
        hist[g] = {"mean": round(float(np.average(v, weights=w)), 3),
                   "fraction": [round(float(x), 4) for x in h]}
    summary["histogram"] = hist
    summary["seconds"] = round(time.time() - t_start, 1)
    return summary


if __name__ == "__main__":
    import json
    if len(sys.argv) != 3:
        sys.exit("usage: ao.py IN.json OUT.json")
    recs = load_records(sys.argv[1])
    result = apply(recs)
    save_records(recs, sys.argv[2])
    print(json.dumps(result, indent=1))
