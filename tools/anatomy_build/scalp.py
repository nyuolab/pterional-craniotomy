"""Illustrative scalp layers over the cranial vault, with the pterional scalp flap.

Adds three tissue sheets over the whole vault (brow to occiput, down to the
zygomatic arch and the level of the external ear), deep to superficial:

=====================  ============  =============================================
record ids             group         content
=====================  ============  =============================================
ILL-pericranium*       pericranium   pericranium (outer-table periosteum): a thin
                                     closed sheet on the outer table, absent over
                                     the temporal fascia (it ends at the superior
                                     temporal line, where the temporal fascia
                                     takes over)
ILL-galea*             galea         galea aponeurotica, continuous with the
                                     temporoparietal fascia over the temporal
                                     region; separated from the pericranium and
                                     the temporal fascia by the loose areolar
                                     (subgaleal) plane, drawn as an empty gap
ILL-scalp*             skin          skin and subcutaneous tissue as one closed
                                     slab (outer skin surface, inner subcutaneous
                                     surface and walls along every cut edge)
=====================  ============  =============================================

Each sheet is split per side at the midline and, per side, into the operative
flap, the stationary operative region around it and the rest of the vault
(``*-context``, contextOnly). Every record is a closed slab with walls along
all of its edges, so cut edges read as solid tissue and neighbouring records
meet with coincident walls (watertight at progress 0).

Method. Surfaces are radial height fields around a centre ``C`` inside the
cranium, parametrised by the azimuthal-equidistant map of the ray direction
(analysis grid) and meshed in its stereographic (conformal) map with Poisson
disc samples and a Delaunay triangulation in which every cut, seam and
boundary curve is a chain of mesh edges. Rays cast inward from outside give the
outer radius of the atlas skull and temporalis and of the illustrative temporal
layers (``temporal_layers.py``). Each layer is offset outward from the one
beneath with smooth "cover" fields that never dip below their base (as in
``temporal_layers.py``), so the layers stay clear of each other, of the temporal
layers and of the bone, except sub-millimetre bone bumps (up to about 0.7 mm,
parietal and occipital) under the context pericranium, whose bone field is
median-filtered. Offsets are divided by the cosine between the ray and the
surface normal, so the stated thicknesses are measured along the normal.

Incision. The course comes from the atlas schematic marker ``MARK-incision``
(its centreline, smoothed). Its lower third is replaced because the marker ends
about 2.4 cm above the zygomatic arch and 1.8 cm in front of the tragus and its
middle runs over the posterior part of the renderer's bone window; the cut used
here starts at the upper border of the zygomatic arch about 1 cm anterior to
the tragus, bows gently posteriorly behind the bone window and rejoins the
marker's upper course, which ends at the midline behind the hairline. The final
polyline (on the skin) is stored as ``incisionLine`` on the scalp flap records.

Flaps. Skin slab and galea anterior to the incision form the scalp flap; the
pericranium over the frontal and frontotemporal bone forms the pericranial flap
(cut 2 mm in front of the skin incision, kept at least 1.5 mm outside the renderer's
bone window, and along the temporal fascia, anteriorly based). Both carry a ``flap`` field per the FLAP CONTRACT (reference numpy
implementation: ``flap_deform``): ``{key, pivot, axis, away, lift, radius,
angle}`` (scalp: key "scalp" on the skin and galea flaps, which move together;
pericranium: key "pericranium"). Both use the SAME fields: the hinge plane passes
through ``C`` and the medial end of the incision and reaches the free edge at the
lateral orbital rim, so the hinge seam is the same curve in every layer, and since
the curl maps each layer by depth, the pericranium (raised later) and any deeper
layer folded with these fields stay outside the already-reflected scalp flap.
The axis is tangent to the skin from outside and turned in the hinge plane to
minimise how far the curling skin lies below it; a straight hinge cannot follow
the skin all along, so the optional ``hingeProfile`` (local skin height along
the axis, quadratic) lets a renderer curl about the local skin height instead,
which keeps the roll uniform and folds the large flap compactly over the face.
Radius 5 mm, angle 1.2 pi (folded back past flat, leaning toward the face).

Layout. Each half is meshed once in the canonical right frame. The left half
uses the same cut, hinge, seam and sizing (so both halves meet exactly at the
midline) with its radii taken from the native left anatomy (fields mirrored into
the right frame and back); its pericranium outline follows the left temporal
fascia. Walls stop 0.25 mm short of the slab surfaces so coincident walls of
neighbouring records never show as hairlines.

Thicknesses (nominal, along the surface normal): skin 2.5 mm, subcutaneous tissue 1.9 mm (1.8 mm
over the temporal fascia), galea 0.8 mm (temporoparietal fascia 1.2 mm), loose areolar gap 0.5 mm,
pericranium 0.4 mm on a 0.05 mm gap. The smooth "cover" fields add a little where the surface
beneath is uneven; measured over the vault (right side, 5th/50th/95th percentile): whole scalp
6.7/7.1/9.0 mm, skin slab 4.5/4.7/5.7, areolar gap 0.7/0.8/1.3, pericranium-to-bone gap 0.2/0.3/0.4.
Illustrative values within published ranges; not segmented from source data.

The loose areolar layer is represented by the empty gap only (no veil record): it is the plane in
which the flap separates, so the open gap is what the cut edges should show, and a translucent sheet
between two opaque layers would be invisible while closed and need its own motion rule when open.

Superficial temporal vessels. The atlas superficial temporal artery and its frontal branch (contextOnly)
lie at or below the bone/temporal-fascia level over most of their course, deeper than this galea, and the
illustrative superficial temporal vein (``veins.py``) follows them. Where the scalp covers them they are
re-seated radially into the subcutaneous layer, resting on the galea / temporoparietal fascia
(``reseat_superficial_vessels``: each vertex moves along the shell radius by the offset of its local
tube centre, faded in over ``VESSEL_FADE`` inside the scalp's free edge, so the vessels stay continuous
where they leave the scalp at the zygomatic arch; normals are recomputed). The renderer moves the parts
anterior to the incision with the scalp flap. The atlas epicranial aponeurosis, frontalis, occipitalis
and temporoparietalis records (contextOnly muscle group) overlap these shells in Full Head Context.

CLI: python3 tools/anatomy_build/scalp.py IN.json OUT.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (decode, encode, find, load_records, mirror_x, normalize, ray_nearest,  # noqa: E402
                    save_records, vertex_normals, weld)

MM = 0.05                                   # display units per millimetre
C = np.array([0.0, -0.4, -0.4])             # parametrisation centre (on the midline)
POL_MAX = 2.7                               # analysis grid covers polar angles below this (radians; 155 deg)
H = 0.006                                   # analysis grid spacing (radians, ~0.5 mm at the scalp)

# Layer thicknesses (display units, along the surface normal).
T_SKIN = 2.5 * MM
T_SUBCUT = 1.9 * MM
T_SUBCUT_TEMPORAL = 1.8 * MM
T_GALEA = 0.8 * MM
T_TPF = 1.2 * MM
T_AREOLAR = 0.5 * MM
T_PERI = 0.4 * MM
T_PERI_GAP = 0.05 * MM
CLEAR = 0.004                               # minimum clearance kept by cover() (0.08 mm)
CLOSE_CELLS = 11                            # grey closing of the envelope (bridges pits up to ~5 mm wide)
SMOOTH_GALEA = 8.0                          # grid cells (~4 mm): the galea bridges small steps of what lies beneath
SMOOTH_SKIN = 18.0                          # grid cells (~9 mm): the outer skin surface is smooth over steps
SKIN_GAP = 0.06 * MM                        # skin slab inner surface sits this far above the galea
WALL_INSET = 0.0125                         # walls stop 0.25 mm short of the slab surfaces (max 1/4 thickness)

# Bones excluded from the outer envelope (internal or not part of the vault surface).
SMALL_BONES = ("incus", "malleus", "stapes", "concha", "cells of ethmoid", "hyoid", "vomer", "palatine",
               "sinus of")
TEMPORAL_IDS = ["ILL-subfat", "ILL-deepfascia", "ILL-interfat", "ILL-superfascia", "ILL-suprafat"]


# ======================================================================= parametrisation
def ae_of(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal-equidistant coordinates (u ~ x, v ~ z) of points about C, and their radii."""
    d = np.asarray(P, dtype=np.float64) - C
    r = np.linalg.norm(d, axis=-1)
    d = d / np.maximum(r, 1e-12)[..., None]
    pol = np.arccos(np.clip(d[..., 1], -1, 1))
    h = np.hypot(d[..., 0], d[..., 2])
    s = np.where(h > 1e-12, pol / np.maximum(h, 1e-12), 1.0)
    return np.stack([d[..., 0] * s, d[..., 2] * s], -1), r


def ae_dir(uv: np.ndarray) -> np.ndarray:
    """Unit ray direction for azimuthal-equidistant coordinates."""
    uv = np.asarray(uv, dtype=np.float64)
    pol = np.hypot(uv[..., 0], uv[..., 1])
    s = np.where(pol > 1e-12, np.sin(pol) / np.maximum(pol, 1e-12), 1.0)
    return np.stack([uv[..., 0] * s, np.cos(pol), uv[..., 1] * s], -1)


def ae_to_st(uv: np.ndarray) -> np.ndarray:
    """Azimuthal-equidistant -> stereographic (conformal) plane, same azimuth."""
    pol = np.hypot(uv[..., 0], uv[..., 1])
    k = np.where(pol > 1e-12, 2 * np.tan(pol / 2) / np.maximum(pol, 1e-12), 1.0)
    return uv * k[..., None]


def st_to_ae(st: np.ndarray) -> np.ndarray:
    rho = np.hypot(st[..., 0], st[..., 1])
    k = np.where(rho > 1e-12, 2 * np.arctan(rho / 2) / np.maximum(rho, 1e-12), 1.0)
    return st * k[..., None]


class Grid:
    """Regular analysis grid in azimuthal-equidistant coordinates."""

    def __init__(self) -> None:
        n = int(np.ceil(POL_MAX / H)) + 2
        self.g = np.arange(-n, n + 1) * H
        self.U, self.V = np.meshgrid(self.g, self.g, indexing="ij")
        self.shape = self.U.shape
        self.inside = np.hypot(self.U, self.V) < POL_MAX
        self.D = ae_dir(np.stack([self.U, self.V], -1))

    def idx(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (uv[..., 0] - self.g[0]) / H, (uv[..., 1] - self.g[0]) / H

    def sample(self, field: np.ndarray, uv: np.ndarray, order: int = 1) -> np.ndarray:
        fi, fj = self.idx(uv)
        return ndimage.map_coordinates(field, [fi, fj], order=order, mode="nearest")

    def mirror(self, field: np.ndarray) -> np.ndarray:
        """Field as seen in the mirrored (x -> -x, i.e. u -> -u) frame."""
        return field[::-1, :].copy()


# ======================================================================= ray casting
def _cast(args) -> np.ndarray:
    """Outer radius along every grid direction (nan for misses). Runs in a worker process."""
    tris, rows = args
    grid = Grid()
    U, V = grid.U[rows], grid.V[rows]
    D = grid.D[rows]
    uv, _ = ae_of(tris.reshape(-1, 3))
    uv = uv.reshape(-1, 3, 2)
    keep = np.hypot(uv[..., 0], uv[..., 1]).max(1) < 3.0
    tris, uv = tris[keep], uv[keep]
    pad = 0.02
    lo, hi = uv.min(1) - pad, uv.max(1) + pad
    R = np.full(U.shape, np.nan)
    tile, far = 16, 12.0
    inside = grid.inside[rows]
    for i in range(0, U.shape[0], tile):
        for j in range(0, U.shape[1], tile):
            if not inside[i:i + tile, j:j + tile].any():
                continue
            u, v = U[i:i + tile, j:j + tile], V[i:i + tile, j:j + tile]
            sel = (hi[:, 0] >= u.min()) & (lo[:, 0] <= u.max()) & (hi[:, 1] >= v.min()) & (lo[:, 1] <= v.max())
            if not sel.any():
                continue
            T = tris[sel]
            Vv = T.reshape(-1, 3)
            F = np.arange(len(Vv)).reshape(-1, 3)
            d = D[i:i + tile, j:j + tile].reshape(-1, 3)
            t, _ = ray_nearest(C + far * d, -d, Vv, F, chunk=64)
            r = far - t
            r[~np.isfinite(t)] = np.nan
            R[i:i + u.shape[0], j:j + u.shape[1]] = r.reshape(u.shape)
    return R


def source_meshes(records: list[dict]) -> dict[str, np.ndarray]:
    """Triangle soups (T, 3, 3) of the structures the scalp rests on, keyed by field name."""
    def tris(ms):
        return np.concatenate([decode(m)[0].reshape(-1, 3, 3) for m in ms])
    bones = [m for m in records if m["group"] == "bone"
             and not any(s in m["name"].lower() for s in SMALL_BONES)]
    out = {"bone": tris(bones)}
    for suffix, key in (("", "r"), ("-l", "l")):
        out[f"tl_{key}"] = tris([find(records, id=i + suffix) for i in TEMPORAL_IDS])
        out[f"df_{key}"] = tris([find(records, id="ILL-deepfascia" + suffix)])
        out[f"tm_{key}"] = tris([find(records, native="Temporalis muscle." + key)])
    return out


def cast_fields(meshes: dict[str, np.ndarray], workers: int | None = None) -> dict[str, np.ndarray]:
    """Outer-radius fields on the analysis grid for every mesh (deterministic)."""
    grid = Grid()
    n = grid.shape[0]
    bands = [slice(s, min(n, s + 64)) for s in range(0, n, 64)]
    jobs = [(k, b) for k in meshes for b in bands]
    workers = workers or min(16, os.cpu_count() or 1)
    args = [(meshes[k], b) for k, b in jobs]
    if workers <= 1:
        res = [_cast(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_cast, args, chunksize=1))
    out = {k: np.full(grid.shape, np.nan) for k in meshes}
    for (k, b), R in zip(jobs, res):
        out[k][b] = R
    for k in out:
        out[k][~grid.inside] = np.nan
    return out


# ======================================================================= domain boundary
# Free edge of the scalp: polar angle of the ray direction (degrees from +Y about C) as a function of
# azimuth (degrees; 0 = anterior, -90 = right lateral, +-180 = posterior), right side, mirrored for
# the left. Read off the atlas surfaces: the brow about 3 mm above the supraorbital margin, the level of
# the external acoustic meatus in front of and behind the ear, and the occiput just below the inion.
# Between the brow and the ear the edge is pushed down to just beyond the anterior and lower border of
# the temporal fascia (lateral orbital rim, upper border of the zygomatic arch; see edge_series), so the
# scalp covers the temporal layers and stops short of the arch, whose outer surface lies lateral to them.
EDGE_TABLE = [(0, 97.5), (-10, 98.5), (-15, 97.6), (-20, 96.8), (-25, 97.6), (-30, 100.0), (-35, 100.5),
              (-40, 104.0), (-45, 112.0), (-50, 116.0), (-60, 118.0), (-70, 118.0), (-80, 119.0),
              (-90, 124.4), (-100, 124.6), (-110, 124.0), (-120, 122.6), (-135, 118.7), (-150, 115.5),
              (-165, 113.5), (-180, 112.5)]
EDGE_SMOOTH = 2.5    # degrees of azimuthal smoothing that rounds the corners
EDGE_PAST_FASCIA = 0.5  # degrees: the edge passes this far beyond the lower border of the temporal fascia
EDGE_MARGIN = 0.006  # radians beyond the edge in which atlas data still shapes the layers
EDGE_STEP = 0.25     # degrees of azimuth per edge sample


def fascia_border(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Largest polar angle (degrees) of the temporal fascia per EDGE_STEP azimuth bin, both sides folded
    onto the right (so the symmetric edge covers both fasciae); -inf where there is no fascia."""
    grid = Grid()
    az = np.degrees(np.arctan2(grid.U, grid.V))
    pol = np.degrees(np.hypot(grid.U, grid.V))
    nb = int(round(360 / EDGE_STEP))
    out = np.full(nb, -np.inf)
    for key, sgn in (("df_r", 1.0), ("df_l", -1.0)):
        m = np.isfinite(raw[key])
        a = az[m] * sgn
        b = np.floor((a + 180.0) / EDGE_STEP).astype(int) % nb
        np.maximum.at(out, b, pol[m])
    out = np.maximum(out, out[np.r_[0, np.arange(nb - 1, 0, -1)]])   # symmetric
    return out


def edge_series(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Free edge sampled every EDGE_STEP degrees of azimuth over [-180, 180): polar angles in radians."""
    from scipy.interpolate import PchipInterpolator
    az = np.array([a for a, _ in EDGE_TABLE], dtype=np.float64)
    pol = np.array([p for _, p in EDGE_TABLE], dtype=np.float64)
    az = np.concatenate([az[::-1], -az[1:-1]])     # -180 .. 0 .. 180 (mirror; symmetric)
    pol = np.concatenate([pol[::-1], pol[1:-1]])
    o = np.argsort(az)
    az, pol = az[o], pol[o]
    t = np.arange(-180.0, 180.0, EDGE_STEP)
    p = PchipInterpolator(np.concatenate([az, [az[0] + 360]]), np.concatenate([pol, [pol[0]]]))(
        np.where(t < az[0], t + 360, t))
    fb = ndimage.maximum_filter1d(fascia_border(raw), 5, mode="wrap") + EDGE_PAST_FASCIA
    p = np.maximum(p, fb)
    p = ndimage.gaussian_filter1d(p, EDGE_SMOOTH / EDGE_STEP, mode="wrap")
    p = np.maximum(p, fb - 0.5 * EDGE_PAST_FASCIA)
    p = ndimage.gaussian_filter1d(p, 1.0 / EDGE_STEP, mode="wrap")
    p = 0.5 * (p + p[np.r_[0, np.arange(len(t) - 1, 0, -1)]])   # exact left/right symmetry
    return np.radians(p)


def edge_pol(az: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Polar angle of the free edge at azimuth ``az`` (radians), periodic linear interpolation."""
    x = (np.degrees(np.asarray(az)) + 180.0) / EDGE_STEP
    i0 = np.floor(x).astype(int) % len(table)
    f = x - np.floor(x)
    return table[i0] * (1 - f) + table[(i0 + 1) % len(table)] * f


def edge_curve(table: np.ndarray, n: int = 4000) -> np.ndarray:
    """Closed free-edge curve in azimuthal-equidistant coordinates (counter-clockwise, no repeat)."""
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    p = edge_pol(t, table)
    return np.stack([p * np.sin(t), p * np.cos(t)], -1)


def in_domain(uv: np.ndarray, table: np.ndarray, margin: float = 0.0) -> np.ndarray:
    az = np.arctan2(uv[..., 0], uv[..., 1])
    return np.hypot(uv[..., 0], uv[..., 1]) < edge_pol(az, table) + margin


# ======================================================================= layer fields
def fill_nan(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def cover(field: np.ndarray, sigma: float, clear: float = CLEAR, reach: int = 3) -> np.ndarray:
    """Smooth surface at least ``clear`` above ``field`` and its (reach x reach) neighbourhood.

    The lift is built from smooth bumps only (no max() creases), so piecewise-linear meshes of the
    result stay clear of meshes of ``field`` between samples (as in temporal_layers.cover).
    """
    out = ndimage.gaussian_filter(field, sigma)
    floor = ndimage.maximum_filter(field, reach) + clear
    for _ in range(80):
        viol = floor - out
        if viol.max() <= 0:
            break
        out = out + ndimage.gaussian_filter(ndimage.maximum_filter(viol.clip(0), 7), 2.0) * 1.2 + 2e-5
    return out


def normal_cosine(grid: Grid, r: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Cosine between the ray and the normal of the surface C + r * dir (smoothed, >= 0.55)."""
    P = C + r[..., None] * grid.D
    du = np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    n = normalize(np.cross(du, dv))
    c = np.abs(np.einsum("ijk,ijk->ij", n, grid.D))
    return np.clip(ndimage.gaussian_filter(c, sigma), 0.55, 1.0)


def smooth_indicator(mask: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(mask.astype(np.float64), sigma)


def layer_fields(raw: dict[str, np.ndarray], edge: np.ndarray) -> dict[str, np.ndarray]:
    """Radius fields of every layer surface on the analysis grid (display coordinates, both sides).

    Keys: bone (covered outer table), env (bone/temporal soft-tissue envelope), p0/p1 (pericranium),
    g0/g1 (galea), s0/s1 (skin slab), plus 'fascia' (smooth temporal-fascia footprint indicator, 0..1),
    'tsub' (subcutaneous thickness used) and the domain mask.
    """
    grid = Grid()
    uv = np.stack([grid.U, grid.V], -1)
    dom = in_domain(uv, edge)
    # Fields are trusted only slightly beyond the free edge and extended outward as constants beyond
    # it, so the cliffs of the orbit and the face below the edge cannot pull the smoothed layers.
    near = in_domain(uv, edge, margin=EDGE_MARGIN)
    rb = raw["bone"].copy()
    rb = ndimage.median_filter(fill_nan(np.where(near, rb, np.nan)), 3)
    tl = np.fmax(raw["tl_r"], raw["tl_l"])
    tm = np.fmax(raw["tm_r"], raw["tm_l"])
    soft = np.fmax(tl, np.where(np.isfinite(tl), tm, np.nan))   # temporalis only under the layers' footprint
    # Outer envelope of bone and temporal soft tissue, with narrow pits closed (sutural gaps, the
    # external acoustic meatus, the slots between the fascia sheet's edge and the arch / lateral orbital
    # rim, which radial rays see as deep holes); the scalp bridges these.
    env = np.fmax(fill_nan(raw["bone"]), fill_nan(np.where(np.isfinite(raw["bone"]), soft, np.nan)))
    env = np.where(np.isfinite(soft), env, np.fmax(fill_nan(raw["bone"]), 0))
    env = ndimage.grey_closing(fill_nan(env), size=(CLOSE_CELLS, CLOSE_CELLS))
    env = np.fmax(env, rb)
    env = fill_nan(np.where(near, env, np.nan))
    fascia = np.isfinite(raw["df_r"]) | np.isfinite(raw["df_l"])
    fas = smooth_indicator(fascia, 2.0)

    # pericranium: on the covered outer table where bone is the outermost surface outside the temporal
    # fascia (the bone field is used only there, so the cliffs of the temporal fossa, the arch and the
    # pits closed above cannot lift or sink it)
    peri_cells = near & (fas < 0.5) & (env - rb < 1.5 * MM)
    bone_c = cover(fill_nan(np.where(peri_cells, rb, np.nan)), sigma=1.5)
    cb = normal_cosine(grid, bone_c)
    p0 = bone_c + T_PERI_GAP / cb
    p1 = p0 + T_PERI / cb
    # galea / temporoparietal fascia: loose areolar gap above the pericranium or the temporal layers
    base_g = np.fmax(env, np.where(peri_cells, p1, -np.inf))
    base_g = cover(base_g, sigma=SMOOTH_GALEA)
    cg = normal_cosine(grid, base_g)
    g0 = base_g + T_AREOLAR / cg
    t_g = T_GALEA + (T_TPF - T_GALEA) * fas
    g1 = g0 + t_g / normal_cosine(grid, g0)
    # skin slab: subcutaneous inner surface on the galea, outer skin surface smoothed further
    s0 = g1 + SKIN_GAP
    cs = normal_cosine(grid, s0, sigma=4.0)
    tsub = T_SUBCUT + (T_SUBCUT_TEMPORAL - T_SUBCUT) * fas
    s1 = cover(s0 + (tsub + T_SKIN) / cs, sigma=SMOOTH_SKIN, clear=0.0, reach=3)
    s1 = np.fmax(s1, s0 + 0.9 * (tsub + T_SKIN) / cs)
    return {"bone": bone_c, "env": env, "p0": p0, "p1": p1, "g0": g0, "g1": g1, "s0": s0, "s1": s1,
            "fascia": fas, "domain": dom, "rb": rb, "peri": smooth_indicator(peri_cells, 1.5)}


# ======================================================================= incision
# Bone window of the renderer (assets/app.js): bone with |x| > 1.9 inside this (y, z) ellipse is removed.
WINDOW = dict(y=0.3, z=1.25, ry=1.55, rz=1.65)
WINDOW_CLEAR = 0.15          # the cut passes this far (3 mm) behind the window's posterior margin
INCISION_Z_ARCH = 0.35       # start on the arch ~1 cm anterior to the tragus (EAM porus at z ~ -0.4)
MARKER_JOIN_Y = 1.93         # above this height the cut follows the atlas schematic marker


def window_posterior_z(y: np.ndarray) -> np.ndarray:
    q = np.clip(1 - ((np.asarray(y) - WINDOW["y"]) / WINDOW["ry"]) ** 2, 0, 1)
    return WINDOW["z"] - WINDOW["rz"] * np.sqrt(q)


def marker_centreline(m: dict) -> np.ndarray:
    """Centreline of a ribbon/tube marker: vertices ordered by geodesic distance from one end, averaged
    in bins, then smoothed. Returned from the lower (zygomatic) end to the upper (midline) end."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from common import weld
    pos, _ = decode(m)
    V, F = weld(pos, 1e-4)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    A = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(V), len(V))).tocsr()
    A = A + A.T
    a = int(np.argmax(dijkstra(A, indices=int(np.argmin(V[:, 1])))))
    d = dijkstra(A, indices=a)
    t = d / d[np.isfinite(d)].max()
    bins = np.linspace(0, 1, 81)
    cl = [V[(t >= bins[i]) & (t < bins[i + 1])].mean(0) for i in range(80) if ((t >= bins[i]) & (t < bins[i + 1])).any()]
    cl = np.array(cl)
    if cl[0, 1] > cl[-1, 1]:
        cl = cl[::-1]
    return ndimage.gaussian_filter1d(cl, 1.5, axis=0, mode="nearest")


class SkinLookup:
    """Find skin points on the right lateral surface by their (y, z) coordinates."""

    def __init__(self, grid: Grid, skin: np.ndarray, dom: np.ndarray):
        P = C + skin[..., None] * grid.D
        sel = dom & (P[..., 0] < -1.2)
        self.uv = np.stack([grid.U, grid.V], -1)[sel]
        self.P = P[sel]
        self.tree = cKDTree(self.P[:, 1:])

    def uv_at(self, yz: np.ndarray, k: int = 4) -> np.ndarray:
        d, i = self.tree.query(np.atleast_2d(yz), k=k)
        w = 1.0 / np.maximum(d, 1e-6)
        return (self.uv[i] * w[..., None]).sum(1) / w.sum(1)[:, None]


def resample(P: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])], -1)


def incision_uv(grid: Grid, fields: dict, edge: np.ndarray, marker_cl: np.ndarray) -> tuple[np.ndarray, dict]:
    """Incision polyline (azimuthal-equidistant coordinates, canonical right frame), extended a little
    below the free edge (the mesh arrangement clips it there). Returns (uv, info)."""
    look = SkinLookup(grid, fields["s1"], fields["domain"])
    # skin point on the free edge at the arch with z closest to INCISION_Z_ARCH
    B = edge_curve(edge, 7200)
    PB = C + grid.sample(fields["s1"], B)[:, None] * ae_dir(B)
    lat = (PB[:, 0] < -2.5) & (np.abs(PB[:, 2] - INCISION_Z_ARCH) < 0.6) & (PB[:, 1] < -2.0)
    k = np.flatnonzero(lat)[np.argmin(np.abs(PB[lat, 2] - INCISION_Z_ARCH))]
    y0 = PB[k, 1]
    ys = np.array([y0, y0 + 0.45, -1.6, -1.0, -0.5, 0.0, 0.3, 0.8, 1.3, 1.6, 1.8])
    zs = window_posterior_z(ys) - WINDOW_CLEAR
    zs[0], zs[1], zs[2] = INCISION_Z_ARCH, INCISION_Z_ARCH - 0.04, min(zs[2], INCISION_Z_ARCH - 0.11)
    lower = np.stack([ys, zs], -1)
    # upper part from the atlas marker (projected radially onto the skin)
    mk = marker_cl[marker_cl[:, 1] >= MARKER_JOIN_Y]
    uv_up, _ = ae_of(mk)
    uv_low = look.uv_at(lower)
    # smooth joint: Catmull-Rom through the control points in uv
    from common import catmull_rom
    ctrl = np.concatenate([uv_low, uv_up[::3], uv_up[-1:]])
    _, keep = np.unique(np.round(ctrl, 6), axis=0, return_index=True)
    ctrl = ctrl[np.sort(keep)]
    cur = catmull_rom(ctrl, 24)
    cur = ndimage.gaussian_filter1d(cur, 6, axis=0, mode="nearest")
    cur[-1] = uv_up[-1]
    # extend below the edge along the initial tangent so the arrangement finds the edge crossing
    t0 = normalize(cur[0] - cur[6])
    ext = cur[0] + np.outer(np.linspace(0.06, 0.005, 6), t0)
    cur = np.concatenate([ext, cur])
    cur = resample(cur, 0.002)
    info = {"edgeY": float(y0), "controlYZ": lower.round(3).tolist(), "markerJoinY": MARKER_JOIN_Y}
    return cur, info


# ======================================================================= flap (FLAP CONTRACT)
def flap_deform(pos: np.ndarray, nrm: np.ndarray | None, flap: dict, progress: float, profile: bool = False):
    """Reference implementation of the FLAP CONTRACT deformation (numpy).

    v = p - pivot; d = v.away; h = v.lift; rest = v - d away - h lift. Points with d <= 0 stay.
    Otherwise T = angle * progress, s = min(d, radius T), phi = s / radius,
    away' = cos(phi) away + sin(phi) lift, lift' = -sin(phi) away + cos(phi) lift,
    p' = pivot + rest + radius sin(phi) away + radius (1 - cos(phi)) lift + (d - s) away' + h lift'.
    Normals rotate by phi in the (away, lift) plane.

    ``profile=True`` applies the optional ``hingeProfile`` refinement: the pivot moves along lift to the
    local skin height h_s(a) at each vertex's axial coordinate a (clamped to the profile range).
    """
    piv = np.asarray(flap["pivot"], dtype=np.float64)
    if profile and "hingeProfile" in flap:
        ax_ = np.asarray(flap["axis"], dtype=np.float64)
        li_ = np.asarray(flap["lift"], dtype=np.float64)
        c0, c1, c2 = flap["hingeProfile"]["c"]
        lo, hi = flap["hingeProfile"]["range"]
        a = np.clip((np.asarray(pos, dtype=np.float64) - piv) @ ax_, lo, hi)
        piv = piv + (c0 + c1 * a + c2 * a * a)[:, None] * li_
    aw = np.asarray(flap["away"], dtype=np.float64)
    li = np.asarray(flap["lift"], dtype=np.float64)
    R, T = float(flap["radius"]), float(flap["angle"]) * float(progress)
    v = np.asarray(pos, dtype=np.float64) - piv
    d = v @ aw
    h = v @ li
    rest = v - d[:, None] * aw - h[:, None] * li
    s = np.clip(np.minimum(d, R * T), 0.0, None)
    phi = s / R
    c, sn = np.cos(phi)[:, None], np.sin(phi)[:, None]
    aw2 = c * aw + sn * li
    li2 = -sn * aw + c * li
    out = piv + rest + R * sn * aw + R * (1 - c) * li + (d - s)[:, None] * aw2 + h[:, None] * li2
    moved = d > 0
    out = np.where(moved[:, None], out, pos)
    if nrm is None:
        return out
    n = np.asarray(nrm, dtype=np.float64)
    na, nl = n @ aw, n @ li
    nrest = n - na[:, None] * aw - nl[:, None] * li
    n2 = nrest + na[:, None] * aw2 + nl[:, None] * li2
    return out, np.where(moved[:, None], n2, n)


def hinge_plane(e1_uv: np.ndarray, h0_uv: np.ndarray) -> np.ndarray:
    """Unit normal of the plane through C and the ray directions of E1 and H0."""
    return normalize(np.cross(ae_dir(e1_uv), ae_dir(h0_uv)))


def great_circle_uv(n: np.ndarray, e1_uv: np.ndarray, t0: float, t1: float, num: int = 2000) -> np.ndarray:
    """Directions in the plane through C with normal n, from the direction of E1 (t = 0) turning by t."""
    a = ae_dir(e1_uv)
    b = normalize(np.cross(n, a))
    t = np.linspace(t0, t1, num)
    d = np.cos(t)[:, None] * a + np.sin(t)[:, None] * b
    uv, _ = ae_of(C + d)
    return uv


# ======================================================================= planar meshing
# All meshing happens in the stereographic (conformal) plane of the ray direction, so isotropic
# point spacing there gives near-isotropic triangles on the near-spherical scalp.
def seg_intersections(A: np.ndarray, B: np.ndarray) -> list[tuple[int, float, int, float]]:
    """Proper intersections between the segments of polylines A and B: (i, s, j, t) with
    A[i] + s (A[i+1] - A[i]) == B[j] + t (B[j+1] - B[j]), s, t in [0, 1)."""
    a0, a1 = A[:-1], A[1:]
    b0, b1 = B[:-1], B[1:]
    mb = 0.5 * (b0 + b1)
    lb = np.linalg.norm(b1 - b0, axis=1).max() / 2
    tree = cKDTree(mb)
    la = np.linalg.norm(a1 - a0, axis=1)
    out = []
    cand = tree.query_ball_point(0.5 * (a0 + a1), la / 2 + lb + 1e-9)
    for i, js in enumerate(cand):
        if not js:
            continue
        js = np.asarray(js)
        r = a1[i] - a0[i]
        q = b1[js] - b0[js]
        den = r[0] * q[:, 1] - r[1] * q[:, 0]
        ok = np.abs(den) > 1e-14
        w = b0[js] - a0[i]
        s = np.where(ok, (w[:, 0] * q[:, 1] - w[:, 1] * q[:, 0]) / np.where(ok, den, 1), -1)
        t = np.where(ok, (w[:, 0] * r[1] - w[:, 1] * r[0]) / np.where(ok, den, 1), -1)
        hit = ok & (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
        out += [(i, float(s[k]), int(js[k]), float(t[k])) for k in np.flatnonzero(hit)]
    return out


class Arrangement:
    """Planar straight-line graph from polylines: intersections become shared junction points, the
    polylines are split into pieces between junctions and each piece is kept or dropped by a rule."""

    def __init__(self) -> None:
        self.curves: list[dict] = []

    def add(self, name: str, pts: np.ndarray, closed: bool = False, keep=None) -> None:
        P = np.asarray(pts, dtype=np.float64)
        if closed:
            P = np.concatenate([P, P[:1]])
        self.curves.append({"name": name, "P": P, "closed": closed, "keep": keep, "cuts": []})

    def build(self) -> tuple[np.ndarray, list[dict]]:
        cs = self.curves
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                for i, s, j, t in seg_intersections(cs[a]["P"], cs[b]["P"]):
                    p = cs[a]["P"][i] + s * (cs[a]["P"][i + 1] - cs[a]["P"][i])
                    key = tuple(np.round(p, 12))
                    cs[a]["cuts"].append((i, s, key))
                    cs[b]["cuts"].append((j, t, key))
        junction: dict[tuple, int] = {}
        jpts: list[np.ndarray] = []

        def jid(key):
            if key not in junction:
                junction[key] = len(jpts)
                jpts.append(np.array(key))
            return junction[key]

        pieces = []
        for c in cs:
            P = c["P"]
            n = len(P) - 1
            cuts = sorted(c["cuts"], key=lambda x: (x[0], x[1]))
            # parameter positions of cuts along the polyline
            seq = []   # list of (param, point, junction id)
            if not c["closed"]:
                # open-curve end points are junctions too, so curves sharing an end point share a vertex
                P[0] = np.round(P[0], 12)
                P[-1] = np.round(P[-1], 12)
            for i, s, key in cuts:
                seq.append((i + s, np.array(key), jid(key)))
            if c["closed"]:
                if not seq:
                    pieces.append({"name": c["name"], "P": P, "ends": (None, None)})
                    continue
                for k in range(len(seq)):
                    p0, q0, j0 = seq[k]
                    p1, q1, j1 = seq[(k + 1) % len(seq)]
                    if k + 1 == len(seq):
                        p1 = p1 + n
                    idx = [m % n for m in range(int(np.floor(p0)) + 1, int(np.floor(p1)) + 1)]
                    body = P[idx] if idx else np.zeros((0, 2))
                    pts = np.concatenate([q0[None], body, q1[None]])
                    pieces.append({"name": c["name"], "P": dedup(pts), "ends": (j0, j1)})
            else:
                bounds = ([(0.0, P[0], jid(tuple(P[0])))] + [x for x in seq if 0 < x[0] < n]
                          + [(float(n), P[-1], jid(tuple(P[-1])))])
                for k in range(len(bounds) - 1):
                    p0, q0, j0 = bounds[k]
                    p1, q1, j1 = bounds[k + 1]
                    idx = list(range(int(np.floor(p0)) + 1, int(np.ceil(p1))))
                    body = P[idx] if idx else np.zeros((0, 2))
                    pts = np.concatenate([q0[None], body, q1[None]])
                    if len(pts) >= 2 and np.linalg.norm(pts[-1] - pts[0]) + polylen(pts) > 1e-9:
                        pieces.append({"name": c["name"], "P": dedup(pts), "ends": (j0, j1)})
        kept = []
        rules = {c["name"]: c["keep"] for c in cs}
        for pc in pieces:
            rule = rules[pc["name"]]
            mid = point_at(pc["P"], 0.5)
            if rule is None or rule(mid):
                kept.append(pc)
        return np.array(jpts) if jpts else np.zeros((0, 2)), kept


def dedup(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > eps])
    return P[keep]


def polylen(P: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def point_at(P: np.ndarray, f: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = f * s[-1]
    return np.array([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def resample_by_size(P: np.ndarray, size) -> np.ndarray:
    """Resample an open polyline keeping both ends, with local spacing ``size(points)`` (plane units)."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return P[[0, -1]]
    fine = np.linspace(0, s[-1], max(8, int(s[-1] / 0.0015)))
    Q = np.stack([np.interp(fine, s, P[:, 0]), np.interp(fine, s, P[:, 1])], -1)
    h = size(Q)
    u = np.concatenate([[0], np.cumsum(np.diff(fine) / (0.5 * (h[1:] + h[:-1])))])
    n = max(1, int(np.round(u[-1])))
    tgt = np.linspace(0, u[-1], n + 1)
    t = np.interp(tgt, u, fine)
    out = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], -1)
    out[0], out[-1] = P[0], P[-1]
    return out


def poisson_points(lo, hi, size, inside, rng: np.random.Generator, cell: float = 0.04) -> np.ndarray:
    """Variable-radius Poisson disc samples: candidates with density ~ 1/size^2, then greedy acceptance
    in random priority order, resolved in parallel rounds with k-d trees."""
    gx = np.arange(lo[0], hi[0], cell)
    gy = np.arange(lo[1], hi[1], cell)
    X, Y = np.meshgrid(gx + cell / 2, gy + cell / 2, indexing="ij")
    cen = np.stack([X.ravel(), Y.ravel()], -1)
    hs = size(cen)
    cnt = np.ceil(5.0 * cell * cell / np.maximum(hs, 1e-6) ** 2).astype(int)
    cand = np.repeat(cen, cnt, axis=0) + rng.uniform(-0.5, 0.5, (int(cnt.sum()), 2)) * cell
    cand = cand[inside(cand)]
    cand = cand[rng.permutation(len(cand))]
    r = size(cand)
    accepted = np.zeros((0, 2))
    acc_r = np.zeros(0)
    pri = np.arange(len(cand))
    while len(cand):
        if len(accepted):
            d, k = cKDTree(accepted).query(cand, k=1, distance_upper_bound=float(r.max()) * 1.1)
            d, k = d[:, None], k[:, None]
            ok = np.ones(len(cand), bool)
            for col in range(d.shape[1]):
                kk = np.minimum(k[:, col], len(accepted) - 1)
                hit = np.isfinite(d[:, col]) & (d[:, col] < np.minimum(r, acc_r[kk]))
                ok &= ~hit
            cand, r, pri = cand[ok], r[ok], pri[ok]
            if not len(cand):
                break
        tree = cKDTree(cand)
        nb = tree.query_ball_point(cand, r)
        # conflict when closer than the smaller of the two radii (each point queries its own radius,
        # which finds every such neighbour); the lowest priority index in a conflict group wins
        win = np.array([all(pri[j] >= pri[i] or np.linalg.norm(cand[j] - cand[i]) >= min(r[i], r[j])
                            for j in nb[i]) for i in range(len(cand))])
        accepted = np.concatenate([accepted, cand[win]])
        acc_r = np.concatenate([acc_r, r[win]])
        cand, r, pri = cand[~win], r[~win], pri[~win]
    return accepted


def conforming_delaunay(fixed: np.ndarray, segs: np.ndarray, interior: np.ndarray, size,
                        max_rounds: int = 30):
    """Delaunay triangulation in which every constraint segment is an edge. Missing segments are split
    at their midpoint (the new point joins the constraint) and interior points inside the segment's
    diametral circle are removed, until all segments are present."""
    fixed = np.asarray(fixed, dtype=np.float64)
    segs = [tuple(s) for s in np.asarray(segs, dtype=np.int64)]
    interior = np.asarray(interior, dtype=np.float64)
    # drop interior points that crowd the constraints
    if len(interior):
        mids = 0.5 * (fixed[[s[0] for s in segs]] + fixed[[s[1] for s in segs]])
        lens = np.linalg.norm(fixed[[s[0] for s in segs]] - fixed[[s[1] for s in segs]], axis=1)
        d, k = cKDTree(np.concatenate([fixed, mids])).query(interior)
        lim = np.concatenate([0.6 * size(fixed), np.maximum(0.6 * lens, 0.6 * size(mids))])
        interior = interior[d > lim[k]]
    for _ in range(max_rounds):
        P = np.concatenate([fixed, interior])
        tri = Delaunay(P, qhull_options="Qbb Qc Qz Q12")
        S = tri.simplices
        E = np.sort(np.concatenate([S[:, [0, 1]], S[:, [1, 2]], S[:, [2, 0]]]), axis=1)
        have = set(map(tuple, E.tolist()))
        missing = [s for s in segs if tuple(sorted(s)) not in have]
        if not missing:
            return P, S, segs, len(fixed)
        new_fixed = list(fixed)
        drop = np.zeros(len(interior), bool)
        tree = cKDTree(interior) if len(interior) else None
        newsegs = []
        miss = set(missing)
        for s in segs:
            if s not in miss:
                newsegs.append(s)
                continue
            a, b = fixed[s[0]], fixed[s[1]]
            m = 0.5 * (a + b)
            k = len(new_fixed)
            new_fixed.append(m)
            newsegs += [(s[0], k), (k, s[1])]
            if tree is not None:
                for i in tree.query_ball_point(m, 0.5 * np.linalg.norm(b - a) * 1.05):
                    drop[i] = True
        fixed = np.array(new_fixed)
        segs = newsegs
        interior = interior[~drop]
    raise RuntimeError("conforming Delaunay did not converge")


def flood_faces(S: np.ndarray, segs: list[tuple[int, int]]) -> np.ndarray:
    """Label triangles by connectivity across non-constraint edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([S[:, [0, 1]], S[:, [1, 2]], S[:, [2, 0]]])
    key = np.sort(E, axis=1)
    tri = np.tile(np.arange(len(S)), 3)
    cons = set(tuple(sorted(s)) for s in segs)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = (ks[1:] == ks[:-1]).all(1)
    a, b = tri[order][:-1][same], tri[order][1:][same]
    blocked = np.array([tuple(k) in cons for k in ks[:-1][same]], dtype=bool)
    a, b = a[~blocked], b[~blocked]
    A = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(S), len(S)))
    return connected_components(A, directed=False)[1]


def contours(field: np.ndarray, level: float, gx: np.ndarray, gy: np.ndarray) -> list[np.ndarray]:
    """Marching-squares iso-lines of a grid field (axis 0 = gx, axis 1 = gy), chained into polylines.
    Closed loops repeat their first point at the end."""
    f = field - level
    H_, W_ = f.shape
    segs = []
    # edges of cell (i, j): 0 bottom (i,j)-(i+1,j), 1 right (i+1,j)-(i+1,j+1), 2 top (i,j+1)-(i+1,j+1),
    # 3 left (i,j)-(i,j+1); crossing points keyed by grid edge so neighbouring cells share them
    def key(i, j, e):
        return ("h", i, j) if e == 0 else ("v", i + 1, j) if e == 1 else ("h", i, j + 1) if e == 2 else ("v", i, j)

    def pt(k):
        kind, i, j = k
        if kind == "h":
            a, b = f[i, j], f[i + 1, j]
            t = a / (a - b)
            return np.array([gx[i] + t * (gx[i + 1] - gx[i]), gy[j]])
        a, b = f[i, j], f[i, j + 1]
        t = a / (a - b)
        return np.array([gx[i], gy[j] + t * (gy[j + 1] - gy[j])])

    pos = f > 0
    c0, c1, c2, c3 = pos[:-1, :-1], pos[1:, :-1], pos[1:, 1:], pos[:-1, 1:]
    code = c0 * 1 + c1 * 2 + c2 * 4 + c3 * 8
    table = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)],
             7: [(3, 2)], 8: [(2, 3)], 9: [(2, 0)], 10: [(0, 1), (2, 3)], 11: [(2, 1)], 12: [(1, 3)],
             13: [(1, 0)], 14: [(0, 3)]}
    ii, jj = np.nonzero((code > 0) & (code < 15))
    for i, j in zip(ii, jj):
        for a, b in table[int(code[i, j])]:
            segs.append((key(i, j, a), key(i, j, b)))
    nxt: dict = {}
    for a, b in segs:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    seen = set()
    out = []
    for start in list(nxt):
        if start in seen:
            continue
        # walk to one end (or around a loop)
        chain = [start]
        seen.add(start)
        cur = start
        while True:
            cand = [x for x in nxt[cur] if x not in seen]
            if not cand:
                break
            cur = cand[0]
            seen.add(cur)
            chain.append(cur)
        # extend backwards from the start for open chains
        cur = start
        back = []
        while True:
            cand = [x for x in nxt[cur] if x not in seen]
            if not cand:
                break
            cur = cand[0]
            seen.add(cur)
            back.append(cur)
        chain = back[::-1] + chain
        P = np.array([pt(k) for k in chain])
        if len(chain) > 2 and chain[0] in nxt[chain[-1]]:
            P = np.concatenate([P, P[:1]])
        out.append(P)
    return out


def smooth_loop(P: np.ndarray, sigma: float) -> np.ndarray:
    Q = P[:-1]
    Q = ndimage.gaussian_filter1d(Q, sigma, axis=0, mode="wrap")
    return np.concatenate([Q, Q[:1]])


# ======================================================================= per-side plan
HINGE_AZ = -36.0             # degrees: the hinge reaches the free edge at the lateral orbital rim
FLAP_RADIUS = 0.25           # bend radius of the reflected flaps (5 mm)
FLAP_ANGLE = 1.2 * np.pi     # flaps fold back past flat, leaning toward the face, at progress 1
OP_MARGIN = 25 * MM          # operative region extends this far behind the incision (seam to context)
PERI_CUT_OFFSET = 2.0 * MM   # pericranial cut runs this far in front of the skin incision
PERI_WINDOW_CLEAR = 1.5 * MM  # ... and at least this far outside the renderer's bone window (y-z ellipse)
PERI_EDGE_INSET = 0.004      # pericranium stops this far (radians, ~0.3 mm) inside the scalp edge
SEED_FLAP = np.array([-3.6, 0.3, 1.25])      # bone-window centre: inside every flap
SEED_CONTEXT = np.array([-2.0, 2.0, -3.5])   # back of the head: context region
SEED_PERI_FLAP = np.array([-3.0, 1.6, 2.1])  # frontal bone above the superior temporal line, in the window
# target 3D spacing (display units) of mesh vertices
H_FINE, H_CURL, H_FLAP, H_OP, H_CTX, H_GROW = 1.1 * MM, 1.3 * MM, 2.0 * MM, 2.6 * MM, 5.0 * MM, 0.3
POISSON_FACTOR = 0.9          # Poisson radius / target spacing (Poisson spacing averages ~1.2 radii)


def rasterize(grid: Grid, poly: np.ndarray) -> np.ndarray:
    P = resample(poly, H * 0.4)
    fi, fj = grid.idx(P)
    m = np.zeros(grid.shape, bool)
    for di in (0, 1):
        for dj in (0, 1):
            m[np.clip(np.floor(fi).astype(int) + di, 0, grid.shape[0] - 1),
              np.clip(np.floor(fj).astype(int) + dj, 0, grid.shape[1] - 1)] = True
    return m


def offset_curve(P: np.ndarray, dist_rad: np.ndarray | float, side_point: np.ndarray, away_side: bool,
                 sigma: float = 25.0) -> np.ndarray:
    """Offset a polyline in the plane by ``dist_rad`` toward (away_side False) or away from (True) the
    side containing ``side_point``; smoothed."""
    T = normalize(np.gradient(ndimage.gaussian_filter1d(P, 5, axis=0, mode="nearest"), axis=0))
    N = np.stack([-T[:, 1], T[:, 0]], -1)
    k = len(P) // 2
    toward = np.sign(np.dot(side_point - P[k], N[k]))
    sgn = -toward if away_side else toward
    Q = P + sgn * np.asarray(dist_rad)[..., None] * N
    return ndimage.gaussian_filter1d(Q, sigma, axis=0, mode="nearest")


def remove_loops(P: np.ndarray) -> np.ndarray:
    """Cut the small loops out of a self-intersecting open polyline (e.g. an offset curve)."""
    for _ in range(50):
        hits = [(i, s_, j, t_) for i, s_, j, t_ in seg_intersections(P, P) if j > i + 1]
        if not hits:
            return P
        i, s_, j, t_ = min(hits, key=lambda x: x[2] - x[0])
        x = P[i] + s_ * (P[i + 1] - P[i])
        P = np.concatenate([P[: i + 1], x[None], P[j + 1:]])
    return P


def extend(P: np.ndarray, start: float = 0.0, end: float = 0.0, n: int = 12) -> np.ndarray:
    """Extend a polyline straight along its end tangents by the given lengths."""
    parts = []
    if start > 0:
        t = normalize(P[0] - P[min(8, len(P) - 1)])
        parts.append(P[0] + np.outer(np.linspace(start, 0, n, endpoint=False), t))
    parts.append(P)
    if end > 0:
        t = normalize(P[-1] - P[max(-9, -len(P))])
        parts.append(P[-1] + np.outer(np.linspace(0, end, n + 1)[1:], t))
    return np.concatenate(parts)


def window_q(P: np.ndarray, clear: float = 0.0) -> np.ndarray:
    """Normalised y-z ellipse radius of points against the renderer's bone window grown by ``clear``."""
    return np.hypot((P[:, 1] - WINDOW["y"]) / (WINDOW["ry"] + clear), (P[:, 2] - WINDOW["z"]) / (WINDOW["rz"] + clear))


def clear_window(cut: np.ndarray, F: dict, grid: Grid) -> np.ndarray:
    """Pericranial cut (canonical frame) pushed outside the bone window grown by PERI_WINDOW_CLEAR: points
    whose pericranium surface point lies inside it move outward along the ellipse radius in y-z (x kept,
    direction re-taken from C), then the curve is smoothed; repeated until clear."""
    P = cut.copy()
    for _ in range(6):
        X = C + grid.sample(F["p1"], P)[:, None] * ae_dir(P)
        q = window_q(X, PERI_WINDOW_CLEAR + 0.2 * MM)
        bad = (q < 1) & (X[:, 0] < -1.9)
        if not bad.any():
            break
        Y = X.copy()
        Y[bad, 1] = WINDOW["y"] + (X[bad, 1] - WINDOW["y"]) / q[bad]
        Y[bad, 2] = WINDOW["z"] + (X[bad, 2] - WINDOW["z"]) / q[bad]
        P = ndimage.gaussian_filter1d(ae_of(Y)[0], 3.0, axis=0, mode="nearest")
    return P


def side_plan(grid: Grid, F: dict, edge: np.ndarray, marker_cl: np.ndarray, fascia_side: np.ndarray,
              reuse: dict | None = None) -> dict:
    """Curves (azimuthal-equidistant coordinates), flap parameters and sizing for one side, computed in
    the canonical right-side frame. ``fascia_side``: smooth temporal-fascia indicator of this side.

    With ``reuse`` (the right side's plan) the left side takes the same cut, hinge, seam and sizing in
    the canonical frame, so both halves share one mesh layout and meet exactly at the midline; the
    flap frame (pivot height) and the pericranium outline still come from the left side's own fields."""
    UV = np.stack([grid.U, grid.V], -1)
    dom = F["domain"] & (grid.U <= 0)
    if reuse is not None:
        inc, inc_info = reuse["incision"], dict(reuse["incisionInfo"], sharedWithRight=True)
    else:
        inc, inc_info = incision_uv(grid, F, edge, marker_cl)
    e1 = inc[-1].copy()
    a = np.radians(HINGE_AZ)
    ph = edge_pol(a, edge)
    h0 = np.array([ph * np.sin(a), ph * np.cos(a)])
    n = hinge_plane(e1, h0)
    hin = great_circle_uv(n, e1, 0.0, 1.8, 3600)
    if np.linalg.norm(hin - h0, axis=1).min() > 0.05:          # wrong turning direction
        hin = great_circle_uv(n, e1, 0.0, -1.8, 3600)
    j = np.argmin(np.linalg.norm(hin - h0, axis=1))
    hin = hin[: min(len(hin), j + 200)]                          # a little past the edge
    seed_uv = ae_of(SEED_FLAP)[0]
    away = n if np.dot(ae_dir(seed_uv), n) > 0 else -n

    # regions on the analysis grid (for sizing and seeds)
    bar = rasterize(grid, inc[6:]) | rasterize(grid, hin)
    lab, _ = ndimage.label(dom & ~bar)
    fi, fj = grid.idx(seed_uv)
    flap = lab == lab[int(round(float(fi))), int(round(float(fj)))]
    r_skin = F["s1"]
    d_plane = r_skin * np.einsum("ijk,k->ij", grid.D, away)

    # flap frame: lift outward, axis along the hinge (in the hinge plane), pivot on the axis, which is
    # tangent to the skin from outside. A straight hinge cannot follow the curved skin: the axis
    # direction is chosen (rotated in the plane) to minimise the spread of skin heights below it over the
    # part of the flap that curls, and the pivot height keeps every skin point at or below the axis.
    hp_uv = hin[in_domain(hin, edge)]
    hp = C + grid.sample(r_skin, hp_uv)[:, None] * ae_dir(hp_uv)
    axis0 = normalize(hp[-1] - hp[0])
    axis0 = normalize(axis0 - away * np.dot(axis0, away))
    lift0 = np.cross(away, axis0)
    if np.dot(lift0, hp.mean(0) - C) < 0:
        axis0, lift0 = -axis0, -lift0
    Pfl = (C + r_skin[..., None] * grid.D)[flap & (d_plane < FLAP_RADIUS * FLAP_ANGLE)]
    best = None
    for beta in np.radians(np.arange(-45.0, 45.1, 2.5)):
        ax_ = np.cos(beta) * axis0 + np.sin(beta) * lift0
        li_ = np.cross(away, ax_)
        hh = (Pfl - C) @ li_
        score = float(np.percentile(hh.max() - hh, 90))
        if best is None or score < best[0]:
            best = (score, ax_, li_)
    _, axis, lift = best
    hmax = float(max(((hp - C) @ lift).max(), ((Pfl - C) @ lift).max()))
    pivot = C + hmax * lift + axis * float((hp.mean(0) - C) @ axis)
    # optional refinement data: local skin height along the axis (quadratic in the axial coordinate)
    a_s = (Pfl - pivot) @ axis
    h_s = (Pfl - pivot) @ lift
    bins = np.linspace(a_s.min(), a_s.max(), 25)
    k = np.clip(np.digitize(a_s, bins) - 1, 0, len(bins) - 2)
    top = np.array([h_s[k == i].max() if (k == i).any() else np.nan for i in range(len(bins) - 1)])
    mid = 0.5 * (bins[1:] + bins[:-1])
    ok = np.isfinite(top)
    coef = np.polyfit(mid[ok], top[ok], 2)
    flap_params = {"pivot": pivot, "axis": axis, "away": away, "lift": lift,
                   "radius": FLAP_RADIUS, "angle": FLAP_ANGLE,
                   "hingeProfile": {"c": [float(coef[2]), float(coef[1]), float(coef[0])],
                                    "range": [float(a_s.min()), float(a_s.max())]}}

    # operative seam: the incision offset posteriorly, continued to the midline and past the edge
    core = inc[6:]
    r_inc = grid.sample(r_skin, core)
    seam = remove_loops(offset_curve(core, OP_MARGIN / r_inc, seed_uv, away_side=True))
    seam = ndimage.gaussian_filter1d(resample(seam, 0.002), 10, axis=0, mode="nearest")
    seam = extend(seam, start=0.15, end=0.0)
    t_end = normalize(seam[-1] - seam[-30])
    tail = seam[-1] + np.outer(np.linspace(0.004, 0.6, 150), t_end)
    tail = tail[: np.argmax(tail[:, 0] > 0.05) + 1] if (tail[:, 0] > 0.05).any() else tail
    seam = np.concatenate([seam, tail])

    # pericranium: boundary loops of its indicator, and the pericranial cut
    peri_ind = np.where(in_domain(UV, edge, margin=-PERI_EDGE_INSET), F["peri"], 0.0)
    loops = [smooth_loop(L, 2.0) for L in contours(peri_ind, 0.5, grid.g, grid.g)
             if len(L) > 40 and np.allclose(L[0], L[-1])]
    loops = [L for L in loops if np.abs(polygon_area(L[:-1])) > 0.01]
    cut = remove_loops(offset_curve(core, PERI_CUT_OFFSET / r_inc, seed_uv, away_side=False, sigma=4.0))
    cut = clear_window(cut, F, grid)
    cut = extend(cut, start=0.1, end=0.08)

    # sizing (3D spacing in display units) on the analysis grid
    lab2, _ = ndimage.label(dom & ~(bar | rasterize(grid, seam)))
    cx = ae_of(SEED_CONTEXT)[0]
    fi, fj = grid.idx(cx)
    ctx = lab2 == lab2[int(round(float(fi))), int(round(float(fj)))]
    cell = H * np.nanmedian(r_skin[dom])
    hgrid = np.full(grid.shape, H_CTX)

    def lower(mask, value):
        nonlocal hgrid
        if mask.any():
            hgrid = np.minimum(hgrid, value + H_GROW * ndimage.distance_transform_edt(~mask) * cell)
    lower(dom & ~ctx, H_OP)
    lower(flap, H_FLAP)
    lower(flap & (d_plane < FLAP_RADIUS * FLAP_ANGLE + 0.1), H_CURL)
    lower(rasterize(grid, inc[6:]), H_FINE)
    lower(rasterize(grid, cut), H_FINE * 1.2)
    if reuse is not None:
        hgrid, flap, ctx, seam, cut = reuse["h3d"], reuse["flapMask"], reuse["ctxMask"], reuse["seam"], reuse["periCut"]
    return {"incision": inc, "incisionInfo": inc_info, "e1": e1, "h0": h0, "plane": n, "hinge": hin,
            "flap": flap_params, "seam": seam, "periLoops": loops, "periCut": cut, "h3d": hgrid,
            "flapMask": flap, "ctxMask": ctx, "seedFlap": seed_uv, "seedCtx": cx}


def polygon_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ======================================================================= meshing one side
class SurfaceMap:
    """Stereographic-plane points -> 3D on a radius field (canonical frame)."""

    def __init__(self, grid: Grid, field: np.ndarray):
        self.grid, self.field = grid, field

    def radius(self, st: np.ndarray) -> np.ndarray:
        return self.grid.sample(self.field, st_to_ae(st))

    def point(self, st: np.ndarray) -> np.ndarray:
        ae = st_to_ae(st)
        return C + self.grid.sample(self.field, ae)[:, None] * ae_dir(ae)

    def scale(self, st: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """3D length per unit length in the stereographic plane (geometric mean of the two directions)."""
        p0 = self.point(st)
        px = self.point(st + [eps, 0.0])
        py = self.point(st + [0.0, eps])
        return np.sqrt(np.linalg.norm(np.cross(px - p0, py - p0), axis=1)) / eps


def build_mesh(kind: str, plan: dict, F: dict, edge: np.ndarray, grid: Grid, seed: int) -> dict:
    """Constrained triangulation (stereographic plane) of one side's sheet: 'scalp' (shared by the skin
    slab and the galea) or 'peri'. Faces are labelled 'flap', 'op' (stationary operative) and 'ctx'."""
    smap = SurfaceMap(grid, F["s1"] if kind == "scalp" else F["p1"])

    def size(st):
        return POISSON_FACTOR * grid.sample(plan["h3d"], st_to_ae(st)) / np.maximum(smap.scale(st), 1e-6)

    away = plan["flap"]["away"]

    def in_dom(st):
        ae = st_to_ae(st)
        ok = in_domain(ae, edge) & (st[..., 0] <= 1e-12)
        if kind == "peri":
            ok &= (grid.sample(np.where(in_domain(np.stack([grid.U, grid.V], -1), edge,
                                                  margin=-PERI_EDGE_INSET), F["peri"], 0.0), ae) > 0.5)
        return ok

    def flap_side(st):
        return ae_dir(st_to_ae(st)) @ away > 0

    st = ae_to_st
    arr = Arrangement()
    if kind == "scalp":
        arr.add("edge", st(edge_curve(edge, 8000)), closed=True, keep=lambda m: m[0] < 0)
    else:
        for i, L in enumerate(plan["periLoops"]):
            arr.add(f"peri{i}", st(L[:-1]), closed=True, keep=lambda m: m[0] < 0)
    vv = np.linspace(-2.6, 2.6, 2601)
    arr.add("midline", st(np.stack([np.zeros_like(vv), vv], -1)), keep=lambda m: bool(in_dom(m[None])[0]))
    inside = lambda m: bool(in_dom(m[None])[0])      # noqa: E731
    if kind == "scalp":
        arr.add("incision", st(plan["incision"]), keep=inside)
    arr.add("hinge", st(plan["hinge"]), keep=inside)
    arr.add("seam", st(plan["seam"]), keep=inside)
    if kind == "peri":
        arr.add("cut", st(plan["periCut"]), keep=lambda m: inside(m) and bool(flap_side(m[None])[0]))
    jpts, pieces = arr.build()

    # constraint points and segments
    used = sorted({j for pc in pieces for j in pc["ends"] if j is not None})
    remap = {j: i for i, j in enumerate(used)}
    fixed = [jpts[j] for j in used]
    segs = []
    for pc in pieces:
        Q = resample_by_size(pc["P"], lambda q: 0.9 * size(q))
        ids = []
        for k, q in enumerate(Q):
            if k == 0 and pc["ends"][0] is not None:
                ids.append(remap[pc["ends"][0]])
            elif k == len(Q) - 1 and pc["ends"][1] is not None:
                ids.append(remap[pc["ends"][1]])
            else:
                ids.append(len(fixed))
                fixed.append(q)
        segs += [(ids[k], ids[k + 1]) for k in range(len(ids) - 1) if ids[k] != ids[k + 1]]
    fixed = np.array(fixed)
    allc = np.concatenate([p["P"] for p in pieces])
    lo, hi = allc.min(0) - 0.05, allc.max(0) + 0.05
    rng = np.random.default_rng(seed)
    interior = poisson_points(lo, hi, size, in_dom, rng)
    P, S, segs, nfixed = conforming_delaunay(fixed, np.array(segs), interior, size)
    # orient triangles counter-clockwise in the plane
    a, b, c = P[S[:, 0]], P[S[:, 1]], P[S[:, 2]]
    cw = ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) < 0
    S = S.copy()
    S[cw] = S[cw][:, [0, 2, 1]]
    cen = P[S].mean(1)
    keep = in_dom(cen)
    lab = flood_faces(S, segs)
    S, lab, cen = S[keep], lab[keep], cen[keep]

    def face_of(uv):
        q = ae_to_st(uv)[None]
        d = np.linalg.norm(cen - q, axis=1)
        return lab[int(np.argmin(d))]
    seed_flap = plan["seedFlap"] if kind == "scalp" else ae_of(SEED_PERI_FLAP)[0]
    f_flap = face_of(seed_flap)
    f_ctx = face_of(plan["seedCtx"])
    region = np.where(lab == f_flap, "flap", np.where(lab == f_ctx, "ctx", "op"))
    # small stray faces on the far side of the seam (pericranium islands) belong to the context
    if kind == "peri":
        for f in np.unique(lab):
            if f in (f_flap, f_ctx):
                continue
            m = lab == f
            if not flap_side(cen[m]).any() and point_behind_seam(cen[m].mean(0), plan):
                region[m] = "ctx"
    return {"P": P, "S": S, "region": region, "segs": segs, "faces": lab, "nfixed": nfixed, "kind": kind}


def point_behind_seam(st_point: np.ndarray, plan: dict) -> bool:
    """True if a plane point lies on the posterior (context) side of the operative seam."""
    seam = ae_to_st(plan["seam"])
    k = int(np.argmin(np.linalg.norm(seam - st_point, axis=1)))
    k = min(max(k, 1), len(seam) - 2)
    t = seam[k + 1] - seam[k - 1]
    nrm = np.array([-t[1], t[0]])
    ref = ae_to_st(plan["seedCtx"])
    return bool(np.sign(np.dot(st_point - seam[k], nrm)) == np.sign(np.dot(ref - seam[k], nrm)))


# ======================================================================= slabs -> triangles
def normal_field(grid: Grid, r: np.ndarray) -> np.ndarray:
    """Outward unit normal of the surface C + r * dir on the analysis grid."""
    P = C + r[..., None] * grid.D
    n = normalize(np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1)))
    flip = np.einsum("ijk,ijk->ij", n, grid.D) < 0
    n[flip] *= -1
    return n


def sample_normals(grid: Grid, nf: np.ndarray, ae: np.ndarray) -> np.ndarray:
    n = np.stack([grid.sample(nf[..., k], ae) for k in range(3)], -1)
    return normalize(n)


def slab_triangles(grid: Grid, mesh: dict, region: str, top: np.ndarray, bot: np.ndarray,
                   ntop: np.ndarray, nbot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed slab for the triangles of ``region``: top surface (normals outward), bottom surface
    (normals inward) and walls along every boundary edge of the region. Unindexed (pos, nrm)."""
    T = mesh["S"][mesh["region"] == region]
    used, Ti = np.unique(T, return_inverse=True)
    Ti = Ti.reshape(-1, 3)
    ae = st_to_ae(mesh["P"][used])
    D = ae_dir(ae)
    Vt = C + grid.sample(top, ae)[:, None] * D
    Vb = C + grid.sample(bot, ae)[:, None] * D
    Nt = sample_normals(grid, ntop, ae)
    Nb = -sample_normals(grid, nbot, ae)
    # orient top triangles outward
    fn = np.cross(Vt[Ti[:, 1]] - Vt[Ti[:, 0]], Vt[Ti[:, 2]] - Vt[Ti[:, 0]])
    inward = np.einsum("ij,ij->i", fn, Vt[Ti].mean(1) - C) < 0
    Ti = Ti.copy()
    Ti[inward] = Ti[inward][:, [0, 2, 1]]
    pos = [Vt[Ti].reshape(-1, 3), Vb[Ti[:, [0, 2, 1]]].reshape(-1, 3)]
    nrm = [Nt[Ti].reshape(-1, 3), Nb[Ti[:, [0, 2, 1]]].reshape(-1, 3)]
    # walls: boundary edges directed as in the (outward) top triangles
    E = np.concatenate([Ti[:, [0, 1]], Ti[:, [1, 2]], Ti[:, [2, 0]]])
    key = np.sort(E, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    B = E[cnt[inv] == 1]
    a, b = B[:, 0], B[:, 1]
    # face normal of each wall quad (pointing away from the region)
    fnw = normalize(np.cross(Vb[b] - Vt[a], Vt[b] - Vt[a]))
    # smooth wall normals per vertex where the boundary bends gently (keep corners sharp)
    acc = np.zeros_like(Vt)
    np.add.at(acc, a, fnw)
    np.add.at(acc, b, fnw)
    vn = normalize(acc)
    na = np.where((np.einsum("ij,ij->i", vn[a], fnw) > 0.8)[:, None], vn[a], fnw)
    nb = np.where((np.einsum("ij,ij->i", vn[b], fnw) > 0.8)[:, None], vn[b], fnw)
    # walls stop WALL_INSET short of both surfaces, so where two records meet along a seam their walls
    # never reach the visible surface (no hairline at the seam); at free edges the gap is sub-pixel
    rt_, rb_ = np.linalg.norm(Vt - C, axis=1), np.linalg.norm(Vb - C, axis=1)
    inset = np.minimum(WALL_INSET, 0.25 * (rt_ - rb_))[:, None]
    Wt = C + (rt_[:, None] - inset) * D
    Wb = C + (rb_[:, None] + inset) * D
    q1 = np.stack([Wt[a], Wb[b], Wt[b]], 1)
    q2 = np.stack([Wt[a], Wb[a], Wb[b]], 1)
    n1 = np.stack([na, nb, nb], 1)
    n2 = np.stack([na, na, nb], 1)
    pos += [q1.reshape(-1, 3), q2.reshape(-1, 3)]
    nrm += [n1.reshape(-1, 3), n2.reshape(-1, 3)]
    return np.concatenate(pos), np.concatenate(nrm)


# ======================================================================= drape target (optional)
# The FLAP CONTRACT folds a flap rigidly beyond its curl; on a strongly curved head a large scalp flap then
# stands far off the face. For renderers that want the reflected flap to lie on the face, each flap
# record also carries ``foldData`` (same layout as ``data``): the reflected rest pose at progress 1,
# precomputed here. Near the hinge it is the contract curl (radius FLAP_RADIUS, angle pi) taken about the
# local skin height, so the roll is the same size along the whole hinge; beyond the roll the flap is
# mirrored across the hinge plane (which passes through the shell centre, so the mirror of a near-
# spherical shell lies on the shell) and laid, layer order reversed, on a smooth landing surface: the
# stationary skin, or over the face a closed envelope of the facial skeleton plus 5 mm of soft tissue.
FOLD_TARGETS = False       # experimental; see above (off: the drape still crumples at the flap's lower corner)
DRAPE_GAP = 0.3 * MM       # clearance between the landing surface and the turned-over skin
DRAPE_DECAY = 1.0          # display units over which the roll height settles onto the landing surface
DRAPE_BLEND = 0.4          # display units over which the roll's straight continuation blends into the drape
SOFT_FACE = 5.0 * MM       # soft tissue over the facial skeleton (eyelids, brow, cheek) for the landing surface


def landing_surface(grid: Grid, F: dict, bone_raw: np.ndarray, flap_mask: np.ndarray) -> np.ndarray:
    face = ndimage.grey_closing(fill_nan(bone_raw), size=(91, 91)) + SOFT_FACE
    land = np.where(F["domain"], F["s1"], face)
    land = np.fmax(land, np.where(F["domain"], -np.inf, face))
    return cover(land, sigma=12.0, clear=0.0)


def drape(pos: np.ndarray, nrm: np.ndarray, flap: dict, grid: Grid, F: dict, land: np.ndarray,
          hinge_skin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflected rest pose of flap vertices (canonical frame); see the comment above."""
    piv = np.asarray(flap["pivot"])
    aw, li, ax = (np.asarray(flap[k]) for k in ("away", "lift", "axis"))
    Rr = float(flap["radius"])
    v = pos - piv
    d, h, a = v @ aw, v @ li, v @ ax
    ah = (hinge_skin - piv) @ ax
    hh = (hinge_skin - piv) @ li
    o = np.argsort(ah)
    hs = np.interp(a, ah[o], ndimage.gaussian_filter1d(hh[o], 3))
    hl = h - hs
    base = piv + a[:, None] * ax + hs[:, None] * li
    # roll (contract curl about the local skin height), angle pi
    phi = np.clip(d / Rr, 0, np.pi)[:, None]
    c, sn = np.cos(phi), np.sin(phi)
    roll = base + Rr * sn * aw + Rr * (1 - c) * li + hl[:, None] * (-sn * aw + c * li)
    na, nl = nrm @ aw, nrm @ li
    nrest = nrm - na[:, None] * aw - nl[:, None] * li
    roll_n = nrest + na[:, None] * (c * aw + sn * li) + nl[:, None] * (-sn * aw + c * li)
    # straight continuation of the roll (rotation by pi) beyond it
    ex = np.clip(d - np.pi * Rr, 0, None)[:, None]
    rigid = base + 2 * Rr * li - ex * aw - hl[:, None] * li
    rigid_n = nrest - na[:, None] * aw - nl[:, None] * li
    # drape: mirror across the hinge plane (shifted by half the roll length), lay on the landing surface
    rho = np.linalg.norm(pos - C, axis=1)
    dirs = (pos - C) / rho[:, None]
    uv, _ = ae_of(pos)
    rs = grid.sample(F["s1"], uv)
    t = np.clip(rs - rho, 0, None)
    delta = np.pi * Rr / (2 * rs)
    dm = normalize(dirs - 2 * ((dirs @ aw) - delta)[:, None] * aw)
    uvm, _ = ae_of(C + dm)
    w = np.exp(-ex[:, 0] / DRAPE_DECAY)
    rt = grid.sample(land, uvm) + DRAPE_GAP + t + 2 * Rr * w
    drp = C + rt[:, None] * dm
    nref = nrm - 2 * (nrm @ aw)[:, None] * aw
    drp_n = nref - 2 * np.einsum("ij,ij->i", nref, dm)[:, None] * dm
    sb = np.clip((d - np.pi * Rr) / DRAPE_BLEND, 0, 1)[:, None]
    sb = sb * sb * (3 - 2 * sb)
    far = rigid * (1 - sb) + drp * sb
    far_n = normalize(rigid_n * (1 - sb) + drp_n * sb)
    inroll = (d <= np.pi * Rr)[:, None]
    out = np.where(inroll, roll, far)
    out_n = np.where(inroll, normalize(roll_n), far_n)
    still = (d <= 0)[:, None]
    return np.where(still, pos, out), np.where(still, nrm, out_n)


def encode_fold(pos: np.ndarray, nrm: np.ndarray) -> str:
    import base64
    arr = np.concatenate([pos, normalize(nrm)], axis=1).astype("<f4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ======================================================================= records
ID_PREFIXES = ("ILL-scalp", "ILL-galea", "ILL-pericranium")
GROUPS = {"scalp": "skin", "galea": "galea", "pericranium": "pericranium"}
COLORS = {"scalp": "#d9a184", "galea": "#d8dad3", "pericranium": "#dcd6e4"}
INNER_COLORS = {"scalp": "#e6c97e"}   # subcutaneous fat on the inner surface and the cut walls
SOURCE = ("Original anatomical illustration; layered radial offset fit over the native Z-Anatomy/BodyParts3D "
          "skull and temporalis and the illustrative temporal layers (tools/anatomy_build/scalp.py); "
          "CC BY-SA 4.0")
TITLES = {
    ("scalp", "flap"): "scalp flap (skin and subcutaneous tissue)",
    ("scalp", "op"): "scalp around the incision (skin and subcutaneous tissue)",
    ("scalp", "ctx"): "scalp of the vault (skin and subcutaneous tissue)",
    ("galea", "flap"): "galea aponeurotica, scalp flap",
    ("galea", "op"): "galea aponeurotica around the incision",
    ("galea", "ctx"): "galea aponeurotica of the vault",
    ("pericranium", "flap"): "pericranial flap",
    ("pericranium", "op"): "pericranium around the craniotomy",
    ("pericranium", "ctx"): "pericranium of the vault",
}
SUFFIX = {"flap": "-flap", "op": "", "ctx": "-context"}
LAYER_TEXT = {
    "scalp": ("Skin and subcutaneous connective tissue drawn as one closed slab: outer skin surface, inner "
              "subcutaneous surface and walls along every cut or free edge (illustrative 2.5 mm skin and "
              "about 1.9 mm subcutaneous tissue; the outer surface is smoothed, so the slab is thicker over "
              "hollows such as the temporal fossa margin). The whole scalp measures about 7 mm over the vault."),
    "galea": ("Galea aponeurotica (epicranial aponeurosis), continuous over the temporal region with the "
              "temporoparietal fascia, drawn as a closed sheet 0.8 mm thick (1.2 mm as temporoparietal "
              "fascia) under the subcutaneous tissue. The loose areolar (subgaleal) plane beneath it is "
              "drawn as an empty gap of about 0.5-1 mm over the pericranium and the temporal fascia."),
    "pericranium": ("Pericranium (periosteum of the outer table) drawn as a closed sheet 0.4 mm thick on the "
                    "outer table. It is absent over the temporal fascia: its edge follows the upper and "
                    "posterior border of the illustrative temporal fascia (superior temporal line region), "
                    "where the pericranium blends with the fascia's attachment."),
}
REGION_TEXT = {
    ("scalp", "flap"): " This record is the part of the slab anterior to the curvilinear pterional incision, "
                       "reflected anteriorly over the orbital rim with the galea in the loose areolar plane.",
    ("galea", "flap"): " This record is the galea of the scalp flap, which moves with the skin.",
    ("pericranium", "flap"): " This record is an anteriorly based pericranial flap over the frontal and "
                             "frontotemporal bone exposed by the craniotomy, cut 2 mm in front of the skin "
                             "incision and along the temporal fascia and raised after the scalp flap.",
    ("scalp", "op"): " This record is the stationary scalp around the flap: posterior to the incision (about "
                     "2.5 cm) and the forehead in front of the flap's base.",
    ("galea", "op"): " This record is the stationary galea around the flap.",
    ("pericranium", "op"): " This record is the stationary pericranium around the pericranial flap.",
    ("scalp", "ctx"): " This record is the rest of the scalp of this half of the vault (context).",
    ("galea", "ctx"): " This record is the rest of the galea of this half of the vault (context).",
    ("pericranium", "ctx"): " This record is the rest of the pericranium of this half of the vault (context).",
}
CAVEAT = (" Illustrative shell fitted to the atlas skull, temporalis and illustrative temporal layers; extent, "
          "thickness and cut lines are illustrative, not segmented from source data. The incision course "
          "follows the upper part of the atlas schematic incision marking and is redrawn below it (see the "
          "build notes).")


def make_record(layer: str, region: str, side: str, pos: np.ndarray, nrm: np.ndarray) -> dict:
    right = side == "r"
    base = "ILL-" + layer + SUFFIX[region]
    rid = base if right else base + "-l"
    title = TITLES[(layer, region)]
    sname = "Right" if right else "Left"
    m = {
        "id": rid,
        "name": f"{sname} {title} — illustration",
        "nativeName": f"{sname} {title} — illustration" + (".r" if right else ".l"),
        "procedureSide": "right" if right else "left",
        "group": GROUPS[layer],
        "color": COLORS[layer],
        "source": SOURCE,
        "description": LAYER_TEXT[layer] + REGION_TEXT[(layer, region)] + CAVEAT
        + ("" if right else " Fitted on the native left atlas anatomy."),
        "illustrative": True,
        "shellCentre": [float(x) for x in C],
    }
    if layer in INNER_COLORS:
        m["innerColor"] = INNER_COLORS[layer]
    if region == "ctx":
        m["contextOnly"] = True
    if not right:
        m["baseId"] = base
        m["independentlyFitted"] = True
    encode(m, pos, nrm)
    return m


def mirror_vec(v) -> list[float]:
    return [float(-v[0]), float(v[1]), float(v[2])]


def flap_field(key: str, F: dict, side: str) -> dict:
    """FLAP CONTRACT fields (left side: x-mirrored vectors). ``hingeProfile`` is an optional extra:
    the outer skin's height along ``lift`` relative to the axis, h_s(a) = c0 + c1 a + c2 a^2 for the axial
    coordinate a = (p - pivot).axis within ``range``; a renderer may curl about pivot + h_s(a) lift (with
    h - h_s(a) in place of h) so the roll keeps the same size along the whole hinge."""
    out = {"key": key}
    for k in ("pivot", "axis", "away", "lift"):
        v = np.asarray(F[k], dtype=np.float64)
        out[k] = [round(float(x), 6) for x in (v if side == "r" else mirror_vec(v))]
    out["radius"] = float(F["radius"])
    out["angle"] = round(float(F["angle"]), 6)
    if "hingeProfile" in F:
        hp = F["hingeProfile"]
        out["hingeProfile"] = {"c": [round(float(x), 6) for x in hp["c"]],
                               "range": [round(float(x), 4) for x in hp["range"]]}
    return out


def mirror_fields(grid: Grid, L: dict) -> dict:
    return {k: (grid.mirror(v) if isinstance(v, np.ndarray) and v.shape == grid.shape else v) for k, v in L.items()}


def build_side(grid: Grid, L: dict, raw: dict, edge: np.ndarray, marker_cl: np.ndarray, side: str,
               right: dict | None = None) -> tuple[list, dict]:
    F = L if side == "r" else mirror_fields(grid, L)
    df = raw["df_r"] if side == "r" else grid.mirror(raw["df_l"])
    plan = side_plan(grid, F, edge, marker_cl, smooth_indicator(np.isfinite(df), 2.0),
                     reuse=None if right is None else right["plan"])
    seed = 20261110
    meshes = {"scalp": build_mesh("scalp", plan, F, edge, grid, seed) if right is None else right["meshes"]["scalp"],
              "peri": build_mesh("peri", plan, F, edge, grid, seed + 10)}
    nf = {k: normal_field(grid, F[k]) for k in ("s1", "s0", "g1", "g0", "p1", "p0")}
    bone_raw = raw["bone"] if side == "r" else grid.mirror(raw["bone"])
    land = landing_surface(grid, F, bone_raw, plan["flapMask"])
    # skin points along the whole hinge plane section (beyond the free edge too, on the extended field),
    # so every flap vertex finds the local skin height at the hinge for its axial position
    hu = great_circle_uv(plan["plane"], plan["e1"], -0.6, 1.6, 4400)
    hu = hu[np.hypot(hu[:, 0], hu[:, 1]) < POL_MAX - 0.05]
    hinge_skin = C + grid.sample(F["s1"], hu)[:, None] * ae_dir(hu)
    recs = []
    stats = {"triangles": {}}
    for layer, top, bot, mk in (("scalp", "s1", "s0", "scalp"), ("galea", "g1", "g0", "scalp"),
                                ("pericranium", "p1", "p0", "peri")):
        for region in ("flap", "op", "ctx"):
            pos, nrm = slab_triangles(grid, meshes[mk], region, F[top], F[bot], nf[top], nf[bot])
            fold = None
            if region == "flap" and FOLD_TARGETS:
                fold = drape(pos, nrm, plan["flap"], grid, F, land, hinge_skin)
            if side == "l":
                pos, nrm = mirror_x(pos, nrm)
                if fold is not None:
                    fold = mirror_x(*fold)
            m = make_record(layer, region, side, pos, nrm)
            if region == "flap":
                m["flap"] = flap_field("scalp" if layer != "pericranium" else "pericranium", plan["flap"], side)
                if fold is not None:
                    m["foldData"] = encode_fold(*fold)
            if layer == "scalp" and region == "flap":
                inc = plan["incision"]
                inc = inc[in_domain(inc, edge)]
                P3 = C + grid.sample(F["s1"], inc)[:, None] * ae_dir(inc)
                P3 = resample(P3, 0.04)
                if side == "l":
                    P3 = P3 * np.array([-1.0, 1.0, 1.0])
                m["incisionLine"] = [[round(float(x), 4) for x in p] for p in P3]
            recs.append(m)
            stats["triangles"][m["id"]] = m["count"] // 3
    stats["incision"] = plan["incisionInfo"]
    stats["flap"] = flap_field("scalp", plan["flap"], side)
    stats["plan"] = plan
    stats["meshes"] = meshes
    stats["fields"] = F
    stats["land"] = land
    return recs, stats


def apply(records: list[dict], raw: dict[str, np.ndarray] | None = None, keep_debug: bool = False) -> dict:
    """Add the scalp, galea and pericranium records (both sides). ``raw`` (from cast_fields) may be
    passed in to skip ray casting when iterating. Refuses to run twice."""
    existing = [m["id"] for m in records if m["id"].startswith(ID_PREFIXES)]
    if existing:
        raise RuntimeError(f"scalp records already present ({existing[:3]}...); run on records without them")
    grid = Grid()
    if raw is None:
        raw = cast_fields(source_meshes(records))
    edge = edge_series(raw)
    L = layer_fields(raw, edge)
    marker_cl = marker_centreline(find(records, id="MARK-incision"))
    summary: dict = {"records": [], "thicknessMm": {k: round(v / MM, 3) for k, v in {
        "skin": T_SKIN, "subcutaneous": T_SUBCUT, "subcutaneousTemporal": T_SUBCUT_TEMPORAL,
        "galea": T_GALEA, "temporoparietalFascia": T_TPF, "areolarGap": T_AREOLAR,
        "pericranium": T_PERI, "pericraniumGap": T_PERI_GAP}.items()}}
    new = []
    right = None
    for side in ("r", "l"):
        recs, stats = build_side(grid, L, raw, edge, marker_cl, side, right)
        right = stats if side == "r" else right
        new += recs
        summary[f"incision_{side}"] = stats["incision"]
        summary[f"flap_{side}"] = stats["flap"]
        if keep_debug:
            summary[f"debug_{side}"] = stats
    for m in new:
        summary["records"].append({"id": m["id"], "triangles": m["count"] // 3})
    records.extend(new)
    summary["superficialVessels"] = reseat_superficial_vessels(records, L, edge)
    # the stationary pericranium must not reach into the renderer's bone window (cylinder along x)
    for sfx in ("", "-l"):
        p, _ = decode(find(records, id="ILL-pericranium" + sfx))
        cen = p.reshape(-1, 3, 3).mean(1)
        inside = int(((window_q(cen) < 1) & (np.abs(cen[:, 0]) > 1.9)).sum())
        summary["periInWindow" + sfx] = inside
        if inside:
            raise RuntimeError(f"stationary pericranium{sfx} reaches into the bone window ({inside} triangles)")
    if keep_debug:
        summary["debug_fields"] = {"L": L, "raw": raw, "edge": edge, "grid": grid}
    return summary


# ======================================================================= superficial temporal vessels
VESSEL_NATIVE = ("Superficial temporal artery", "Frontal branch of superficial temporal artery")
VESSEL_IDS = ("ILL-vein-stv",)
VESSEL_FADE = 0.012          # radians (~6 mm) inside the free edge over which the re-seat fades in
VESSEL_NOTE = (" Re-seated by tools/anatomy_build/scalp.py where the illustrative scalp covers it: moved radially "
               "into the subcutaneous layer, resting on the galea / temporoparietal fascia (the atlas course lies "
               "at the bone / temporal-fascia level there), faded in above the zygomatic arch; course otherwise "
               "unchanged.")


def reseat_superficial_vessels(records: list[dict], L: dict, edge: np.ndarray) -> dict:
    """Move the superficial temporal artery, its frontal branch and the illustrative superficial temporal vein
    (both sides) radially so their tube centre lies on the galea outer surface plus one tube radius (inside
    the subcutaneous layer), where the scalp covers them. Each vertex moves by the offset of its local tube
    centre (mean of the welded vertices within 1.5 mm), so tube cross-sections keep their shape."""
    grid = Grid()
    out = {}
    for m in records:
        native = (m.get("nativeName") or "").rsplit(".", 1)[0]
        base = m["id"][:-2] if m["id"].endswith("-l") else m["id"]
        if native not in VESSEL_NATIVE and base not in VESSEL_IDS:
            continue
        p, n = decode(m)
        V, Fc = weld(p.astype(np.float64), 1e-6)
        tree = cKDTree(V)
        cen = np.array([V[nb].mean(0) for nb in tree.query_ball_point(V, 0.075)])
        rv = float(np.median(np.linalg.norm(V - cen, axis=1)))
        uv, rc = ae_of(cen)
        w = np.mean([in_domain(uv, edge, margin=-mg) for mg in np.linspace(0.0, VESSEL_FADE, 7)], axis=0)
        target = grid.sample(L["s0"], uv) + max(rv, 0.5 * T_SUBCUT_TEMPORAL)
        target = np.minimum(target, grid.sample(L["s1"], uv) - rv)
        delta = w * (target - rc)
        V2 = V + (delta[:, None] * normalize(cen - C))
        P2 = V2[Fc].reshape(-1, 3)
        N2 = vertex_normals(V2, Fc)[Fc].reshape(-1, 3)
        encode(m, P2, N2)
        m["description"] = (m.get("description") or "").rstrip() + VESSEL_NOTE
        uv2, r2 = ae_of(V2)
        moved = w[...] > 0.5
        below = (r2 - grid.sample(L["g1"], uv2))[np.isin(np.arange(len(V2)), np.where(moved)[0])]
        out[m["id"]] = {"tubeRadiusMm": round(rv / MM, 2), "movedFraction": round(float(moved.mean()), 3),
                        "liftMedianMm": round(float(np.median(delta[moved]) / MM), 2) if moved.any() else 0.0,
                        "minAboveGaleaMm": round(float(below.min() / MM), 2) if len(below) else None}
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit(__doc__.split("CLI: ")[1])
    records = load_records(argv[1])
    summary = apply(records)
    save_records(records, argv[2])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv)
