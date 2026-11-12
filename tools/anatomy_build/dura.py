"""Replace the flat illustrative dura discs with sheets fitted to the inner table.

The right (``ILL-dura``) and left (``ILL-dura-l``) dura records are rebuilt
independently from the native atlas bones and cortex of their own side:

1. A centre ``C`` deep in the cranial cavity is chosen on each side and rays
   are cast outward over a (theta, phi) direction grid (theta toward anterior,
   phi toward superior, both measured from the lateral direction).
2. A coarse cast finds the patch of inner table to cover: the app.js bone
   window ellipse in (y, z) enlarged by ``MARGIN`` (so the sheet runs well
   under the cut bone edge) plus the lateral sphenoid ridge / anterior
   temporal and frontal base just below the window, lateral to ``COVER_X``.
3. That patch is parametrised by a disc (an elliptical square-to-disc grid
   mapping), which gives a smooth sheet edge. Along every grid ray we record the
   inner-table hit, the outer-table hit and the outermost cortex hit.
4. Sutural gaps in the inner-table height field are filled from a median
   filter. The sheet is smoothed inside a corridor: ``OFFSET`` (0.5 mm) inside
   the inner table preferred, between ``MIN_BONE_GAP`` and ``MAX_BONE_GAP``
   inside it, and at least ``BRAIN_GAP`` outside the cortex. The atlas cortex
   crosses the atlas inner table in places; inside the app.js bone window
   (whose bone is discarded once the flap is raised and covered by the outer
   table before that) the sheet may then enter the bone slab, up to
   ``SLAB_KEEP`` inside the outer table. Outside the window it stays inside
   the inner table and such contacts are only reported. Bone just outside the
   window cylinder is drawn, and its cut face is seen from inside the
   cylinder, so a final pass (5b) removes the slab allowance from every ray
   whose sheet point would lie in the slab outside the (slightly shrunken)
   cylinder and re-smooths; the sheet then never shows as a lip over the cut
   bone edge. Cortex is cleared by ``BRAIN_GAP`` over the outermost cortex hit
   of each ray's 3x3 neighbourhood (cortex can bulge between grid rays).
   Where the sheet still does not clear visible cortex inside the window
   (at the rim, where it may not enter the slab), those cortex vertices are
   moved radially inward under the sheet by at most ``TUCK_MAX``; this is the
   only change this step makes to atlas geometry, and the affected records'
   descriptions say so.
5. The grid is triangulated with unit normals pointing outward (toward the
   bone and the surgeon), and clearances to bone and cortex are reported.

Shape and thickness are illustrative: the sheet is a single surface placed in
the epidural plane, not a segmented dura.

usage: python3 tools/anatomy_build/dura.py IN.json OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (decode, encode, find, load_records, normalize, ray_nearest,  # noqa: E402
                    save_records, vertex_normals)

# ----------------------------------------------------------------- parameters
OFFSET = 0.025        # sheet sits 0.5 mm inside the inner table (display units)
BRAIN_GAP = 0.015     # clearance (0.3 mm) kept above the cortex where space allows
MIN_BONE_GAP = 0.008  # the sheet may move this close to the bone to clear the cortex
OUTER_SEARCH = 0.45   # outer table = outermost bone hit within this distance of the inner table
SLAB_KEEP = 0.03      # inside the window the sheet stays this far inside the outer table
COVER_X = 1.75        # the sheet covers inner table lateral to |x| = COVER_X only
WINDOW_X = 1.9        # app.js discards bone with |x| > WINDOW_X inside the window ellipse
MARGIN = 1.32         # coverage = shader window ellipse scaled by this factor
WINDOW = dict(y=0.3, z=1.25, ry=1.55, rz=1.65)   # app.js bone window, (y, z) ellipse
SMOOTH_ITERS = 1000   # corridor smoothing iterations
DATA_WEIGHT = 0.008   # pull toward the preferred offset during smoothing
MAX_BONE_GAP = 0.05   # never more than 1 mm inside the inner table
SLAB_SPREAD = 8       # grid cells around cortex contacts where the sheet may enter the slab
GRID = 101            # grid points per side of the square that is mapped to a disc
TUCK_MAX = 0.14       # cortex vertices in the window may move inward by at most 2.8 mm (tuck_cortex)
WINDOW_SLAB_K = 0.99   # sheet inside the bone slab must lie within this scale of the window ellipse
LIP_PASSES = 6        # passes that remove slab allowance at the window rim (step 5b)
LIP_ITERS = 200       # smoothing iterations after each pass
DEBUG_DIR = None      # set to a directory to dump intermediate fields
CENTRE = np.array([0.55, 0.15, 1.0])              # |x| mirrored per side
BONES = ("Frontal bone", "Parietal bone", "Temporal bone", "Sphenoid bone",
         "Occipital bone", "Ethmoid bone", "Zygomatic bone")

SOURCE = ("Original anatomical illustration; height field fitted to the inner table of the "
          "native atlas skull (Z-Anatomy/BodyParts3D, CC BY-SA 4.0); CC BY-SA 4.0")


def _desc(side: str) -> str:
    return (f"Original dural illustration for the {side} pterional exposure: a smooth sheet cast "
            "radially onto the inner table of the native "
            f"{side} frontal, parietal, temporal and sphenoid bones and offset 0.5 mm inward, "
            "covering the bone window with a margin under the cut edge and the lateral sphenoid "
            "ridge and adjacent frontal and temporal base. Sutural gaps are bridged by smoothing. "
            "Within the bone window the sheet is kept outside the native cortex, entering the bone "
            "thickness where the atlas cortex crosses the atlas inner table (only inside the window "
            "cylinder, so it never shows over the cut bone edge). Shape and thickness "
            "are illustrative; not a native segmentation.")


# ------------------------------------------------------------------ geometry
def _dirs(theta: np.ndarray, phi: np.ndarray, sx: float) -> np.ndarray:
    """Unit directions; theta from lateral toward anterior, phi toward superior."""
    c = np.cos(phi)
    return np.stack([sx * c * np.cos(theta), np.sin(phi), c * np.sin(theta)], -1)


def _angles(v: np.ndarray, sx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta, phi, lateral component) of vectors from the centre."""
    lat = sx * v[..., 0]
    th = np.arctan2(v[..., 2], lat)
    ph = np.arctan2(v[..., 1], np.hypot(lat, v[..., 2]))
    return th, ph, lat


class Caster:
    """Ray caster from a fixed centre, with triangles binned by their angular bounding boxes."""

    def __init__(self, C: np.ndarray, tris: np.ndarray, sx: float):
        self.C, self.sx = C, sx
        rel = tris - C
        th, ph, lat = _angles(rel, sx)
        # keep triangles on the lateral half-space that subtend a small solid angle
        ok = (lat > 0.05).all(1) & ((th.max(1) - th.min(1)) < 0.6) & ((ph.max(1) - ph.min(1)) < 0.6)
        self.tris = tris[ok]
        self.lo = np.stack([th[ok].min(1), ph[ok].min(1)], 1)
        self.hi = np.stack([th[ok].max(1), ph[ok].max(1)], 1)
        n = np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])
        self.nrm = normalize(n)

    def cast(self, theta: np.ndarray, phi: np.ndarray, tile: int = 8, mode: str = "first",
             limit: np.ndarray | None = None, start: np.ndarray | None = None):
        """Cast rays on a (H, W) angle grid. mode 'first' = nearest hit, 'last' = farthest hit
        closer than ``limit`` (per ray, default 8 units). ``start`` (per ray) moves the
        origin of 'first' rays outward along the ray; distances are still measured from C.

        Returns (t, normal) arrays of shape (H, W) / (H, W, 3); t is inf for misses.
        """
        H, W = theta.shape
        D = _dirs(theta, phi, self.sx)
        T = np.full((H, W), np.inf)
        N = np.zeros((H, W, 3))
        eps = 1e-3
        for i in range(0, H, tile):
            for j in range(0, W, tile):
                th = theta[i:i + tile, j:j + tile]
                ph = phi[i:i + tile, j:j + tile]
                sel = ((self.hi[:, 0] >= th.min() - eps) & (self.lo[:, 0] <= th.max() + eps)
                       & (self.hi[:, 1] >= ph.min() - eps) & (self.lo[:, 1] <= ph.max() + eps))
                if not sel.any():
                    continue
                tri = self.tris[sel]
                d = D[i:i + tile, j:j + tile].reshape(-1, 3)
                V = tri.reshape(-1, 3)
                F = np.arange(len(V)).reshape(-1, 3)
                if mode == "first":
                    s0 = (np.zeros(len(d)) if start is None
                          else np.nan_to_num(start[i:i + tile, j:j + tile].reshape(-1), posinf=8.0))
                    t, k = ray_nearest(self.C + s0[:, None] * d, d, V, F)
                    t = t + s0
                else:  # farthest hit: cast from far outside back toward the centre
                    far = (np.full(len(d), 8.0) if limit is None
                           else np.minimum(limit[i:i + tile, j:j + tile].reshape(-1), 8.0))
                    t2, k = ray_nearest(self.C + far[:, None] * d, -d, V, F)
                    t2 = np.where(t2 < far, t2, np.inf)
                    k = np.where(np.isfinite(t2), k, -1)
                    t = np.where(np.isfinite(t2), far - t2, np.inf)
                hit = k >= 0
                nn = np.zeros((len(d), 3))
                nn[hit] = self.nrm[np.flatnonzero(sel)[k[hit]]]
                h, w = th.shape
                T[i:i + h, j:j + w] = t.reshape(h, w)
                N[i:i + h, j:j + w] = nn.reshape(h, w, 3)
        return T, N


def _side_records(records: list[dict], sx: float):
    suffix = ".r" if sx < 0 else ".l"
    bones = [m for m in records if m["group"] == "bone" and any(
        m.get("nativeName", "") in (b, b + suffix) for b in BONES)]
    soft = [m for m in records if m["group"] == "brain"
            and m["bounds"][1 if sx > 0 else 0][0] * sx > 0.5]
    return bones, soft


def _tris(ms: list[dict]) -> np.ndarray:
    return np.concatenate([decode(m)[0].reshape(-1, 3, 3) for m in ms])


def _in_window(P: np.ndarray, k: float) -> np.ndarray:
    return (((P[..., 1] - WINDOW["y"]) / (WINDOW["ry"] * k)) ** 2
            + ((P[..., 2] - WINDOW["z"]) / (WINDOW["rz"] * k)) ** 2) < 1


def _coverage(P: np.ndarray, sx: float) -> np.ndarray:
    """Inner-table points the sheet must cover."""
    win = _in_window(P, MARGIN) & (P[..., 0] * sx > COVER_X)
    # lateral sphenoid ridge and adjacent frontal / temporal base below the window
    basal = ((P[..., 1] > -1.75) & (P[..., 1] < -0.2) & (P[..., 2] > -0.2) & (P[..., 2] < 3.0)
             & (P[..., 0] * sx > COVER_X))
    return win | basal


def _disc_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Elliptical square-to-disc mapping of an n x n grid: returns (rho, psi)."""
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    a = u * np.sqrt(1 - v * v / 2)
    b = v * np.sqrt(1 - u * u / 2)
    return np.hypot(a, b), np.arctan2(b, a)


def build_side(records: list[dict], sx: float, log: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one side's sheet. Returns welded (V, F, N); diagnostics go into ``log``."""
    from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                               gaussian_filter1d, label, maximum_filter, median_filter)

    C = CENTRE * np.array([sx, 1.0, 1.0])
    bones, soft = _side_records(records, sx)
    bone = Caster(C, _tris(bones), sx)
    brain = Caster(C, _tris(soft), sx)

    # 1. coarse cast: which directions land on inner table we need to cover
    st = 0.02
    th1, ph1 = np.meshgrid(np.arange(-1.3, 1.55, st), np.arange(-1.25, 1.25, st))
    t1, _ = bone.cast(th1, ph1)
    P1 = C + t1[..., None] * _dirs(th1, ph1, sx)
    cov = np.isfinite(t1) & _coverage(np.where(np.isfinite(P1), P1, 0), sx)
    # close sutural misses, keep the component containing the window centre
    cov = binary_fill_holes(binary_closing(cov, iterations=3))
    lab, _ = label(cov)
    wc = np.array([-3.2 * -sx, WINDOW["y"], WINDOW["z"]]) - C
    tc, pc, _ = _angles(wc, sx)
    ci = np.unravel_index(np.argmin((th1 - tc) ** 2 + (ph1 - pc) ** 2), th1.shape)
    cov = lab == lab[ci]
    thc, phc = th1[cov].mean(), ph1[cov].mean()

    # 2. boundary radius per polar angle around (thc, phc)
    psi = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    rho = np.arange(0, 2.0, st / 2)
    TH = thc + rho[None] * np.cos(psi)[:, None]
    PH = phc + rho[None] * np.sin(psi)[:, None]
    ii = np.clip(np.round((PH - ph1[0, 0]) / st).astype(int), 0, th1.shape[0] - 1)
    jj = np.clip(np.round((TH - th1[0, 0]) / st).astype(int), 0, th1.shape[1] - 1)
    inside = cov[ii, jj] & (PH >= ph1[0, 0]) & (PH <= ph1[-1, 0]) & (TH >= th1[0, 0]) & (TH <= th1[0, -1])
    first_out = np.where((~inside).any(1), (~inside).argmax(1), len(rho))
    rmax = rho[np.maximum(first_out - 1, 1)] + st
    # smooth circularly (removes narrow tabs along sutures) with a small allowance
    rmax = gaussian_filter1d(rmax, 8, mode="wrap") + 0.02

    # 3. final disc grid
    r01, ps = _disc_grid(GRID)
    rb = np.interp(ps, psi, rmax, period=2 * np.pi)
    TH = thc + r01 * rb * np.cos(ps)
    PH = phc + r01 * rb * np.sin(ps)
    D = _dirs(TH, PH, sx)
    tb, nb = bone.cast(TH, PH)
    # ray directions are outward; incidence factors for normal-distance offsets
    cb = np.clip(np.abs(np.einsum("ijk,ijk->ij", nb, D)), 0.6, 1.0)

    # 4. bone height field: fill misses and sutural spikes
    miss = ~np.isfinite(tb)
    med = median_filter(np.where(miss, np.nanmedian(np.where(miss, np.nan, tb)), tb), size=7, mode="nearest")
    spike = miss | (tb > med + 0.06)
    tb_clean = np.where(spike, med, tb)
    cb = np.where(spike, 1.0, cb)
    # outer table: outermost bone surface within OUTER_SEARCH of the inner table
    to, _ = bone.cast(TH, PH, mode="last", limit=tb_clean + OUTER_SEARCH)
    thick = np.where(spike | ~np.isfinite(to), 0.1, to - tb_clean)
    thick = np.clip(median_filter(thick, size=5, mode="nearest"), 0.03, OUTER_SEARCH)
    # Inside the shader bone window the bone is discarded once the flap is raised, and the
    # outer table hides everything beneath it before that. There the sheet may enter the bone
    # slab to stay outside atlas cortex that crosses the inner table (the atlas brain and
    # skull do not match exactly). Outside the window it stays inside the inner table.
    Pin = C + tb_clean[..., None] * D
    win = _in_window(Pin, 1.0) & (Pin[..., 0] * sx > WINDOW_X)
    slab = np.where(win, tb_clean + thick - SLAB_KEEP, tb_clean)
    tr, _ = brain.cast(TH, PH, mode="last", limit=slab)
    hi_pref = tb_clean - OFFSET / cb            # preferred: 0.5 mm inside the inner table
    hi_hard = tb_clean - MIN_BONE_GAP / cb      # normal limit toward the bone
    lo_far = tb_clean - MAX_BONE_GAP / cb       # never farther from the bone than this
    # cortex can bulge between neighbouring grid rays, so the lower bound uses the outermost
    # cortex hit of each ray's 3x3 neighbourhood
    tr_nb = maximum_filter(np.where(np.isfinite(tr), tr, -np.inf), size=3, mode="nearest")
    lo = np.where(np.isfinite(tr_nb), tr_nb + BRAIN_GAP, 0.0)
    cap = np.maximum(hi_hard, slab)
    # the allowance is widened a little around contacts so the sheet can stay smooth
    need = binary_dilation(lo > hi_hard, iterations=SLAB_SPREAD) & win
    ub = np.where(need, cap, hi_hard)
    conflict = lo > ub                          # cortex the sheet cannot clear
    lb = np.minimum(np.maximum(lo, lo_far), ub)

    # 5. smooth sheet inside the corridor [lb, ub], pulled toward the preferred offset; where
    # the cortex lifts it, the clamp and the smoothing spread the lift into a gentle swell
    target = np.minimum(hi_pref, ub)
    r = target.copy()
    for _ in range(SMOOTH_ITERS):
        p = np.pad(r, 1, mode="edge")
        lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4 - r
        r = np.clip(r + 0.5 * lap + DATA_WEIGHT * (target - r), lb, ub)
    # 5b. Bone just outside the window cylinder stays, and its cut face is drawn from the inside,
    # so sheet that runs inside the bone slab must lie within the cylinder or it shows as a white
    # lip over the cut bone edge (anterior-inferior margin, where the cut runs obliquely through
    # the thick greater wing and sphenoid ridge). Rays whose sheet point would sit in the slab
    # outside the (slightly shrunken) cylinder lose the slab allowance and the sheet is
    # re-smoothed; a few passes settle it.
    lip_rays = np.zeros(r.shape, bool)
    for _ in range(LIP_PASSES):
        Pv = C + r[..., None] * D
        lip = (r > hi_hard + 1e-4) & ~(_in_window(Pv, WINDOW_SLAB_K) & (Pv[..., 0] * sx > WINDOW_X))
        if not lip.any():
            break
        lip_rays |= lip
        ub = np.where(lip, hi_hard, ub)
        lb = np.minimum(lb, ub)
        for _ in range(LIP_ITERS):
            p = np.pad(r, 1, mode="edge")
            lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4 - r
            r = np.clip(r + 0.5 * lap + DATA_WEIGHT * (target - r), lb, ub)
    Pv = C + r[..., None] * D
    lip_left = (r > hi_hard + 1e-4) & ~(_in_window(Pv, WINDOW_SLAB_K) & (Pv[..., 0] * sx > WINDOW_X))
    into_bone = r > hi_hard + 1e-4

    P = C + r[..., None] * D

    # 6. triangulate (grid is already a disc, no masking), orient outward
    idx = np.arange(GRID * GRID).reshape(GRID, GRID)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    V = P.reshape(-1, 3)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    out = V[F].mean(1) - C
    flip = np.einsum("ij,ij->i", fn, out) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    # drop degenerate triangles from the disc mapping corners
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    F = F[area > 1e-10]
    N = vertex_normals(V, F)
    N = np.where((np.einsum("ij,ij->i", N, V - C) < 0)[:, None], -N, N)

    # 7. report
    gap_bone = tb_clean - r            # radial gap to inner table
    gap_brain = np.where(np.isfinite(tr), r - tr, np.inf)
    near = np.isfinite(tr) & (tb_clean - tr < BRAIN_GAP + MIN_BONE_GAP)
    log.update({
        "centre": C.round(3).tolist(),
        "triangles": int(len(F)),
        "grid": GRID,
        "angular_extent_rad": [round(float(rmax.min()), 3), round(float(rmax.max()), 3)],
        "bone_misses_filled": int(miss.sum()),
        "bone_spikes_filled": int((spike & ~miss).sum()),
        "rays": int(r.size),
        "rays_cortex_within_0.018_of_bone": int(near.sum()),
        "rays_cortex_beyond_inner_table": int((np.isfinite(tr) & (tr > tb_clean)).sum()),
        "rays_sheet_moved_into_bone_slab": int(into_bone.sum()),
        "rays_cortex_unresolved_in_window": int((conflict & win).sum()),
        "rays_cortex_unresolved_under_bone": int((conflict & ~win).sum()),
        "bone_thickness_median": round(float(np.median(thick)), 4),
        "radial_gap_to_bone": {"min": round(float(gap_bone.min()), 4),
                               "median": round(float(np.median(gap_bone)), 4),
                               "p95": round(float(np.percentile(gap_bone, 95)), 4)},
        "radial_gap_to_cortex": {"min": round(float(gap_brain[np.isfinite(gap_brain)].min()), 4),
                                 "median": round(float(np.median(gap_brain[np.isfinite(gap_brain)])), 4),
                                 "below_0.0095": int((gap_brain < 0.0095).sum())},
        "bounds": [V.min(0).round(3).tolist(), V.max(0).round(3).tolist()],
        "rays_slab_allowance_removed_at_rim": int(lip_rays.sum()),
        "rays_in_slab_outside_window_after": int(lip_left.sum()),
    })
    log["_check"] = (C, V, F, bone, brain)
    log["_dbg"] = dict(tb=tb, tb_clean=tb_clean, thick=thick, tr=tr, hi=ub, lo=lb, r=r, conflict=conflict, TH=TH, PH=PH,
                       cov=cov, th1=th1, ph1=ph1)
    return V, F, N


def check_penetration(C, V, rb, caster: Caster, sx: float, outside: bool) -> dict:
    """Vertices of ``caster`` triangles on the wrong side of the sheet, radially from C.

    outside=True: bone vertices must not lie inside (closer to C than) the sheet.
    outside=False: cortex vertices lying between the sheet and the inner table
    (radius ``rb`` per sheet vertex) are counted; cortex behind the bone is ignored.
    The sheet radius is interpolated linearly over its (theta, phi) parametrisation.
    """
    from scipy.interpolate import LinearNDInterpolator
    vt, vp, _ = _angles(V - C, sx)
    f = LinearNDInterpolator(np.stack([vt, vp], 1), np.stack([np.linalg.norm(V - C, axis=1), rb], 1))
    pts, first = np.unique(caster.tris.reshape(-1, 3), axis=0, return_index=True)
    facing = np.einsum("ij,ij->i", np.repeat(caster.nrm, 3, 0)[first], pts - C)
    th, ph, lat = _angles(pts - C, sx)
    rs, rbone = f(th, ph).T
    ok = np.isfinite(rs) & (lat > 0)
    rr = np.linalg.norm(pts - C, axis=1)
    depth = (rs - rr) if outside else (rr - rs)
    bad = ok & (depth > 0) & (outside | (rr < rbone))
    # bone inside the shader window is discarded once the flap is raised and is covered by
    # the outer table before that, so only contacts outside it can show
    cut = _in_window(pts, 1.0) & (pts[:, 0] * sx > WINDOW_X)

    def stat(mask):
        return {"vertices": int(mask.sum()),
                "max_depth": round(float(depth[mask].max()) if mask.any() else 0.0, 4)}
    if outside:  # outward-facing bone (outer table, ridge flanks) inside the sheet = poke-through
        return {"inside_window_cut": stat(bad & cut), "outside_window_cut": stat(bad & ~cut),
                "outward_facing_outside_window_cut": stat(bad & ~cut & (facing > 0))}
    return {"inside_window_cut": stat(bad & cut), "outside_window_cut": stat(bad & ~cut)}


def tuck_cortex(records: list[dict], sx: float, C: np.ndarray, V: np.ndarray) -> dict:
    """Move visible cortex vertices that lie outside (or within BRAIN_GAP/2 of) the sheet inside the
    window cylinder radially inward to BRAIN_GAP under it, by at most TUCK_MAX.

    At the window rim the sheet may not enter the bone slab (step 5b), so where the atlas cortex
    crosses the atlas inner table there a few cortex vertices would show through the closed dura.
    Each vertex moves as a function of its position only, so coincident vertices of the unindexed
    triangles move together and the surface stays closed. Normals are left as they are.
    """
    from scipy.interpolate import LinearNDInterpolator
    vt, vp, _ = _angles(V - C, sx)
    sheet = LinearNDInterpolator(np.stack([vt, vp], 1), np.linalg.norm(V - C, axis=1))
    out = {"vertices_moved": 0, "max_move_mm": 0.0, "records": [], "left_outside": 0}
    for m in records:
        if m["group"] != "brain" or m.get("contextOnly") or m["bounds"][1 if sx > 0 else 0][0] * sx <= 0.5:
            continue
        p, n = decode(m)
        cyl = _in_window(p, 1.0) & (p[:, 0] * sx > WINDOW_X)
        if not cyl.any():
            continue
        d = p - C
        th, ph, _ = _angles(d, sx)
        r = np.linalg.norm(d, axis=1)
        rs = np.where(cyl, sheet(th, ph), np.nan)
        bad = np.isfinite(rs) & (r > rs - BRAIN_GAP / 2)
        if not bad.any():
            continue
        new_r = rs - BRAIN_GAP
        ok = bad & (r - new_r <= TUCK_MAX)
        out["left_outside"] += int((bad & ~ok).sum())
        if not ok.any():
            continue
        p[ok] = C + d[ok] / r[ok, None] * new_r[ok, None]
        move = float((r - new_r)[ok].max()) * 20
        anchor = m["anchor"]
        encode(m, p, n)
        m["anchor"] = anchor
        uniq = len(np.unique(np.round(p[ok], 5), axis=0))
        out["vertices_moved"] += uniq
        out["max_move_mm"] = round(max(out["max_move_mm"], move), 2)
        out["records"].append(m["id"])
        note = (" A few vertices at the edge of the illustrative bone window are moved inward (at most "
                f"{TUCK_MAX * 20:.1f} mm) so the atlas cortex stays under the illustrative dura there.")
        if note not in m.get("description", ""):
            m["description"] = m.get("description", "") + note
    return out


def apply(records: list[dict]) -> dict:
    summary = {}
    for rid, sx, side in (("ILL-dura", -1.0, "right"), ("ILL-dura-l", 1.0, "left")):
        m = find(records, id=rid)
        log: dict = {}
        V, F, N = build_side(records, sx, log)
        C, _, _, bone, brain = log.pop("_check")
        dbg = log.pop("_dbg")
        if DEBUG_DIR:
            np.savez(Path(DEBUG_DIR) / f"{rid}.npz", **dbg)
        rb = dbg["tb_clean"].reshape(-1)  # inner-table radius per sheet vertex
        log["bone_vertices_inside_sheet"] = check_penetration(C, V, rb, bone, sx, outside=True)
        log["cortex_vertices_between_sheet_and_bone"] = check_penetration(C, V, rb, brain, sx, outside=False)
        log["cortex_tucked_under_sheet_in_window"] = tuck_cortex(records, sx, C, V)
        idx = F.reshape(-1)
        encode(m, V[idx], N[idx])
        # encode() puts the anchor at the bounding-box centre, which for a curved sheet lies
        # inside the head; keep it on the sheet at the window centre, as for the old disc
        aim = normalize(np.array([3.1 * sx, WINDOW["y"], WINDOW["z"]]) - C)
        m["anchor"] = V[np.argmax(normalize(V - C) @ aim)].tolist()
        m["source"] = SOURCE
        m["description"] = _desc(side)
        summary[rid] = log
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    recs = load_records(sys.argv[1])
    s = apply(recs)
    save_records(recs, sys.argv[2])
    print(json.dumps(s, indent=1))
