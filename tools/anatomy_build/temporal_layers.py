"""Temporal soft-tissue layers that follow the native temporalis surface.

Replaces the flat elliptical discs used for the five illustrative temporal
layers (right ``ILL-*`` and left ``ILL-*-l``) with sheets and lens-shaped fat
pads built as offset height fields over the native Z-Anatomy temporalis and
skull surfaces. Deep to superficial:

=================  ===========  ==================================================
id                 group        shape
=================  ===========  ==================================================
ILL-subfat         fat          deep temporal (subfascial) fat pad: lens in the
                                lower anterior fossa, between muscle and the deep
                                lamina, thickest just above the zygomatic arch
ILL-deepfascia     deepfascia   deep lamina: thin closed sheet over the whole
                                fascial footprint (superior temporal line to arch)
ILL-interfat       fat          superficial temporal (interfascial) fat pad: lens
                                between the laminae from the arch up to the split
ILL-superfascia    fascia       superficial lamina: lies on the deep lamina above
                                the split, lifted over the interfascial pad below
ILL-suprafat       fat          suprafascial loose areolar fat: thin lens over the
                                anteroinferior fascia (frontal-branch plane)
=================  ===========  ==================================================

Method. Surfaces are parametrised by direction (azimuth, elevation) from a
fixed centre medial to the fossa. Rays cast inward give the outer radius of the
temporalis and of the outermost skull bone. The fascial footprint is where the
muscle is the outermost surface above the zygomatic arch, extended a few
millimetres onto bone at the upper and posterior margins (superior temporal
line / supramastoid crest). A smoothed base surface that never dips below the
muscle or bone is offset outward layer by layer with small clearances, so the
layers cannot intersect each other or the atlas structures they rest on. Fat
pad thicknesses are smooth functions of the surface distance from the arch
and taper to zero at their edges (closed lenses); laminae are closed sheets of
constant thickness with a rim.

The left side is fitted independently on native left anatomy (mirrored into
the right-side frame for the computation and mirrored back).

Thicknesses and extents are illustrative values within published ranges, not
segmentations.

CLI: python3 tools/anatomy_build/temporal_layers.py IN.json OUT.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (decode, encode, find, load_records, mirror_x, normalize,  # noqa: E402
                    ray_nearest, save_records, triangles_near, weld)

MM = 0.05  # display units per millimetre

# Parametrisation (right-side frame; +X is patient left, so the fossa is at -X).
CENTRE = np.array([-0.5, -0.8, 0.3])
PHI = (-62.0, 78.0)      # azimuth, degrees: 0 = straight lateral, + = anterior
THETA = (-58.0, 50.0)    # elevation, degrees: + = superior
FINE = 0.6               # degrees per fine (analysis) cell
STEP = 2                 # output vertex every STEP fine cells (1.2 degrees ~ 1.4 mm)

# Layer parameters (display units).
CLEAR = 0.01             # clearance between neighbouring surfaces (0.2 mm)
LAMINA = 0.6 * MM        # each fascial lamina
SUBFAT_MAX = 3.6 * MM    # deep temporal fat pad (profile peak before edge tapers)
INTERFAT_MAX = 3.6 * MM  # superficial temporal (interfascial) fat pad
SUPRAFAT_MAX = 1.4 * MM  # suprafascial loose areolar fat
SPLIT = 25.0 * MM        # fascial split height above the arch (2.5 cm)
MARGIN = 3.0 * MM        # fascia extension beyond the muscle onto bone (STL)
SETBACK = 1.5 * MM       # sheet edge setback from bone lateral to the muscle (arch, zygoma)

BONES = ["Temporal bone.{s}", "Zygomatic bone.{s}", "Frontal bone", "Parietal bone.{s}", "Sphenoid bone"]
IDS = ["ILL-subfat", "ILL-deepfascia", "ILL-interfat", "ILL-superfascia", "ILL-suprafat"]


def smoothstep(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ sampling
class Frame:
    """Direction grid around CENTRE; positions are CENTRE + r * dir(i, j)."""

    def __init__(self) -> None:
        self.phi = np.radians(np.arange(PHI[0], PHI[1] + 1e-9, FINE))
        self.theta = np.radians(np.arange(THETA[0], THETA[1] + 1e-9, FINE))
        self.shape = (len(self.phi), len(self.theta))
        P, T = np.meshgrid(self.phi, self.theta, indexing="ij")
        self.P, self.T = P, T
        self.D = self.dirs(P, T)

    @staticmethod
    def dirs(P, T):
        return np.stack([-np.cos(T) * np.cos(P), np.sin(T), np.cos(T) * np.sin(P)], -1)

    def at(self, fi, fj):
        """Direction at fractional fine-grid indices."""
        p = np.radians(PHI[0]) + np.asarray(fi) * np.radians(FINE)
        t = np.radians(THETA[0]) + np.asarray(fj) * np.radians(FINE)
        return self.dirs(p, t)



def _cast(args) -> np.ndarray:
    """Outer radius along each frame direction (nan for misses). Runs in a worker."""
    pos, = args
    V, F = weld(pos)
    frame = Frame()
    D = frame.D.reshape(-1, 3)
    F = triangles_near(V, F, [-6.5, -5.0, -4.0], [-0.8, 3.5, 5.0])
    far = 9.0
    t, _ = ray_nearest(CENTRE + far * D, -D, V, F, chunk=48)
    r = far - t
    r[~np.isfinite(t)] = np.nan
    return r.reshape(frame.shape)


def cast_all(meshes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cast every mesh in parallel (keyed results). Deterministic: no randomness."""
    keys = list(meshes)
    workers = min(len(keys), os.cpu_count() or 1)
    if workers <= 1:
        return {k: _cast((meshes[k],)) for k in keys}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(zip(keys, ex.map(_cast, [(meshes[k],) for k in keys])))


def fill_nan(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def interp(field: np.ndarray, fi, fj) -> np.ndarray:
    return ndimage.map_coordinates(field, [np.asarray(fi), np.asarray(fj)], order=1, mode="nearest")


# ------------------------------------------------------------------ meshing
def snapped_grid(frame: Frame, sdf: np.ndarray):
    """Output vertex grid (fractional fine indices) for the region ``sdf > 0``.

    Vertices on the region boundary are moved by a few Newton steps onto the zero
    contour of ``sdf`` so the outline is smooth instead of a grid staircase.
    """
    gi = np.arange(0, frame.shape[0], STEP, dtype=np.float64)
    gj = np.arange(0, frame.shape[1], STEP, dtype=np.float64)
    FI, FJ = np.meshgrid(gi, gj, indexing="ij")
    FI0, FJ0 = FI.copy(), FJ.copy()
    mask = interp(sdf, FI, FJ) > 0
    # drop isolated vertices / one-cell slivers that cannot form a quad
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    gx = np.gradient(sdf, axis=0)
    gy = np.gradient(sdf, axis=1)
    bnd = mask & ndimage.binary_dilation(~mask, structure=np.ones((3, 3)))
    for _ in range(6):
        sv = interp(sdf, FI, FJ)
        ax = interp(gx, FI, FJ)
        ay = interp(gy, FI, FJ)
        g2 = np.maximum(ax * ax + ay * ay, 1e-12)
        FI = np.where(bnd, FI + np.clip(-sv * ax / g2, -0.35 * STEP, 0.35 * STEP), FI)
        FJ = np.where(bnd, FJ + np.clip(-sv * ay / g2, -0.35 * STEP, 0.35 * STEP), FJ)
        FI = np.clip(FI, FI0 - 0.75 * STEP, FI0 + 0.75 * STEP)
        FJ = np.clip(FJ, FJ0 - 0.75 * STEP, FJ0 + 0.75 * STEP)
    return FI, FJ, mask


def boundary_edges(F: np.ndarray) -> np.ndarray:
    """Directed boundary edges (a, b) as they appear in faces of F."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(E, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv] == 1]


def surface(frame: Frame, FI, FJ, use, radius_field: np.ndarray):
    """Triangulate the masked vertex grid with radii sampled from ``radius_field``."""
    r = interp(radius_field, FI, FJ)
    P = CENTRE + r[..., None] * frame.at(FI, FJ)
    P = np.where(use[..., None], P, np.nan)
    return P


def slab(frame: Frame, FI, FJ, use, r_bot: np.ndarray, r_top: np.ndarray, rim: bool,
         r_rim: np.ndarray | None = None):
    """Closed sheet between two radius fields. Returns unindexed (pos, nrm).

    ``rim`` adds side walls along the outline; ``r_rim`` (optional) lets the wall reach
    deeper than the sheet's underside, e.g. down to the attachment on the arch.
    """
    Pt = surface(frame, FI, FJ, use, r_top)
    Pb = surface(frame, FI, FJ, use, r_bot)
    Pr = Pb if r_rim is None else surface(frame, FI, FJ, use, r_rim)
    H, W = use.shape
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    ok = use[:-1, :-1] & use[:-1, 1:] & use[1:, :-1] & use[1:, 1:]
    F = np.concatenate([np.stack([a, c, b], -1)[ok], np.stack([b, c, d], -1)[ok]])
    # drop vertices not referenced
    used = np.unique(F)
    remap = np.full(H * W, -1)
    remap[used] = np.arange(len(used))
    F = remap[F]
    Vt = Pt.reshape(-1, 3)[used]
    Vb = Pb.reshape(-1, 3)[used]
    Vr = Pr.reshape(-1, 3)[used]
    # orient top faces outward (away from CENTRE)
    fn = np.cross(Vt[F[:, 1]] - Vt[F[:, 0]], Vt[F[:, 2]] - Vt[F[:, 0]])
    if np.einsum("ij,ij->i", fn, Vt[F].mean(1) - CENTRE).sum() < 0:
        F = F[:, [0, 2, 1]]
    Nt = vertex_normals_safe(Vt, F, outward_from=CENTRE, sign=1.0)
    Nb = vertex_normals_safe(Vb, F[:, [0, 2, 1]], outward_from=CENTRE, sign=-1.0)
    pos = [Vt[F].reshape(-1, 3), Vb[F[:, [0, 2, 1]]].reshape(-1, 3)]
    nrm = [Nt[F].reshape(-1, 3), Nb[F[:, [0, 2, 1]]].reshape(-1, 3)]
    if rim:
        E = boundary_edges(F)  # directed as in top faces (outward winding)
        a0, b0 = E[:, 0], E[:, 1]
        # quad (top a, top b, bottom b, bottom a); with top winding a->b the rim
        # triangles (a_t, b_b, b_t) and (a_t, a_b, b_b) face away from the sheet.
        q1 = np.stack([Vt[a0], Vr[b0], Vt[b0]], 1)
        q2 = np.stack([Vt[a0], Vr[a0], Vr[b0]], 1)
        tris = np.concatenate([q1, q2])
        fnr = normalize(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))
        # make sure rim normals point away from the sheet interior
        mid = np.concatenate([(Vt[a0] + Vt[b0]) / 2] * 2)
        cen = Vt.mean(0)
        radial = normalize(mid - CENTRE)
        away = mid - cen
        away -= radial * np.einsum("ij,ij->i", away, radial)[:, None]
        flip = np.einsum("ij,ij->i", fnr, away) < 0
        tris[flip] = tris[flip][:, [0, 2, 1]]
        fnr[flip] *= -1
        good = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) > 1e-10
        pos.append(tris[good].reshape(-1, 3))
        nrm.append(np.repeat(fnr[good], 3, 0))
    return np.concatenate(pos), np.concatenate(nrm)


def vertex_normals_safe(V, F, outward_from, sign):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    radial = normalize(V - outward_from) * sign
    length = np.linalg.norm(n, axis=1)
    n = np.where(length[:, None] > 1e-12, n, radial)
    n = normalize(n)
    flip = np.einsum("ij,ij->i", n, radial) < 0
    n[flip] *= -1
    return n


# ------------------------------------------------------------------ fitting
def cover(field: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth surface lying at least CLEAR above ``field`` and its 3x3 neighbourhood.

    The lift is built from smooth bumps only (no max() creases), so piecewise-linear
    meshes of the result stay clear of meshes of ``field`` between samples.
    """
    out = ndimage.gaussian_filter(field, sigma)
    floor = ndimage.maximum_filter(field, 3) + CLEAR
    for _ in range(60):
        viol = floor - out
        if viol.max() <= 0:
            break
        out = out + ndimage.gaussian_filter(ndimage.maximum_filter(viol.clip(0), 7), 2.0) * 1.2 + 2e-5
    return out


def side_meshes(records: list[dict], suffix: str) -> dict[str, np.ndarray]:
    """Unindexed positions of the reference meshes, mirrored into the right-side frame."""
    sx = np.array([-1.0, 1.0, 1.0]) if suffix == "l" else np.ones(3)
    names = [f"Temporalis muscle.{suffix}"] + [b.format(s=suffix) for b in BONES]
    return {f"{suffix}:{n}": decode(find(records, native=n))[0] * sx for n in names}


def fit_side(radii: dict[str, np.ndarray], suffix: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the five layers for one side. Returns {id: (pos, nrm)} in display coords."""
    mirror = suffix == "l"
    frame = Frame()
    rm = radii[f"{suffix}:Temporalis muscle.{suffix}"]
    rb = np.full(frame.shape, np.nan)
    for b in BONES:
        rb = np.fmax(rb, radii[f"{suffix}:" + b.format(s=suffix)])
    rb = ndimage.median_filter(fill_nan(rb), 3)  # close pinholes in the atlas bone

    deg = np.degrees
    TH, PHd = deg(frame.T), deg(frame.P)
    has_m = np.isfinite(rm)
    lateral_m = has_m & (rm > rb - 0.004)                 # muscle is the outer surface
    arch = has_m & ~lateral_m & (TH < -15)                 # bone lateral to muscle, low
    # arch top line: per azimuth, highest arch elevation; nothing below it counts
    top = np.full(frame.shape[0], -np.inf)
    cols = arch.any(1)
    top[cols] = np.array([TH[i][arch[i]].max() for i in np.where(cols)[0]])
    below = TH <= top[:, None]
    region = lateral_m & ~below
    lab, n = ndimage.label(region)
    sizes = ndimage.sum(region, lab, range(1, n + 1))
    f0 = lab == (1 + int(np.argmax(sizes)))
    f0 = ndimage.binary_fill_holes(ndimage.binary_opening(f0, iterations=2))

    cell = np.radians(FINE) * np.nanmedian(rm[f0])
    # extend the footprint onto bone at the upper / posterior margins, not across the arch
    grow = ndimage.distance_transform_edt(~f0) * cell
    d_bony = ndimage.distance_transform_edt(~(arch | below)) * cell
    # stop SETBACK short of bone that lies lateral to the muscle (arch, zygoma) so the
    # sheet edge meets the arch without passing into it
    foot_ind = ((f0 & (d_bony > SETBACK)) | ((grow < MARGIN) & (d_bony > 4 * MM))).astype(float)
    foot_sdf = ndimage.gaussian_filter(foot_ind, 2.0) - 0.5
    foot = foot_sdf > 0

    # base surface: smooth and never below the muscle/bone
    # (only the footprint itself: the arch and zygoma lie outside it and must not lift
    # the sheet edge into a wall; values are extended outward from the footprint)
    rr = fill_nan(np.where(foot | f0, np.fmax(np.where(f0, rm, np.nan), rb), np.nan))
    base = cover(rr, sigma=2.5)

    # geodesic-ish distances on the base surface
    Pbase = CENTRE + base[..., None] * frame.D
    arch_line = foot & ndimage.binary_dilation(arch | below, iterations=2)
    edge = foot & ndimage.binary_dilation(~foot)
    e_edge = cKDTree(Pbase[edge]).query(Pbase.reshape(-1, 3))[0].reshape(frame.shape)
    e_edge = np.where(foot, e_edge, -e_edge)

    # Height above the upper border of the zygomatic arch, measured along the
    # elevation direction. The border is fitted (quadratic in azimuth) to the arch
    # columns only, excluding the frontal process of the zygoma further forward.
    counts = arch.sum(1)
    true_arch = (counts >= 3) & np.isfinite(top) & (top < -28)
    ph_cols = PHd[:, 0]
    coef = np.polyfit(ph_cols[true_arch], top[true_arch], 2)
    lo, hi = ph_cols[true_arch].min(), ph_cols[true_arch].max()
    ph_c = np.clip(ph_cols, lo, hi)
    top_fit = np.polyval(coef, ph_c) + np.polyval(np.polyder(coef), ph_c) * (ph_cols - ph_c)
    v_arch = np.radians(TH - top_fit[:, None]) * base
    # In front of the arch the fossa is bounded by the frontal process of the zygoma;
    # there the height is the surface distance to that bony border instead.
    d_border = cKDTree(Pbase[arch_line]).query(Pbase.reshape(-1, 3))[0].reshape(frame.shape)
    a_, b_, k_ = np.maximum(v_arch, 0), d_border, 0.25  # smooth minimum (no crease in the outline)
    v_arch = np.clip(np.minimum(a_, b_) - k_ * np.log1p(np.exp(-np.abs(a_ - b_) / k_)), 0, None)
    # arc length along the arch (azimuth) on the base surface
    s_arc = np.radians(PHd) * base * np.cos(np.radians(TH))
    r_arch = float(np.median(base[arch_line]))
    c_arch = float(np.cos(np.radians(np.median(top_fit[true_arch]))))

    def pad(tmax, phi0, half_deg, height, edge_ramp, power=2.0):
        """Lens over an ellipse in (arc length along the arch, height above the arch):
        full thickness at the arch, tapering to zero at the ellipse and footprint edges."""
        s0 = np.radians(phi0) * r_arch * c_arch
        a = np.radians(half_deg) * r_arch * c_arch
        q2 = ((s_arc - s0) / a) ** 2 + (v_arch / height) ** 2
        t = tmax * np.clip(1 - q2, 0, 1) ** power * smoothstep(0.5 * MM, edge_ramp, e_edge)
        return ndimage.gaussian_filter(t, 1.0) * (e_edge > 0)

    # deep temporal fat pad: anterior lower fossa (continues beneath the arch in life)
    t_sub = pad(SUBFAT_MAX, 42, 26, 22 * MM, 9 * MM)
    # superficial temporal fat pad: along the arch up to the fascial split
    t_int = pad(INTERFAT_MAX, 32, 36, SPLIT, 8 * MM)
    # suprafascial loose areolar fat: anteroinferior temporal region
    t_sup = pad(SUPRAFAT_MAX, 40, 40, 42 * MM, 10 * MM, power=1.5)

    def lens_sdf(t):
        return t - 0.02 * MM  # zero contour just inside where the pad vanishes

    r_sub0, r_sub1 = base, base + t_sub
    r_dl0 = cover(r_sub1)
    r_dl1 = r_dl0 + LAMINA
    r_if0 = cover(r_dl1)
    r_if1 = r_if0 + t_int
    r_sl0 = cover(r_if1)
    r_sl1 = r_sl0 + LAMINA
    r_sf0 = cover(r_sl1)
    r_sf1 = r_sf0 + t_sup

    out = {}

    def build(key, sdf, r_bot, r_top, rim, r_rim=None):
        FI, FJ, use = snapped_grid(frame, sdf)
        p, n = slab(frame, FI, FJ, use, r_bot, r_top, rim, r_rim)
        if mirror:
            p, n = mirror_x(p, n)  # canonical right frame -> left side
        out[key] = (p, n)

    build("ILL-subfat", lens_sdf(t_sub), r_sub0, r_sub1, rim=False)
    # Along the arch and zygoma the deep lamina's edge turns down onto the bone (its
    # attachment to the medial lip), closing the gap between the sheet edge and the arch.
    w_attach = 1 - smoothstep(SETBACK + 1 * MM, SETBACK + 4 * MM, d_bony)
    build("ILL-deepfascia", foot_sdf, r_dl0, r_dl1, rim=True, r_rim=r_dl0 - w_attach * (r_dl0 - base))
    build("ILL-interfat", lens_sdf(t_int), r_if0, r_if1, rim=False)
    build("ILL-superfascia", foot_sdf, r_sl0, r_sl1, rim=True)
    build("ILL-suprafat", lens_sdf(t_sup), r_sf0, r_sf1, rim=False)
    stats = {
        "footprint_cells": int(foot.sum()), "arch_border_deg": [float(lo), float(hi)],
        "max_mm": {"subfat": float(t_sub.max() / MM), "interfat": float(t_int.max() / MM),
                   "suprafat": float(t_sup.max() / MM)},
    }
    out["_stats"] = stats
    return out


SOURCE = ("Original anatomical illustration; layered offset fit over native Z-Anatomy temporalis and "
          "skull surfaces (tools/anatomy_build/temporal_layers.py); CC BY-SA 4.0")
DESCRIPTIONS = {
    "ILL-subfat": ("Deep temporal (subfascial) fat pad, drawn as a lens between temporalis and the deep "
                   "fascial lamina in the lower anterior temporal fossa, thickest just above the zygomatic "
                   "arch and tapering to zero upward and posteriorly (illustrative maximum about {subfat:.1f} mm)."),
    "ILL-deepfascia": ("Deep lamina of the temporal fascia, drawn as a 0.6 mm sheet over temporalis from the "
                       "superior temporal line to the upper border of the zygomatic arch. Above the fascial "
                       "split it lies directly beneath the superficial lamina."),
    "ILL-interfat": ("Superficial temporal (interfascial) fat pad, drawn as a lens between the deep and "
                     "superficial laminae from the zygomatic arch up to the fascial split about 2.5 cm "
                     "above it (illustrative maximum about {interfat:.1f} mm)."),
    "ILL-superfascia": ("Superficial lamina of the temporal fascia, drawn as a 0.6 mm sheet that lies on the "
                        "deep lamina above the fascial split and passes over the interfascial fat pad below "
                        "it to the zygomatic arch."),
    "ILL-suprafat": ("Suprafascial loose areolar fat beneath the temporoparietal fascia, drawn as a thin lens "
                     "over the anteroinferior temporal fascia, the plane that carries the frontal branch of "
                     "the facial nerve (illustrative maximum about {suprafat:.1f} mm); drawn translucent."),
}
# The suprafascial areolar layer is thin and loose; it is drawn translucent so it does not read as the
# (deeper, surgically important) interfascial fat pad in the closed steps.
OPACITY = {"ILL-suprafat": 0.45}
CAVEAT = (" Layer order and attachments follow the surgical literature; thickness (measured along the "
          "build direction), extent and the split height are illustrative, not segmented from source data.")


def apply(records: list[dict], radii: dict[str, np.ndarray] | None = None) -> dict:
    """Replace the ten temporal layer records in place. ``radii`` (from ``cast_all``) may be
    passed in to skip ray casting when iterating."""
    summary: dict = {"replaced": []}
    if radii is None:
        radii = cast_all({**side_meshes(records, "r"), **side_meshes(records, "l")})
    for suffix in ("r", "l"):
        layers = fit_side(radii, suffix)
        summary[f"stats_{suffix}"] = layers.pop("_stats")
        for base_id in IDS:
            rid = base_id if suffix == "r" else base_id + "-l"
            m = find(records, id=rid)
            pos, nrm = layers[base_id]
            encode(m, pos, nrm)
            m["source"] = SOURCE
            m["description"] = DESCRIPTIONS[base_id].format(**summary[f"stats_{suffix}"]["max_mm"]) + CAVEAT + (
                " Fitted independently on the native left atlas anatomy." if suffix == "l" else "")
            if suffix == "l":
                m.pop("mirroredFrom", None)
                m["independentlyFitted"] = True
                m["baseId"] = base_id
                if m["nativeName"].startswith("Right "):
                    m["nativeName"] = "Left " + m["nativeName"][len("Right "):]
            m["illustrative"] = True
            if base_id in OPACITY:
                m["opacity"] = OPACITY[base_id]
            summary["replaced"].append({"id": rid, "triangles": m["count"] // 3})
    return summary


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit(__doc__.split("CLI: ")[1])
    records = load_records(argv[1])
    summary = apply(records)
    save_records(records, argv[2])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv)
