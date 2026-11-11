"""Illustrative translucent arachnoid over the lateral cortex and Sylvian fissure.

Appends one open sheet per side (group ``arachnoid``): the sulcus-bridging
outer envelope of the native atlas cortex (see ``veins.Hull``) offset a
fraction of a millimetre outward, covering the bone-window region with a
margin and the Sylvian fissure as far as the sphenoid ridge. Where the
illustrative superficial veins from ``veins.py`` are already present the sheet
is lifted just over them (they are subarachnoid), and everywhere it is kept
below the inner table of the native skull. It is left out where the outermost
cortex is a context-only parcel (orbital gyri, lateral orbital sulcus, temporal
pole), which the procedure views hide to keep the optico-carotid corridor open.

Each record carries ``fissureLine``: points (display coordinates, on the
sheet) tracing the Sylvian fissure, used by the renderer to open the sheet
along the fissure in the Sylvian-split step.

Run after ``veins.py`` so the sheet drapes over the veins, and after ``dura.py``:
where the sheet would reach the illustrative dura it is pulled back to lie at
least ``DURA_CLEAR`` deep to it, so the translucent sheet never veils the
closed dura.

usage: python3 tools/anatomy_build/arachnoid.py IN.json OUT.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import encode, normalize, orient_outward, ray_nearest, triangles_near, vertex_normals  # noqa: E402
from veins import DEG, WINDOW, Hull, arclen, decode, fissure_course, mesh_of, resample  # noqa: E402

COLOR = "#cfe3ea"
SOURCE = "Original anatomical illustration fitted to Z-Anatomy / BodyParts3D atlas surfaces; CC BY-SA 4.0"
LIFT = 0.012          # sheet height over the cortical envelope (0.24 mm)
SEED = 20261110
VEIN_CLEAR = 0.008    # clearance over the illustrative veins (0.16 mm)
BONE_CLEAR = 0.06     # keep >= 1.2 mm under the inner table (room for the dura)
MIN_LIFT = 0.004      # never closer than 0.08 mm to cortex or vein surfaces
MARGIN = 1.15         # window ellipse scale
DURA_CENTRE = np.array([0.55, 0.15, 1.0])  # dura.py CENTRE (x mirrored per side)
DURA_CLEAR = 0.005    # keep >= 0.1 mm deep to the illustrative dura (dura.py output), if present
STEP_DEG = 0.7


def bone_radius_map(records, s, hull, az, el, lo, hi):
    """Distance from the hull centre to the inner table along each grid direction (coarse, interpolated)."""
    names = {"frontal bone", "sphenoid bone", "temporal bone" + (".r" if s < 0 else ".l"),
             "parietal bone" + (".r" if s < 0 else ".l")}
    recs = [m for m in records if m["group"] == "bone" and m.get("nativeName", "").lower() in names]
    V, F = mesh_of(records, recs)
    F = triangles_near(V, F, lo, hi, 0.4)
    ca, ce = az[::3], el[::3]
    A, E = np.meshgrid(ca, ce, indexing="ij")
    D = hull.direction(A.reshape(-1), E.reshape(-1))
    t, _ = ray_nearest(np.repeat(hull.c[None], len(D), 0), D, V, F, chunk=48)
    t = t.reshape(A.shape)
    t[~np.isfinite(t)] = 99.0
    f = RegularGridInterpolator((ca, ce), t, bounds_error=False, fill_value=None)
    AA, EE = np.meshgrid(az, el, indexing="ij")
    return f(np.stack([AA, EE], -1))


def hidden_cortex_cells(records, s, hull, az, el):
    """Grid cells whose outermost cortex belongs to a context-only (hidden) parcel."""
    from veins import sample_surface, side_records
    rng = np.random.default_rng(SEED + (1 if s > 0 else 0))
    top = {True: np.zeros((len(az), len(el))), False: np.zeros((len(az), len(el)))}
    for m in side_records(records, "brain", s):
        if "white matter" in m["name"].lower():
            continue
        p = sample_surface(decode(m)[0], 0.02, rng)
        a, e, r = hull.sph(p)
        ia = np.round((a - az[0]) / (STEP_DEG * DEG)).astype(int)
        ie = np.round((e - el[0]) / (STEP_DEG * DEG)).astype(int)
        ok = (ia >= 0) & (ia < len(az)) & (ie >= 0) & (ie < len(el))
        np.maximum.at(top[bool(m.get("contextOnly"))], (ia[ok], ie[ok]), r[ok])
    hid = top[True] > top[False] + 0.02
    return ndimage.binary_opening(hid, iterations=1)


def cover_top_map(records, s, hull, az, el):
    """Smooth tent over the illustrative superficial veins (outermost radius, 0 where none).

    The per-cell outermost vein radius is spread with a parabolic profile so the
    sheet drapes over the vessels instead of forming steps. Native arteries are
    not draped (the atlas cortical arteries partly float off the cortex).
    """
    top = np.zeros((len(az), len(el)))
    suf_id = "" if s < 0 else "-l"
    for m in records:
        if m["id"] not in {f"ILL-vein-{k}{suf_id}" for k in ("smcv", "trolard", "labbe")}:
            continue
        p, _ = decode(m)
        a, e, r = hull.sph(p)
        ia = np.round((a - az[0]) / (STEP_DEG * DEG)).astype(int)
        ie = np.round((e - el[0]) / (STEP_DEG * DEG)).astype(int)
        ok = (ia >= 0) & (ia < len(az)) & (ie >= 0) & (ie < len(el))
        np.maximum.at(top, (ia[ok], ie[ok]), r[ok])
    cell = STEP_DEG * DEG * 3.3                     # approx. cell size in display units
    tent = np.where(top > 0, top, -np.inf)
    out = tent.copy()
    k = 4
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            d = np.hypot(di, dj) * cell
            if d == 0 or d > k * cell:
                continue
            sh = np.roll(tent, (di, dj), axis=(0, 1)) - 6.0 * d * d
            out = np.maximum(out, sh)
    return np.where(np.isfinite(out), out, 0.0)


def build_sheet(records, s):
    hull = Hull(records, s)
    F = fissure_course(records, s, hull)
    az = np.arange(-75, 95.01, STEP_DEG) * DEG
    el = np.arange(-55, 60.01, STEP_DEG) * DEG
    AA, EE = np.meshgrid(az, el, indexing="ij")
    grid = np.stack([AA, EE], -1)
    Hm = hull._interp(grid)
    # outermost cortex sample within each sheet cell (peaks finer than the envelope bins)
    a, e, r = hull.sph(hull.pts)
    ia = np.round((a - az[0]) / (STEP_DEG * DEG)).astype(int)
    ie = np.round((e - el[0]) / (STEP_DEG * DEG)).astype(int)
    ok = (ia >= 0) & (ia < len(az)) & (ie >= 0) & (ie < len(el))
    cell = np.zeros(Hm.shape)
    np.maximum.at(cell, (ia[ok], ie[ok]), r[ok])
    # the sheet is linear between grid points, so it must clear the outermost cortex of the
    # neighbouring cells as well (gyral crowns peak between grid points)
    Hdil = np.maximum(ndimage.maximum_filter(cell, size=3, mode="nearest"), Hm)
    D = hull.direction(AA, EE)
    P0 = hull.c + Hm[..., None] * D
    # region as a continuous field (< 0 inside): window ellipse (+ margin) on the
    # lateral surface, united with a band along the fissure toward the sphenoid ridge
    y, z = P0[..., 1], P0[..., 2]
    ell = ((y - WINDOW["cy"]) / (WINDOW["ry"] * MARGIN)) ** 2 + ((z - WINDOW["cz"]) / (WINDOW["rz"] * MARGIN)) ** 2
    from scipy.spatial import cKDTree
    fd = cKDTree(F).query(P0.reshape(-1, 3))[0].reshape(Hm.shape)
    band = np.maximum((fd - 0.3) / 0.3, (ell - 1.6) / 1.6)
    field = np.maximum(np.minimum(ell - 1, band), (1.55 - s * P0[..., 0]) / 0.4)
    inside = field < 0
    lab, n = ndimage.label(inside)
    if n > 1:
        sizes = ndimage.sum(inside, lab, range(1, n + 1))
        inside = lab == (1 + int(np.argmax(sizes)))
    inside = ndimage.binary_fill_holes(inside)
    # Leave out cells whose outermost cortex is a context-only parcel (the orbital gyri, lateral
    # orbital sulcus and temporal pole stay hidden in the procedure views to keep the
    # optico-carotid corridor open): a sheet over hidden cortex would hang over the dark corridor.
    hidden = hidden_cortex_cells(records, s, hull, az, el)
    inside &= ~hidden
    field = np.where(hidden, np.maximum(field, 0.05), field)
    lab, n = ndimage.label(inside)
    if n > 1:
        sizes = ndimage.sum(inside, lab, range(1, n + 1))
        inside = lab == (1 + int(np.argmax(sizes)))
    used = ndimage.binary_dilation(inside)
    # snap the outer ring of grid points onto the zero contour of the field
    pa, pe = AA.copy(), EE.copy()
    acc = np.zeros(AA.shape + (2,))
    cnt = np.zeros(AA.shape)
    ring = used & ~inside
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_in = np.roll(inside, (-di, -dj), axis=(0, 1))
        nb_f = np.roll(field, (-di, -dj), axis=(0, 1))
        sel = ring & nb_in
        t = np.clip(field / np.maximum(field - nb_f, 1e-9), 0, 1)
        na = np.roll(AA, (-di, -dj), axis=(0, 1))
        ne = np.roll(EE, (-di, -dj), axis=(0, 1))
        acc[..., 0] += np.where(sel, AA + t * (na - AA), 0)
        acc[..., 1] += np.where(sel, EE + t * (ne - EE), 0)
        cnt += sel
    ok = ring & (cnt > 0)
    pa[ok] = acc[ok, 0] / cnt[ok]
    pe[ok] = acc[ok, 1] / cnt[ok]
    used = inside | ok
    # height: just over the envelope, tented over vessels on the cortex, under the inner table
    cover = cover_top_map(records, s, hull, az, el)
    rad = Hdil + LIFT
    rad = np.where(cover > 0, np.maximum(rad, cover + VEIN_CLEAR), rad)
    rad = np.maximum(ndimage.gaussian_filter(rad, 1.0), Hdil + LIFT)
    rad = np.where(cover > 0, np.maximum(rad, cover + VEIN_CLEAR), rad)
    pts = P0[used]
    bone = bone_radius_map(records, s, hull, az, el, pts.min(0), pts.max(0))
    # Keep under the inner table, except where the atlas cortex itself reaches the
    # bone (the atlas brain and skull overlap by a few mm in places): there the
    # sheet stays just over the cortex and vessels rather than dipping under them.
    floor = np.maximum(Hdil + MIN_LIFT, np.where(cover > 0, cover + MIN_LIFT, 0.0))
    clamp = rad > bone - BONE_CLEAR
    conflict = floor > bone - BONE_CLEAR
    rad = np.maximum(np.minimum(rad, bone - BONE_CLEAR), floor)
    under = np.maximum(Hdil, cover)
    fr = RegularGridInterpolator((az, el), rad, bounds_error=False, fill_value=None)
    Rs = fr(np.stack([pa, pe], -1))
    P = hull.c + Rs[..., None] * hull.direction(pa, pe)
    # triangulate quads with all corners used and at least one strictly inside
    na_, ne_ = AA.shape
    idx = np.arange(na_ * ne_).reshape(na_, ne_)
    q00, q01, q10, q11 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    u = used
    qok = u[:-1, :-1] & u[:-1, 1:] & u[1:, :-1] & u[1:, 1:] & (
        inside[:-1, :-1] | inside[:-1, 1:] | inside[1:, :-1] | inside[1:, 1:])
    Fc = np.concatenate([np.stack([q00, q10, q01], -1)[qok], np.stack([q01, q10, q11], -1)[qok]])
    keep = np.unique(Fc)
    remap = np.full(na_ * ne_, -1)
    remap[keep] = np.arange(len(keep))
    V = P.reshape(-1, 3)[keep]
    Fc = remap[Fc]
    area = np.linalg.norm(np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]]), axis=1)
    Fc = Fc[area > 1e-7]
    # stay deep to the illustrative dura (run after dura.py) so the closed dura is never veiled
    V, dura_moved = clamp_under_dura(records, s, V)
    Fc = orient_outward(V, Fc, hull.c)
    N = vertex_normals(V, Fc)
    radial = normalize(V - hull.c)
    N *= np.sign(np.einsum("ij,ij->i", N, radial))[:, None]
    pos, nrm = V[Fc.reshape(-1)], N[Fc.reshape(-1)]
    mask = inside
    # fissure polyline on the sheet, limited to the covered part
    fr = RegularGridInterpolator((az, el), rad, bounds_error=False, fill_value=None)
    fm = RegularGridInterpolator((az, el), (field < 0).astype(float), bounds_error=False, fill_value=0.0)
    a, e, _ = hull.sph(F)
    inside = fm(np.stack([a, e], -1)) > 0.99
    idx = np.flatnonzero(inside)
    Fin = F[idx[0]:idx[-1] + 1]
    Fl = resample(Fin, arclen(Fin)[-1] / 9.0)
    a, e, _ = hull.sph(Fl)
    Fl = hull.c + fr(np.stack([a, e], -1))[:, None] * hull.direction(a, e)
    Fl, _ = clamp_under_dura(records, s, Fl)
    stats = {
        "dura_clamped_vertices": dura_moved,
        "triangles": int(len(Fc)),
        "height_over_cortex_or_vessel_mm_median": float(np.median((rad - under)[mask]) * 20),
        "height_over_cortex_or_vessel_mm_p95": float(np.percentile((rad - under)[mask], 95) * 20),
        "height_over_cortex_or_vessel_mm_max": float(np.max((rad - under)[mask]) * 20),
        "height_over_envelope_mm_median": float(np.median((rad - Hm)[mask]) * 20),
        "vertices": int(mask.sum()),
        "bone_lowered_vertices": int((clamp & mask).sum()),
        "cortex_within_1p2mm_of_inner_table_vertices": int((conflict & mask).sum()),
        "below_envelope_vertices": int(((rad < Hm) & mask).sum()),
    }
    return pos, nrm, np.round(Fl, 4).tolist(), stats


def clamp_under_dura(records, s, pts):
    """Pull points that lie outside (or within DURA_CLEAR of) the side's dura sheet back under it,
    along rays from the dura build centre. Returns (points, number moved). No-op without dura."""
    rid = "ILL-dura" if s < 0 else "ILL-dura-l"
    dura = next((m for m in records if m["id"] == rid), None)
    if dura is None or len(pts) == 0:
        return pts, 0
    from common import weld
    Vd, Fd = weld(decode(dura)[0])
    C = DURA_CENTRE * np.array([s, 1.0, 1.0])
    tri_dir = normalize(Vd[Fd].mean(1) - C)
    tri_ang = np.arccos(np.clip(np.einsum("tij,ti->tj", normalize(Vd[Fd] - C), tri_dir), -1, 1)).max(1)
    pts = np.array(pts, dtype=float)
    d = normalize(pts - C)
    r = np.linalg.norm(pts - C, axis=1)
    t = np.full(len(pts), np.inf)
    order = np.lexsort((d[:, 1], np.round(np.arctan2(d[:, 2], -s * d[:, 0]), 1)))
    for k in range(0, len(order), 256):
        sel = order[k:k + 256]
        mean = normalize(d[sel].mean(0))
        spread = np.arccos(np.clip(d[sel] @ mean, -1, 1)).max()
        near = np.arccos(np.clip(tri_dir @ mean, -1, 1)) <= spread + tri_ang + 1e-3
        if near.any():
            t[sel] = ray_nearest(C, d[sel], Vd, Fd[near], chunk=64)[0]
    move = np.isfinite(t) & (r > t - DURA_CLEAR)
    pts[move] = C + d[move] * (t[move] - DURA_CLEAR)[:, None]
    return pts, int(move.sum())


def arachnoid_record(s, pos, nrm, line):
    right = s < 0
    side = "Right" if right else "Left"
    m = {
        "id": "ILL-arachnoid" if right else "ILL-arachnoid-l",
        "name": f"{side} arachnoid over the Sylvian region — illustration",
        "nativeName": f"{side} arachnoid over the Sylvian region — illustration" + (".r" if right else ".l"),
        "procedureSide": "right" if right else "left",
        "group": "arachnoid",
        "color": COLOR,
        "opacity": 0.45,
        "source": SOURCE,
        "description": "Original translucent arachnoid illustration: a thin sheet a fraction of a millimetre over the "
                       "sulcus-bridging envelope of the native atlas cortex, covering the bone-window region and the "
                       "Sylvian fissure toward the sphenoid ridge, draped over the illustrative superficial veins and "
                       "kept beneath the inner table and the illustrative dura. It is left out over the "
                       "context-only orbital gyri and temporal pole, which stay hidden in the procedure views. "
                       "It bridges sulci and does not model trabeculae, cisterns or "
                       "real thickness. fissureLine traces the Sylvian fissure derived from the atlas cortex.",
        "illustrative": True,
        "fissureLine": line,
    }
    if not right:
        m["baseId"] = "ILL-arachnoid"
        m["independentlyFitted"] = True
    encode(m, pos, nrm)
    return m


def apply(records: list[dict]) -> dict:
    summary = {"records": []}
    new = []
    for s in (-1, 1):
        pos, nrm, line, stats = build_sheet(records, s)
        m = arachnoid_record(s, pos, nrm, line)
        new.append(m)
        summary["records"].append({"id": m["id"], **stats, "fissureLine": line})
    ids = {m["id"] for m in new}
    records[:] = [m for m in records if m["id"] not in ids] + new
    return summary


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    import json
    from common import load_records, save_records
    records = load_records(argv[1])
    summary = apply(records)
    save_records(records, argv[2])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv)
