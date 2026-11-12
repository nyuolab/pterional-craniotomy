"""Illustrative superficial veins and dural sinuses for the pterional guide.

Appends original illustrative tube meshes (group ``veins``) fitted
independently to the native right and left atlas anatomy:

* superficial middle cerebral (superficial Sylvian) vein: a main channel and a
  smaller duplicate channel along the lateral (Sylvian) fissure, with frontal
  and temporal opercular tributaries;
* sphenoparietal sinus along the underside of the lesser sphenoid wing;
* cavernous sinus, simplified as a rounded body lateral to the sella, placed
  from parasellar landmarks (native oculomotor and abducens nerves) because
  the atlas carotid's cavernous segment lies lower than typical;
* superficial temporal vein running with, and posterior to, the native
  superficial temporal artery;
* inferior (Labbe) and superior (Trolard) anastomotic veins as variable,
  context-only tributaries.

Every course is derived at build time from the records passed in (cortex,
bone, arteries and cranial nerves), so the step adapts if the brain meshes are replaced by an
earlier build step. Shapes, calibres and branching are illustrative, not a
segmentation of venous anatomy.

usage: python3 tools/anatomy_build/veins.py IN.json OUT.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (catmull_rom, decode, encode, normalize, ray_nearest,  # noqa: E402
                    triangles_near, tube, weld)

SEED = 20261109
SOURCE = "Original anatomical illustration fitted to Z-Anatomy / BodyParts3D atlas surfaces; CC BY-SA 4.0"
COLOR = "#5f6fae"
DEG = np.pi / 180

# Window ellipse used by the renderer (y, z centre and radii); right side x < -1.9.
WINDOW = dict(cy=0.3, cz=1.25, ry=1.55, rz=1.65)


# ---------------------------------------------------------------- utilities
def side_suffix(s: int) -> str:
    return ".r" if s < 0 else ".l"


def side_records(records: list[dict], group: str, s: int) -> list[dict]:
    suf = side_suffix(s)
    return [m for m in records if m["group"] == group and m.get("nativeName", "").lower().endswith(suf)]


def by_native(records: list[dict], stem: str, s: int) -> dict:
    """Case-insensitive lookup of '<stem>.r' / '<stem>.l'."""
    want = (stem + side_suffix(s)).lower()
    for m in records:
        if m.get("nativeName", "").lower() == want:
            return m
    raise KeyError(want)


def sample_surface(pos: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Vertices plus area-weighted random points on unindexed triangles."""
    tri = pos.reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = rng.poisson(area / spacing ** 2)
    idx = np.repeat(np.arange(len(tri)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return np.concatenate([pos, pts])


def resample(P: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at (approximately) uniform arc length."""
    P = np.asarray(P, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(2, int(np.ceil(d[-1] / step)) + 1)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, k]) for k in range(3)], 1)


def smooth_polyline(P: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along a uniformly sampled polyline, ends pinned."""
    if sigma <= 0:
        return P
    Q = ndimage.gaussian_filter1d(P, sigma, axis=0, mode="nearest")
    Q[0], Q[-1] = P[0], P[-1]
    return Q


def arclen(P: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def mesh_of(records: list[dict], ids_or_recs) -> tuple[np.ndarray, np.ndarray]:
    Vs, Fs, off = [], [], 0
    for m in ids_or_recs:
        p, _ = decode(m)
        V, F = weld(p)
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


# ---------------------------------------------------------- cortical envelope
class Hull:
    """Sulcus-bridging outer envelope of one hemisphere's cortex.

    The envelope is a radius map over (azimuth, elevation) seen from a point
    inside the hemisphere: the outermost cortical sample per bin, closed with
    a disc (so sulci and small gaps between atlas parcels are bridged, as the
    arachnoid bridges them) and lightly smoothed.
    """

    def __init__(self, records: list[dict], s: int, step_deg: float = 0.5):
        self.s = s
        self.c = np.array([0.45 * s, -0.25, 0.35])
        rng = np.random.default_rng(SEED + (1 if s > 0 else 0))
        recs = [m for m in side_records(records, "brain", s) if "white matter" not in m["name"].lower()]
        pts = np.concatenate([sample_surface(decode(m)[0], 0.02, rng) for m in recs])
        self.az = np.arange(-120, 125.01, step_deg) * DEG
        self.el = np.arange(-75, 80.01, step_deg) * DEG
        a, e, r = self.sph(pts)
        ia = np.clip(np.round((a - self.az[0]) / (step_deg * DEG)).astype(int), 0, len(self.az) - 1)
        ie = np.clip(np.round((e - self.el[0]) / (step_deg * DEG)).astype(int), 0, len(self.el) - 1)
        raw = np.zeros((len(self.az), len(self.el)))
        np.maximum.at(raw, (ia, ie), r)
        k = int(round(3.0 / step_deg))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disc = xx ** 2 + yy ** 2 <= k * k
        closed = ndimage.grey_closing(raw, footprint=disc, mode="nearest")
        sm = ndimage.gaussian_filter(closed, 1.5)
        H = np.maximum(sm, raw)
        H = ndimage.gaussian_filter(H, 0.6)
        H = np.maximum(H, raw)
        self.raw, self.H = raw, H
        self._interp = RegularGridInterpolator((self.az, self.el), H, bounds_error=False, fill_value=None)
        self.pts, self.pts_r = pts, r
        self.pts_hull = self.radius(pts)

    def sph(self, p: np.ndarray):
        d = np.atleast_2d(p) - self.c
        r = np.linalg.norm(d, axis=1)
        a = np.arctan2(d[:, 2], self.s * d[:, 0])
        e = np.arcsin(np.clip(d[:, 1] / np.maximum(r, 1e-12), -1, 1))
        return a, e, r

    def direction(self, a, e):
        a, e = np.asarray(a), np.asarray(e)
        return np.stack([self.s * np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)], -1)

    def radius(self, p: np.ndarray) -> np.ndarray:
        a, e, _ = self.sph(p)
        return self._interp(np.stack([a, e], -1))

    def project(self, p: np.ndarray, offset: float | np.ndarray = 0.0) -> np.ndarray:
        """Move points radially onto the envelope (plus ``offset`` outward)."""
        a, e, _ = self.sph(p)
        r = self._interp(np.stack([a, e], -1)) + offset
        return self.c + r[:, None] * self.direction(a, e)

    def outward(self, p: np.ndarray) -> np.ndarray:
        """Envelope normal estimated by finite differences."""
        a, e, _ = self.sph(p)
        h = 0.4 * DEG
        def pt(da, de):
            r = self._interp(np.stack([a + da, e + de], -1))
            return self.c + r[:, None] * self.direction(a + da, e + de)
        ta = pt(h, 0) - pt(-h, 0)
        te = pt(0, h) - pt(0, -h)
        n = normalize(np.cross(ta, te))
        radial = self.direction(a, e)
        n *= np.sign(np.einsum("ij,ij->i", n, radial))[:, None]
        return n


# ------------------------------------------------------------ fissure course
FRONTOPARIETAL = ["Opercular part of inferior frontal gyrus", "Triangular part of inferior frontal gyrus",
                  "Orbital part of  inferior frontal gyrus", "Insula (Subcentral gyrus and ant. and post. sulci*)",
                  "Precentral gyrus", "Postcentral gyrus", "Supramarginal gyrus"]
TEMPORAL = ["Superior temporal gyrus (Lateral part)"]


def surface_samples(records, names, s, hull, rng, tol=0.05):
    out = []
    for nm in names:
        try:
            m = by_native(records, nm, s)
        except KeyError:
            continue
        p = sample_surface(decode(m)[0], 0.015, rng)
        keep = hull.radius(p) - np.linalg.norm(p - hull.c, axis=1) < tol
        out.append(p[keep])
    return np.concatenate(out) if out else np.zeros((0, 3))


def fissure_course(records: list[dict], s: int, hull: Hull) -> np.ndarray:
    """Sylvian fissure line on the lateral envelope, ordered posterior -> anterior.

    For each anteroposterior bin, the fissure lies midway between the upper lip
    of the superior temporal gyrus and the lower lip of the frontoparietal
    opercula (both taken from cortex lying on the envelope).
    """
    rng = np.random.default_rng(SEED + 7 + (1 if s > 0 else 0))
    T = surface_samples(records, TEMPORAL, s, hull, rng)
    O = surface_samples(records, FRONTOPARIETAL, s, hull, rng)
    tO = cKDTree(O)
    # temporal samples that face the opercula across a narrow gap: the inferior lip
    d, j = tO.query(T, distance_upper_bound=0.35)
    ok = np.isfinite(d) & (O[np.minimum(j, len(O) - 1), 1] > T[:, 1])
    mid = 0.5 * (T[ok] + O[j[ok]])
    # order along the principal axis of the gap midpoints in the y-z plane
    q = mid[:, 1:] - mid[:, 1:].mean(0)
    w, U = np.linalg.eigh(q.T @ q)
    ax = U[:, -1] if U[1, -1] > 0 else -U[:, -1]          # +z (anterior)
    t = q @ ax
    edges = np.arange(t.min(), t.max() + 0.1, 0.1)
    pts = []
    for lo in edges[:-1]:
        sel = (t >= lo) & (t < lo + 0.1)
        if sel.sum() >= 4:
            pts.append(np.median(mid[sel], 0))
    P = np.array(pts)
    P = resample(P, 0.05)
    P = smooth_polyline(P, 4)
    P = hull.project(P)
    return P


# -------------------------------------------------------- native vessel axes
def tube_components(m: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Centreline (ordered points) and radii of each connected tube in a record.

    Atlas arteries are smooth-shaded tubes: the vertex with the most nearly
    opposite normal lies across the lumen, so the midpoint of the pair is a
    centreline point and half their distance is the radius.
    """
    p, n = decode(m)
    V, F = weld(p, 1e-4)
    N = np.zeros_like(V)
    key = np.round(p / 1e-4).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    np.add.at(N, inv.reshape(-1), n)
    V = p[first]
    N = normalize(N)
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V)))
    ncomp, lab = connected_components(A, directed=False)
    out = []
    for c in range(ncomp):
        idx = np.flatnonzero(lab == c)
        if len(idx) < 30:
            continue
        q, nq = V[idx], N[idx]
        k = min(60, len(q))
        _, nb = cKDTree(q).query(q, k=k)
        dots = np.einsum("ik,ijk->ij", nq, nq[nb])
        j = dots.argmin(1)
        good = dots[np.arange(len(q)), j] < -0.9
        opp = q[nb[np.arange(len(q)), j]]
        r = np.linalg.norm(q - opp, axis=1) / 2
        cen, r = (0.5 * (q + opp))[good], r[good]   # midpoint: independent of normal sign
        if len(cen) < 10:
            continue
        # collapse each ring's centres, then take the longest path through the
        # Euclidean minimum spanning tree of the collapsed centres
        vk = np.round(cen / 0.04).astype(np.int64)
        _, vinv = np.unique(vk, axis=0, return_inverse=True)
        vinv = vinv.reshape(-1)
        nv = vinv.max() + 1
        cnt = np.bincount(vinv, minlength=nv)
        node = np.zeros((nv, 3))
        np.add.at(node, vinv, cen)
        node /= cnt[:, None]
        if nv < 4:
            continue
        D = np.linalg.norm(node[:, None] - node[None], axis=2) + 1e-9
        np.fill_diagonal(D, 0)
        T = minimum_spanning_tree(D)
        D0 = shortest_path(T, directed=False, indices=0)
        a = int(np.argmax(D0))
        Da, pred = shortest_path(T, directed=False, indices=a, return_predecessors=True)
        b = int(np.argmax(Da))
        path = [b]
        while path[-1] != a:
            path.append(pred[path[-1]])
        P = node[path]
        # principal-curve refinement: parametrise every centre by its nearest
        # path point, then average centres in arc-length bins
        for _ in range(3):
            P = smooth_polyline(resample(P, 0.02), 3)
            L = arclen(P)
            _, near = cKDTree(P).query(cen)
            t = L[near]
            bins = np.arange(0, L[-1] + 0.05, 0.05)
            k = np.digitize(t, bins)
            ks = np.unique(k)
            P = np.array([cen[k == i].mean(0) for i in ks])
            rad = np.array([np.median(r[k == i]) for i in ks])
            if len(P) < 3:
                break
        if len(P) < 3:
            continue
        P = smooth_polyline(resample(P, 0.03), 2)
        rr = np.interp(arclen(P) / arclen(P)[-1], np.linspace(0, 1, len(rad)), ndimage.median_filter(rad, 5))
        out.append((P, rr))
    return out


# --------------------------------------------------------------- skull base
def bone_mesh(records, s, lo, hi):
    names = {"frontal bone", "sphenoid bone", "temporal bone" + side_suffix(s), "parietal bone" + side_suffix(s)}
    recs = [m for m in records if m["group"] == "bone" and m.get("nativeName", "").lower() in names]
    V, F = mesh_of(records, recs)
    return V, triangles_near(V, F, lo, hi, 0.1)


def sphenoid_ridge(records, s):
    """Free edge of the lesser sphenoid wing: (x, ridge height y, ridge z) per x."""
    lo = [min(2.8 * s, 0.5 * s), -2.8, 0.3]
    hi = [max(2.8 * s, 0.5 * s), 0.2, 2.6]
    V, F = bone_mesh(records, s, lo, hi)
    xs = s * np.arange(2.55, 0.95, -0.05)
    zs = np.arange(0.6, 2.4, 0.02)
    O = np.array([[x, 0.0, z] for x in xs for z in zs])
    D = np.tile([0.0, -1.0, 0.0], (len(O), 1))
    t, _ = ray_nearest(O, D, V, F, chunk=32)
    h = -t.reshape(len(xs), len(zs))
    rows = []
    for i, x in enumerate(xs):
        hh = h[i]
        with np.errstate(invalid="ignore"):
            jump = hh[2:] - hh[:-2]
        jump[~np.isfinite(jump)] = 0
        k = int(np.argmax(jump))
        if jump[k] < 0.3:
            continue
        zr = zs[k + 1]
        hr = np.nanmax(hh[k + 2:k + 6][np.isfinite(hh[k + 2:k + 6])]) if np.isfinite(hh[k + 2:k + 6]).any() else hh[k + 2]
        rows.append([x, hr, zr])
    return np.array(rows), (V, F)


def push_clear(P, r, clouds, iters=40, margin=0.01, keep_ends=False):
    """Nudge tube centres away from sampled surfaces until radius + margin clear."""
    P = P.copy()
    trees = [cKDTree(c) for c in clouds]
    for _ in range(iters):
        moved = False
        for tr, c in zip(trees, clouds):
            d, j = tr.query(P)
            need = r + margin - d
            bad = need > 0
            if keep_ends:
                bad[[0, -1]] = False
            if bad.any():
                moved = True
                P[bad] += normalize(P[bad] - c[j[bad]]) * need[bad, None] * 0.7
        if not moved:
            break
        P = smooth_polyline(P, 1)
    return P


# ------------------------------------------------------------- vessel paths
def taper(n, r0, r1, power=1.0):
    u = np.linspace(0, 1, n)
    return r0 + (r1 - r0) * u ** power


def tangent_frame(hull, P):
    """Outward envelope normal and in-surface perpendicular at each path point."""
    T = normalize(np.gradient(P, axis=0))
    N = hull.outward(P)
    B = normalize(np.cross(N, T))
    return T, N, B


def on_surface(hull, P, r, sink=0.5):
    """Place tube centres so a vessel of radius r lies on the envelope (sunk by sink*r)."""
    return hull.project(P, -sink * np.asarray(r))


def surface_path(hull, ctrl, step=0.03, smooth=2):
    """Smooth path through control points, kept on the envelope."""
    P = catmull_rom(hull.project(np.asarray(ctrl, float)), 10)
    P = resample(P, step)
    P = smooth_polyline(hull.project(P), smooth)
    return hull.project(P)


def tube_mesh(P, r, sides):
    return tube(P, r, sides=sides, caps=True)


def ellipse_tube(P, a, b, up, sides=20):
    """Closed sweep of ellipses (semi-axes a sideways, b along ``up``) with pointed ends."""
    P = np.asarray(P, float)
    n = len(P)
    a = np.broadcast_to(np.asarray(a, float), (n,))
    b = np.broadcast_to(np.asarray(b, float), (n,))
    T = normalize(np.gradient(P, axis=0))
    U = normalize(up[None] - T * (T @ up)[:, None])
    L = np.cross(T, U)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    c, si = np.cos(th)[None, :, None], np.sin(th)[None, :, None]
    ring = P[:, None] + a[:, None, None] * c * L[:, None] + b[:, None, None] * si * U[:, None]
    rn = normalize(c * L[:, None] / a[:, None, None] + si * U[:, None] / b[:, None, None])
    V = np.concatenate([ring.reshape(-1, 3), P[[0, -1]]])
    Nv = np.concatenate([rn.reshape(-1, 3), -T[:1], T[-1:]])
    faces = []
    for i in range(n - 1):
        for j in range(sides):
            p0, p1 = i * sides + j, i * sides + (j + 1) % sides
            faces += [[p0, p1, p0 + sides], [p1, p1 + sides, p0 + sides]]
    e0, e1 = n * sides, n * sides + 1
    for j in range(sides):
        faces += [[e0, (j + 1) % sides, j], [e1, (n - 1) * sides + j, (n - 1) * sides + (j + 1) % sides]]
    F = np.array(faces)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum("ij,ij->i", fn, Nv[F].sum(1)) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V[F.reshape(-1)], Nv[F.reshape(-1)]


def merge(parts):
    return np.concatenate([p for p, _ in parts]), np.concatenate([n for _, n in parts])


# ----------------------------------------------------------------- records
def vein_record(slug, s, title, desc, pos, nrm, *, context=False, variant=False, extra=None):
    right = s < 0
    side = "Right" if right else "Left"
    rid = f"ILL-vein-{slug}"
    m = {
        "id": rid if right else rid + "-l",
        "name": f"{side} {title} — illustration",
        "nativeName": f"{side} {title} — illustration" + (".r" if right else ".l"),
        "procedureSide": "right" if right else "left",
        "group": "veins",
        "color": COLOR,
        "source": SOURCE,
        "description": desc,
        "illustrative": True,
    }
    if not right:
        m["baseId"] = rid
        m["independentlyFitted"] = True
    if context:
        m["contextOnly"] = True
    if variant:
        m["variant"] = True
    if extra:
        m.update(extra)
    encode(m, pos, nrm)
    return m


# ------------------------------------------------------------------ builders
def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def offset_course(hull, F, dist, side_dir):
    """Shift a surface course sideways on the envelope by dist(u) (u = 0..1 along F)."""
    T, N, B = tangent_frame(hull, F)
    B *= np.sign(B @ side_dir)[:, None]
    u = arclen(F) / arclen(F)[-1]
    d = dist(u) if callable(dist) else np.full(len(F), float(dist))
    return hull.project(F + B * d[:, None])


def at_fraction(P, f):
    L = arclen(P)
    return int(np.argmin(np.abs(L - f * L[-1])))


def tributary(hull, trunk_surf, trunk_c, trunk_r, j, side_dir, length, back, rng,
              r_tip=0.016, r_join=0.032, step=0.025):
    """Tributary on the envelope that ends inside the trunk lumen at index j."""
    T, N, B = tangent_frame(hull, trunk_surf)
    t, b = T[j], B[j] * np.sign(B[j] @ side_dir)
    p0 = trunk_surf[j]
    wig = rng.normal(0, 0.06, 2)
    ctrl = [p0 + b * length - t * back,
            p0 + b * length * 0.64 - t * back * 0.6 + t * wig[0],
            p0 + b * length * 0.3 - t * back * 0.2 + t * wig[1],
            p0 + b * 0.03]
    P = surface_path(hull, ctrl, step, 1)
    r = taper(len(P), r_tip, r_join, 0.8)
    C = on_surface(hull, P, r)
    return np.concatenate([C, trunk_c[j][None]]), np.r_[r, r_join]


def long_vein(hull, start_surf, end_point, n_ctrl, rng, jitter=0.05):
    """Gently meandering course on the envelope from a trunk point to a distant target."""
    a0, e0, _ = hull.sph(start_surf[None])
    a1, e1, _ = hull.sph(np.asarray(end_point)[None])
    u = np.linspace(0, 1, n_ctrl)
    a = a0 + (a1 - a0) * u
    e = e0 + (e1 - e0) * u
    a[1:-1] += rng.normal(0, jitter, n_ctrl - 2)
    e[1:-1] += rng.normal(0, jitter * 0.6, n_ctrl - 2)
    ctrl = hull.c + 3.0 * hull.direction(a, e)
    return surface_path(hull, ctrl, 0.03, 2)


def nerve_at(P: np.ndarray, z: float, s: int) -> np.ndarray:
    """Mean centreline point of a cranial-nerve tube near coronal plane z, parasellar part only."""
    sel = P[(np.abs(P[:, 2] - z) < 0.12) & (np.abs(P[:, 0]) < 1.3) & (P[:, 1] > -2.6)]
    return sel.mean(0)


def cavernous_landmarks(records, s):
    """Cavernous sinus axis and half-height from parasellar landmarks.

    The atlas internal carotid runs its horizontal (cavernous) segment about
    1.5 cm below the sella floor and rises to clinoid level only in a long
    vertical genu, so the sinus is not fitted to the artery. It is placed from
    the native oculomotor (III) and abducens (VI) nerves instead: from the
    petrous apex (z 0.22) to the superior orbital fissure (z 1.78), centred
    just lateral to the two nerves, with its roof at the oculomotor nerve
    (which enters the roof) and its floor just below the abducens nerve (which
    runs inside the sinus). That puts it lateral to the sella and sphenoid
    body, from about the sella floor up to just under the anterior clinoid.
    """
    def course(stem):
        m = by_native(records, stem, s)
        return max(tube_components(m), key=lambda c: arclen(c[0])[-1])[0]
    c3, c6 = course("Oculomotor nerve (III)"), course("Abducens nerve (VI)")
    zs = np.linspace(0.22, 1.78, 40)
    X3 = np.array([nerve_at(c3, z, s) for z in zs])
    X6 = np.array([nerve_at(c6, z, s) for z in zs])
    xc = 0.5 * (X3[:, 0] + X6[:, 0]) + s * 0.07
    roof = np.minimum(X3[:, 1] - 0.03, -1.56)
    floor = X6[:, 1] - 0.1
    half = np.minimum(0.5 * (roof - floor), 0.27)
    ax = np.c_[xc, roof - half, zs]
    ax = smooth_polyline(resample(ax, 0.025), 3)
    u = arclen(ax) / arclen(ax)[-1]
    return ax, np.interp(u, np.linspace(0, 1, len(zs)), half)


def join_branch(trunk, branch, rb, gap=0.22):
    """Start a branch inside the trunk's end and blend it smoothly onto its own course."""
    k = int(np.argmax(np.linalg.norm(branch - trunk[-1], axis=1) >= gap))
    P = np.concatenate([trunk[-4:], branch[k:]])
    L = arclen(P)
    r = np.interp(L, L[4:], rb[k:])
    Q = resample(P, 0.03)
    rq = np.interp(arclen(Q), L, r)
    Q = smooth_polyline(Q, 3)
    j = int(np.argmin(np.linalg.norm(Q - trunk[-1], axis=1)))
    return Q[j:], rq[j:]


def build_side(records, s, rng):
    out = []
    hull = Hull(records, s)
    F = fissure_course(records, s, hull)
    ridge, (bV, bF) = sphenoid_ridge(records, s)
    brng = np.random.default_rng(SEED + 3)
    bone_cloud = sample_surface(bV[bF].reshape(-1, 3), 0.012, brng)
    temporal = np.array([0.0, -1.0, 0.0])
    frontal = -temporal

    # ---------------- sphenoparietal sinus: under the free edge of the lesser wing
    rs = 0.055
    rows = ridge[(np.abs(ridge[:, 0]) <= 2.45) & (np.abs(ridge[:, 0]) >= 1.2)]
    sp = np.c_[rows[:, 0], rows[:, 1] - 0.05 - rs, rows[:, 2] - 0.02 - rs]
    cav_ax, cav_half = cavernous_landmarks(records, s)
    # the sphenoparietal sinus enters the upper lateral part of the anterior cavernous sinus
    ka = at_fraction(cav_ax, 0.82)
    cav_front = cav_ax[ka] + np.array([s * 0.1, 0.5 * cav_half[ka], 0.0])
    sp = np.concatenate([sp, [sp[-1] + (cav_front - sp[-1]) * 0.45 + np.array([0, 0.04, 0.05])], [cav_front]])
    sp = resample(catmull_rom(sp, 6), 0.03)
    sp = smooth_polyline(sp, 2)
    sp = push_clear(sp, rs, [bone_cloud], margin=0.006)
    r_sp = np.full(len(sp), rs)
    r_sp[-8:] = np.linspace(rs, 0.07, 8)
    out.append(("sphenoparietal", "sphenoparietal sinus",
                "Illustrative sphenoparietal sinus following the underside of the free edge of the native lesser "
                "sphenoid wing, from the lateral sphenoid ridge (where the superficial middle cerebral vein "
                "drains) medially to the anterior cavernous sinus. The course is derived from the atlas skull "
                "base; calibre and exact relation to the dural folds are schematic. Venous drainage of this "
                "region varies (some superficial middle cerebral veins bypass the sinus).",
                tube_mesh(sp, r_sp, 12), {}))

    # ---------------- cavernous sinus, placed from parasellar landmarks (see cavernous_landmarks)
    u = np.linspace(0, 1, len(cav_ax))
    prof = np.maximum(np.clip(1 - np.abs(2 * u - 1) ** 3.0, 0, 1) ** 0.5, 0.08)
    a_cav, b_cav = 0.21 * prof, cav_half * prof
    out.append(("cavernous", "cavernous sinus",
                "Simplified rounded illustration of the cavernous sinus lateral to the sella turcica and sphenoid "
                "body, from the petrous apex to the superior orbital fissure. It is placed from skull-base "
                "landmarks, not fitted to the atlas carotid: its roof lies at the native oculomotor nerve, which "
                "enters the roof, and its floor just below the native abducens nerve, which runs through it. "
                "The atlas internal carotid runs its horizontal segment lower than typical (below the sella "
                "floor), so only the upper part of its ascending genu lies inside this sinus; the carotid's "
                "cavernous course is schematic. Trabeculations, the lateral-wall dural layers and the "
                "intercavernous connections are not modelled; drawn semi-transparent so the carotid and nerves "
                "stay visible.",
                ellipse_tube(cav_ax, a_cav, b_cav, np.array([0.0, 1.0, 0.0]), 22), {"opacity": 0.55}))

    # ---------------- superficial middle cerebral vein
    post = np.flatnonzero(F[:, 2] < F[:, 2].min() + 0.9)
    i0 = int(post[np.argmax(F[post, 1])]) if len(post) else 0
    course = resample(F[i0:], 0.03)
    A = smooth_polyline(offset_course(hull, course, 0.025, temporal), 2)
    nA = len(A)
    u = np.linspace(0, 1, nA)
    rA = 0.045 + 0.02 * smoothstep(u / 0.5)           # 1.8 -> 2.6 mm diameter
    A_c = on_surface(hull, A, rA)
    # leave the cortex at the anterior end and descend under the ridge into the sinus
    k = at_fraction(A, 0.93)
    tail = np.concatenate([A_c[k:k + 1], A_c[-1:], [0.5 * (A_c[-1] + sp[0]) + np.array([s * 0.02, -0.03, -0.02])], sp[:2]])
    tail = resample(catmull_rom(tail, 8), 0.03)
    tail = push_clear(tail, 0.06, [bone_cloud], margin=0.005, keep_ends=True)
    trunk_c = np.concatenate([A_c[:k], tail])
    trunk_r = np.r_[rA[:k], np.linspace(rA[k], rs * 0.9, len(tail))]
    parts = [tube_mesh(trunk_c, trunk_r, 12)]
    # duplicate channel on the frontal lip, joining the main channel anteriorly
    jB0, jB1 = at_fraction(A, 0.3), at_fraction(A, 0.86)
    seg = A[jB0:jB1 + 1]
    uu = np.linspace(0, 1, len(seg))
    B = offset_course(hull, seg, lambda q: 0.12 * (1 - smoothstep((q - 0.7) / 0.3)) + 0.025, frontal)
    rB = 0.028 + 0.016 * smoothstep(uu / 0.4)         # 1.1 -> 1.8 mm diameter
    B_c = on_surface(hull, B, rB)
    B_c = np.concatenate([B_c, A_c[jB1 + 3:jB1 + 4]])
    parts.append(tube_mesh(B_c, np.r_[rB, rB[-1]], 10))
    # opercular tributaries: two frontal, two temporal
    for f, side_dir, length, back in ((0.42, frontal, 0.62, 0.22), (0.68, frontal, 0.5, 0.12),
                                      (0.3, temporal, 0.6, 0.25), (0.58, temporal, 0.55, 0.18)):
        j = at_fraction(A, f)
        parts.append(tube_mesh(*tributary(hull, A, A_c, rA, j, side_dir, length, back, rng), 8))
    out.append(("smcv", "superficial middle cerebral (Sylvian) vein",
                "Illustrative superficial middle cerebral (superficial Sylvian) vein: a main channel on the "
                "temporal lip and a smaller duplicate channel on the frontal lip of the lateral fissure, with two "
                "frontal and two temporal opercular tributaries. The fissure course is derived from the native "
                "atlas cortex (gap between the superior temporal gyrus and the frontoparietal opercula); the "
                "trunk leaves the cortex at the anterior fissure and drains under the sphenoid ridge into the "
                "sphenoparietal sinus. Number, size and position of channels and tributaries vary between "
                "patients; this is not a segmentation of real veins.",
                merge(parts),
                # the isolated skull-base view (no cortex) keeps only the trunk's last ~1.2 cm at the ridge
                {"isolateKeep": {"center": np.round(sp[0], 4).tolist(), "radius": 0.6}}))

    # ---------------- anastomotic veins (variable; context only)
    jT = at_fraction(A, 0.18)
    sss = np.array([s * 0.3, 3.4, -0.6])
    Tr = long_vein(hull, A[jT], sss, 6, rng)
    rT = taper(len(Tr), 0.04, 0.055)
    Tr_c = np.concatenate([A_c[jT:jT + 1], on_surface(hull, Tr, rT)[1:]])
    out.append(("trolard", "superior anastomotic vein (Trolard), variant",
                "Illustrative superior anastomotic vein (of Trolard) linking the superficial middle cerebral vein "
                "to the superior sagittal sinus region over the convexity. Its presence, size and course are "
                "highly variable; shown as one typical variant only.",
                tube_mesh(Tr_c, rT, 8), {}))
    jL = at_fraction(A, 0.08)
    trans = np.array([s * 2.7, -1.25, -3.6])
    La = long_vein(hull, A[jL], trans, 6, rng)
    rL = taper(len(La), 0.04, 0.055)
    La_c = np.concatenate([A_c[jL:jL + 1], on_surface(hull, La, rL)[1:]])
    out.append(("labbe", "inferior anastomotic vein (Labbé), variant",
                "Illustrative inferior anastomotic vein (of Labbé) running posteroinferiorly over the lateral "
                "temporal lobe from the superficial middle cerebral vein toward the transverse sinus region. "
                "Its presence, size and course are highly variable; shown as one typical variant only.",
                tube_mesh(La_c, rL, 8), {}))

    # ---------------- superficial temporal vein
    comps = tube_components(by_native(records, "Superficial temporal artery", s))
    comps = [(P if P[0, 1] < P[-1, 1] else P[::-1], r if P[0, 1] < P[-1, 1] else r[::-1]) for P, r in comps]
    trunkA, trunkR = min(comps, key=lambda c: c[0][0, 1] - 0.01 * arclen(c[0])[-1])
    cand = [c for c in comps if c[0] is not trunkA and arclen(c[0])[-1] > 1.5]
    def gap(c):
        return cKDTree(trunkA).query(c[0][0])[0]
    parA, parR = min(cand, key=lambda c: gap(c) - 0.3 * c[0][-1, 1])
    jb = int(cKDTree(trunkA).query(parA[0])[1])
    lo = int(np.flatnonzero(trunkA[:, 1] >= -2.85)[0])
    hc = np.array([0.0, -0.4, 0.2])

    def shadow(P, Ra, rv, pref):
        T = normalize(np.gradient(P, axis=0))
        rad = normalize(P - hc)
        perp = normalize(np.cross(rad, T))
        perp *= np.sign(perp @ pref)[:, None]
        Q = P + perp * (Ra + rv + 0.02)[:, None]
        d0 = np.linalg.norm(P - hc, axis=1)
        return hc + normalize(Q - hc) * d0[:, None]

    post_dir = np.array([0.0, 0.0, -1.0])
    tP = trunkA[lo:jb + 1]
    rv_t = taper(len(tP), 0.05, 0.045)
    V_t = smooth_polyline(shadow(tP, trunkR[lo:jb + 1], rv_t, post_dir), 2)
    pP = parA
    rv_p = taper(len(pP), 0.04, 0.02)
    V_p = smooth_polyline(shadow(pP, parR, rv_p, post_dir), 2)
    V_p, rv_p = join_branch(V_t, V_p, rv_p)
    fP = trunkA[jb:]
    fP = fP[: at_fraction(fP, 0.75) + 1]
    rv_f = taper(len(fP), 0.04, 0.02)
    V_f = smooth_polyline(shadow(fP, trunkR[jb:jb + len(fP)], rv_f, np.array([0.0, -1.0, -0.3])), 2)
    V_f, rv_f = join_branch(V_t, V_f, rv_f)
    out.append(("stv", "superficial temporal vein",
                "Illustrative superficial temporal vein accompanying the native superficial temporal artery "
                "posterior to it in front of the ear, with frontal and parietal tributaries following the arterial "
                "branches at the same depth. The course is derived from the atlas artery; the vein's exact relation "
                "to the artery, its branching and its calibre are schematic and vary between patients.",
                merge([tube_mesh(V_t, rv_t, 10), tube_mesh(V_p, rv_p, 8), tube_mesh(V_f, rv_f, 8)]),
                {"contextOnly": True}))
    info = {"hull": hull, "fissure": F, "smcv_surface": A, "ridge": ridge}
    return out, info


def apply(records: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    summary = {"records": [], "triangles": 0}
    new = []
    for s in (-1, 1):
        parts, info = build_side(records, s, rng)
        for slug, title, desc, (pos, nrm), extra in parts:
            extra = dict(extra)
            context = bool(extra.pop("contextOnly", False))
            variant = slug in ("trolard", "labbe")
            m = vein_record(slug, s, title, desc, pos, nrm, context=context or variant, variant=variant,
                            extra=extra)
            new.append(m)
            summary["records"].append({"id": m["id"], "triangles": m["count"] // 3})
            summary["triangles"] += m["count"] // 3
    ids = {m["id"] for m in new}
    records[:] = [m for m in records if m["id"] not in ids] + new
    return summary


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    from common import load_records, save_records
    import json
    records = load_records(argv[1])
    summary = apply(records)
    save_records(records, argv[2])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv)
