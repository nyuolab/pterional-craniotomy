"""Clean schematic markings and instrument arrows.

Regenerates the geometry of every ``annotations`` record (``MARK-*``, both
sides) and every ``instruments`` record (``TOOL-*``) after the scalp step, so
the coloured teaching lines read as clean, even strokes. Ids, names, groups,
colours, stage fields (``markerStages``, ``markerFissure``, ``instrument*``),
``follow`` and the meaning of the reveal fields are kept; ``markerCenter``,
``markerAxis`` and ``markerRange`` are recomputed from the new geometry so the
renderer's reveal (``uMarkKind``: 0 points, 1 angular sweep about
``markerCenter`` in the y-z plane, 2 sweep along ``markerAxis`` over
``markerRange``) still sweeps the whole line once.

Why. The imported markings were chains of separate 8-sided prisms, one per
polyline segment, each with flat end caps and flat (per-face) normals. Segment
lengths varied from under 1 mm to 12 mm with turning angles up to 180 degrees,
the prism cross-sections were not transported from one segment to the next,
and many lines were drawn on surfaces that the guide has since replaced (the
illustrative temporal laminae, and now the illustrative scalp), so they dipped
under or floated several millimetres above the surface they annotate.

Method.

* Each line has a design curve: the analytic shape the renderer cuts where one
  exists (bone window ellipse, burr-hole circles, dural opening circle and the
  low subfrontal ellipse, all projected along x exactly as the shader's
  cylinders cut), the scalp step's incision curve (``incisionLine`` of
  ``ILL-scalp-flap``), or the imported marking smoothed over arc length.
  Where the scalp changed what a marking must show, the design follows it:
  the prepared-field boundary is grown to enclose the new incision with a
  12 mm margin (re-imposed after smoothing; where the incision meets the
  scalp's free edge at the zygomatic arch the boundary follows that edge
  instead, and the incision marking stops ``INCISION_TRIM`` above it, because
  the illustrative scalp ends there); the flap margin is kept between the
  incision and the scalp flap's hinge line; the flap-reflection arrow runs
  along the flap's reflection direction (``-away``); closure stitches sit
  across the new incision at even spacing, buried galeal stitches on the galea
  (galea closure, ``MARK-galea-sutures``, added here) and skin stitches on the
  skin (skin closure); tack-up and fixation dashes point radially across the
  window margin; the clinoid outline is an ellipse through the imported
  points' principal axes.

  =====================  ====================================================
  surface                markings
  =====================  ====================================================
  scalp skin             field, incision, skin closure, head-rotation arrow
  galea                  galeal closure
  bed under the flap     flap margin, flap-reflection arrow, drain route,
                         temporalis reattachment (outermost of pericranium,
                         temporal fascial layers and fat, temporalis, bone)
  pericranium / bone     periosteal exposure boundary
  bone (stationary       bone-flap outline, burr holes, fixation sites
  pericranium on top)
  bone                   pterion, sphenoid ridge
  dura                   dural opening (both), tack-up positions
  deep, smoothed only    Sylvian trace, clinoid outline, optico-carotid region
  =====================  ====================================================
* The design curve is draped on the surface the marking annotates (scalp skin,
  the exposed bed under the scalp flap, bone, pericranium, dura) by casting
  along a smoothed surface normal to the outermost surface of that set, then
  offset along the smoothed normal so the tube's underside clears the surface
  by ``GAP``. Clearance against the true (unsmoothed) mesh is then enforced by
  pushing the line out where it would touch, so a line never dips into bumpy
  bone; ``GAP`` is therefore the minimum clearance. Lines smoothed over bumpy
  hosts (outlines projected onto bone and dura, dashes) ride an envelope
  above it; each record's description states its measured range and median. Deep markings (Sylvian trace, clinoid, optico-carotid region) keep
  their positions and are only smoothed.
* Lines are resampled uniformly by arc length and swept as tubes of constant
  radius with parallel-transport frames (closed loops distribute the holonomy
  twist so the seam matches), smooth radial normals, round caps on open ends
  and cone arrowheads on direction arrows and instruments.
* Left records are built by the same code on native left anatomy (mirrored
  into the right-side frame for the computation and mirrored back); the left
  incision follows the left scalp's own ``incisionLine``. Instruments keep
  each record's own tip (``anchor``) and far end.

The markings are schematic teaching aids, not measured landmarks or
trajectories. Deterministic (no randomness).

CLI: python3 tools/anatomy_build/markers.py IN.json OUT.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (decode, encode, find, load_records, mirror_x, normalize,  # noqa: E402
                    ray_nearest, save_records, weld)

MM = 0.05                        # display units per millimetre
SHELL = np.array([0.0, -0.4, -0.4])   # shell centre used by the scalp step
R_LINE = 0.035                   # tube radius of lines (0.7 mm, as before)
R_DASH = 0.032                   # short dashes (tacks, plates, stitches)
GAP = 0.1 * MM                   # clearance of the tube underside above its surface
SIDES = 8                        # tube cross-section (smooth normals)
DASH_SIDES, DASH_STEP = 8, 0.045 # short dashes: fewer rings (they are only a few millimetres long)
STEP = 0.07                      # ring spacing along a line (1.4 mm)
CAP_RINGS = 3                    # latitude rings of a round cap
ARROW_HEAD = (0.085, 0.22)       # marker arrowhead: base radius, length
TOOL_SHAFT, TOOL_HEAD = 0.055, (0.12, 0.32)   # instrument shaft radius; head base radius, length
TOOL_SIDES = 12

# Renderer analytic cuts (assets/app.js fragment shader, canonical right side, x < -1.9).
WINDOW = dict(cy=0.3, cz=1.25, ry=1.55, rz=1.65)        # bone window ellipse
DURAL = dict(cy=0.3, cz=1.25, ry=1.08, rz=1.08)         # dural opening circle
DURAL_LOW = dict(cy=-0.62, cz=1.65, ry=0.38, rz=0.55)   # low subfrontal dural opening
BURRS = {"MARK-burr-1": (-0.7, 2.45), "MARK-burr-2": (-1.0, 0.8)}
BURR_RING = 0.32                 # marker ring radius around each burr hole (hole radius 0.255)

BONES = ["Frontal bone", "Parietal bone.r", "Parietal bone.l", "Sphenoid bone", "Temporal bone.r",
         "Temporal bone.l", "Zygomatic bone.r", "Zygomatic bone.l", "Occipital bone"]
TEMPORAL = ["ILL-superfascia", "ILL-suprafat", "ILL-deepfascia", "ILL-interfat", "ILL-subfat"]
SKIN = ["ILL-scalp-flap", "ILL-scalp", "ILL-scalp-context"]
GALEA = ["ILL-galea-flap", "ILL-galea", "ILL-galea-context"]
INCISION_TRIM = 5 * MM          # the incision marking stops this far above the scalp's free edge at the arch
FIELD_EDGE_INSET = 1.2 * MM     # field boundary along the free edge where the incision meets it (tube 0.7 mm)
FIELD_MIN_MARGIN = 8 * MM       # build check: incision marking inside the field by this much (away from the edge)
GALEA_SUTURES = "MARK-galea-sutures"
STAGES = {"MARK-incision": ["prepare", "incision", "skin-close"], "MARK-scalp-sutures": ["skin-close"],
          GALEA_SUTURES: ["galea-close"]}
NAMES = {"MARK-scalp-sutures": "Skin closure — schematic", GALEA_SUTURES: "Galeal closure — schematic"}
PERI_STATIONARY = ["ILL-pericranium", "ILL-pericranium-context"]
PERI = ["ILL-pericranium-flap"] + PERI_STATIONARY

SOURCE = ("Original schematic marking (tools/anatomy_build/markers.py), redrawn on the guide's reference and "
          "illustrative surfaces; anatomy reference from the listed operative papers")
TOOL_SOURCE = "Original qualitative instrument-direction illustration (tools/anatomy_build/markers.py)"


# ============================================================== curve utilities
def arclen(P: np.ndarray, closed: bool = False) -> np.ndarray:
    Q = np.vstack([P, P[:1]]) if closed else P
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]


def resample(P: np.ndarray, step: float, closed: bool = False, n: int | None = None) -> np.ndarray:
    """Uniform arc-length resampling (closed: no repeated end point)."""
    Q = np.vstack([P, P[:1]]) if closed else P
    s = arclen(P, closed)
    L = s[-1]
    if n is None:
        n = max(2, int(np.ceil(L / step)) + (0 if closed else 1))
    t = np.linspace(0, L, n, endpoint=not closed)
    return np.stack([np.interp(t, s, Q[:, k]) for k in range(3)], 1)


def gsmooth(P: np.ndarray, sigma: float, closed: bool = False, keep_ends: bool = True) -> np.ndarray:
    """Gaussian smoothing over arc length of a uniformly sampled curve. Open curves are padded by
    point reflection about their end points, which keeps the end points and end tangents."""
    if sigma <= 0 or len(P) < 3:
        return P.copy()
    h = arclen(P, closed)[-1] / (len(P) if closed else len(P) - 1)
    k = max(1, int(np.ceil(3 * sigma / h)))
    w = np.exp(-0.5 * (np.arange(-k, k + 1) * h / sigma) ** 2)
    w /= w.sum()
    if closed:
        idx = (np.arange(len(P))[:, None] + np.arange(-k, k + 1)[None]) % len(P)
        return (P[idx] * w[None, :, None]).sum(1)
    k = min(k, len(P) - 1)
    w = np.exp(-0.5 * (np.arange(-k, k + 1) * h / sigma) ** 2)
    w /= w.sum()
    if keep_ends:
        ext = np.vstack([2 * P[0] - P[k:0:-1], P, 2 * P[-1] - P[-2:-k - 2:-1]])
    else:
        ext = np.vstack([np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)])
    idx = np.arange(len(P))[:, None] + np.arange(2 * k + 1)[None]
    return (ext[idx] * w[None, :, None]).sum(1)


def smooth1d(v: np.ndarray, sigma_n: float, closed: bool) -> np.ndarray:
    """Gaussian smoothing of a per-sample scalar/vector field (sigma in samples)."""
    if sigma_n <= 0:
        return v.copy()
    k = max(1, int(np.ceil(3 * sigma_n)))
    w = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma_n) ** 2)
    w /= w.sum()
    n = len(v)
    idx = np.arange(n)[:, None] + np.arange(-k, k + 1)[None]
    idx = idx % n if closed else np.clip(idx, 0, n - 1)
    vv = v[idx]
    return (vv * (w[None, :, None] if v.ndim == 2 else w[None])).sum(1)


def maxfilter(v: np.ndarray, k: int, closed: bool) -> np.ndarray:
    idx = np.arange(len(v))[:, None] + np.arange(-k, k + 1)[None]
    idx = idx % len(v) if closed else np.clip(idx, 0, len(v) - 1)
    return v[idx].max(1)


def tangents(P: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        return normalize(np.roll(P, -1, 0) - np.roll(P, 1, 0))
    T = np.gradient(P, axis=0)
    return normalize(T)


def curve_stats(P: np.ndarray, closed: bool) -> dict:
    """Turning angle per mm and spacing statistics of a centreline."""
    Q = np.vstack([P[-1:], P, P[:1]]) if closed else P
    d = np.diff(Q, axis=0)
    L = np.linalg.norm(d, axis=1)
    cos = np.einsum("ij,ij->i", d[:-1], d[1:]) / np.maximum(L[:-1] * L[1:], 1e-12)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    seg = L if not closed else L[1:]
    return {"maxTurnDeg": float(turn.max()) if len(turn) else 0.0,
            "maxTurnDegPerMm": float((turn / np.maximum(0.5 * (L[:-1] + L[1:]) / MM, 1e-9)).max()) if len(turn) else 0.0,
            "spacingMm": [float(seg.min() / MM), float(seg.max() / MM)]}


# ===================================================================== hosts
class Host:
    """Union of record surfaces, restricted to faces that face a given direction field."""

    def __init__(self, records: list[dict], facing: str | np.ndarray = "radial", min_dot: float = 0.15,
                 mirror: bool = False):
        T, N = [], []
        for m in records:
            p, n = decode(m)
            if mirror:
                p, n = mirror_x(p, n)
            T.append(p.reshape(-1, 3, 3))
            N.append(n.reshape(-1, 3, 3).sum(1))
        T = np.concatenate(T)
        N = normalize(np.concatenate(N))
        cen = T.mean(1)
        ref = normalize(cen - SHELL) if isinstance(facing, str) else np.broadcast_to(normalize(np.asarray(facing, float)), cen.shape)
        keep = np.einsum("ij,ij->i", N, ref) > min_dot
        area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        keep &= area > 1e-12
        self.T, self.N, self.cen, self.area = T[keep], N[keep], cen[keep], area[keep]
        self.rad = np.linalg.norm(self.T - self.cen[:, None], axis=2).max(1)   # centroid-to-corner bound
        self.tree = cKDTree(self.cen)
        self.V, self.F = weld(self.T.reshape(-1, 3), 1e-7)

    def __len__(self) -> int:
        return len(self.T)

    def cast(self, orig: np.ndarray, dirs: np.ndarray, tmax: float) -> tuple[np.ndarray, np.ndarray]:
        """First hit along dirs from orig (inf / -1 when missed)."""
        ends = orig + dirs * tmax
        lo = np.minimum(orig.min(0), ends.min(0)) - 1e-3
        hi = np.maximum(orig.max(0), ends.max(0)) + 1e-3
        tri = self.V[self.F]
        sel = np.where((tri.max(1) >= lo).all(1) & (tri.min(1) <= hi).all(1))[0]
        t = np.full(len(orig), np.inf)
        face = np.full(len(orig), -1)
        if len(sel):
            # rays are given in curve order, so consecutive batches are spatially compact
            for s in range(0, len(orig), 48):
                ii = np.arange(s, min(s + 48, len(orig)))
                lo2 = np.minimum(orig[ii].min(0), ends[ii].min(0)) - 1e-3
                hi2 = np.maximum(orig[ii].max(0), ends[ii].max(0)) + 1e-3
                sub = sel[(tri[sel].max(1) >= lo2).all(1) & (tri[sel].min(1) <= hi2).all(1)]
                if not len(sub):
                    continue
                tt, ff = ray_nearest(orig[ii], dirs[ii], self.V, self.F[sub], tmin=1e-7, tmax=tmax, chunk=16)
                t[ii] = tt
                face[ii] = np.where(ff >= 0, sub[np.maximum(ff, 0)], -1)
        return t, face

    def _candidates(self, pts: np.ndarray, k: int = 16) -> list[np.ndarray]:
        """Faces that can contain the closest point: the k nearest centroids, widened by a ball query
        wherever a large face farther away could still be closer (mixed triangle sizes)."""
        k = min(k, len(self.T))
        dk, idx = self.tree.query(pts, k=k)
        dk, idx = dk.reshape(len(pts), -1), idx.reshape(len(pts), -1)
        out = []
        for i in range(len(pts)):
            f = idx[i]
            a, b, c = self.T[f, 0], self.T[f, 1], self.T[f, 2]
            d = np.linalg.norm(closest_on_triangles(np.repeat(pts[i:i + 1], len(f), 0), a, b, c) - pts[i], axis=1).min()
            if dk[i, -1] < d + self.rad.max():
                f = np.asarray(self.tree.query_ball_point(pts[i], d + self.rad.max()), dtype=np.int64)
                f = f[np.linalg.norm(self.cen[f] - pts[i], axis=1) <= d + self.rad[f] + 1e-9]
            out.append(f)
        return out

    def closest(self, pts: np.ndarray, k: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closest surface point, face index and distance."""
        q, f, d, _ = self._closest_full(pts, k)
        return q, f, d

    def _closest_full(self, pts: np.ndarray, k: int = 16):
        cand = self._candidates(pts, k)
        lens = np.array([len(c) for c in cand])
        flat = np.concatenate(cand)
        owner = np.repeat(np.arange(len(pts)), lens)
        Q = closest_on_triangles(pts[owner], self.T[flat, 0], self.T[flat, 1], self.T[flat, 2])
        D = np.linalg.norm(pts[owner] - Q, axis=1)
        best = np.full(len(pts), np.inf)
        np.minimum.at(best, owner, D)
        tie = D <= best[owner] + 1e-6
        n = np.zeros((len(pts), 3))
        np.add.at(n, owner[tie], self.N[flat[tie]])
        arg = np.where(D <= best[owner], np.arange(len(D)), len(D))
        first = np.full(len(pts), len(D))
        np.minimum.at(first, owner, arg)
        return Q[first], flat[first], best, n

    def signed(self, pts: np.ndarray, k: int = 16) -> np.ndarray:
        """Signed distance; the sign uses the summed normal of every face that attains the minimum
        distance (a pseudo-normal), so points beside a sheet's rim or a slab's edge are not
        mistaken for points inside it."""
        q, _, best, n = self._closest_full(pts, k)
        return np.sign(np.einsum("ij,ij->i", pts - q, n) + 1e-12) * best

    def normal_field(self, pts: np.ndarray, radius: float) -> np.ndarray:
        """Area-weighted mean face normal within ``radius`` (falls back to the nearest face)."""
        out = np.zeros((len(pts), 3))
        for i, nb in enumerate(self.tree.query_ball_point(pts, radius)):
            if nb:
                nb = np.asarray(nb)
                out[i] = (self.N[nb] * self.area[nb, None]).sum(0)
        miss = np.linalg.norm(out, axis=1) < 1e-12
        if miss.any():
            _, f, _ = self.closest(pts[miss])
            out[miss] = self.N[f]
        return normalize(out)


def closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (n, 3) (Ericson, Real-Time Collision Detection)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    out = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
    done = np.zeros(len(p), bool)

    def put(mask, val):
        nonlocal done
        m = mask & ~done
        out[m] = val[m]
        done |= m

    def safe(x):
        return np.where(np.abs(x) < 1e-30, 1e-30, x)
    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    put((d6 >= 0) & (d5 <= d6), c)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / safe(d1 - d3))[:, None])
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / safe(d2 - d6))[:, None])
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + (c - b) * ((d4 - d3) / safe((d4 - d3) + (d5 - d6)))[:, None])
    return out


# ================================================================= draping
def drape(design: np.ndarray, host: Host, *, closed: bool, r: float, along: np.ndarray | None = None,
          normal_radius: float = 0.25, sigma_normal: float = 0.15, sigma_line: float = 0.03,
          step: float = STEP, keep_ends: bool = True, up: float = 0.6, tmax: float | None = None,
          radial: bool = False, relax_sigma: float = 0.06, relax_passes: int = 4,
          lift_only: bool = False) -> tuple[np.ndarray, dict]:
    """Lay a design curve on the outermost host surface; return (centreline, stats).

    ``along`` (unit vector, pointing out of the surface) projects every point along that fixed
    direction (the renderer's analytic cuts are cylinders along x): the ray starts ``up`` outside
    the design point. Otherwise points are cast along the smoothed local surface normal. The
    centre is offset by r + GAP along the smoothed surface normal (along ``along`` for analytic
    cuts, lengthened so the perpendicular clearance is still r + GAP, which keeps the line exactly
    over the cut), then clearance is enforced against the true mesh by pushing the line outward.
    """
    P = resample(design, 0.02, closed)
    h = arclen(P, closed)[-1] / (len(P) if closed else len(P) - 1)
    if along is not None:
        dirs = np.repeat(normalize(np.asarray(along, float))[None], len(P), 0)
    elif radial:   # the scalp layers are radial offsets from the shell centre, so cast radially through them
        dirs = normalize(P - SHELL)
    else:
        dirs = normalize(smooth1d(host.normal_field(P, normal_radius), sigma_normal / h, closed))
    t, _ = host.cast(P + up * dirs, -dirs, 2 * up if tmax is None else tmax)
    miss = ~np.isfinite(t)
    if (~miss).any():
        # outer envelope: a ray that slips through a narrow gap (suture, seam) must not pull the line
        # into it, so take the shallowest hit within +-1.2 mm along the line
        t = np.where(miss, np.inf, t)
        t = np.minimum(t, -maxfilter(-t, max(1, int(round(1.2 * MM / h))), closed))
        miss = ~np.isfinite(t)
    S = P + (up - np.where(miss, up, t))[:, None] * dirs
    if miss.any():   # fall back to the closest surface point
        S[miss] = host.closest(P[miss])[0]
    Ns = normalize(smooth1d(host.normal_field(S, normal_radius), sigma_normal / h, closed))
    if along is not None:
        cosang = np.clip(np.einsum("ij,ij->i", Ns, dirs), 0.35, 1.0)
        off_dir, off = dirs, (r + GAP) / cosang
    else:
        off_dir, off = Ns, np.full(len(S), r + GAP)
    Q = S + off_dir * off[:, None]
    if lift_only:   # keep the design where it is; only push it clear of the surface (toward the viewer)
        Q = P.copy()
        off_dir = Ns if along is None else dirs
    need = r + GAP
    a = None if along is None else normalize(np.asarray(along, float))

    def relax(Q, sigma):
        if sigma <= 0:
            return Q
        if a is None:
            return gsmooth(Q, sigma, closed, keep_ends)
        hgt = Q @ a          # analytic outlines: smooth only the height, so the line stays over the cut
        return Q + (smooth1d(hgt, sigma / h, closed) - hgt)[:, None] * a

    def push_out(Q):
        k = max(1, int(round(0.06 / h)))
        for _ in range(12):   # push out wherever the tube would touch the true surface, with wide ramps
            deficit = need - host.signed(Q)
            if deficit.max() <= 2e-5:
                break
            push = smooth1d(maxfilter(np.maximum(deficit, 0), k, closed), k, closed)
            Q = Q + off_dir * np.minimum(np.maximum(push, deficit) + 1e-4, 0.05)[:, None]
        return Q

    Q = push_out(relax(Q, sigma_line))
    for _ in range(relax_passes):   # alternate smoothing and pushing: an upper envelope without kinks
        Q = push_out(relax(Q, relax_sigma))
    if closed:
        step = min(step, arclen(Q, True)[-1] / 48)
    Q = resample(Q, step, closed)
    sd = host.signed(Q)
    return Q, {"clearanceMinMm": round(float((sd.min() - r) / MM), 3),
               "clearanceMaxMm": round(float((sd.max() - r) / MM), 3),
               "clearanceMedMm": round(float((np.median(sd) - r) / MM), 3), "projectionMisses": int(miss.sum())}


# =================================================================== meshing
def transport_frames(P: np.ndarray, closed: bool, up_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-minimising frames (double reflection, Wang et al. 2008). Closed curves spread the
    holonomy twist evenly so the last frame meets the first."""
    T = tangents(P, closed)
    n = len(P)
    if up_hint is None:
        ref = np.array([0.0, 1.0, 0.0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    else:
        ref = up_hint
    N = np.zeros_like(P)
    N[0] = normalize(ref - T[0] * (ref @ T[0]))
    idx = list(range(n)) + ([0] if closed else [])
    for a, b in zip(idx[:-1], idx[1:]):
        v1 = P[b] - P[a]
        c1 = v1 @ v1
        if c1 < 1e-20:
            nb = N[a]
        else:
            rL = N[a] - (2 / c1) * (v1 @ N[a]) * v1
            tL = T[a] - (2 / c1) * (v1 @ T[a]) * v1
            v2 = T[b] - tL
            c2 = v2 @ v2
            nb = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-20 else rL
        if b == 0 and closed:
            # holonomy: angle from transported frame back to N[0] about T[0]
            nb = normalize(nb - T[0] * (nb @ T[0]))
            ang = np.arctan2(np.cross(nb, N[0]) @ T[0], nb @ N[0])
            s = arclen(P, True)[:n] / arclen(P, True)[-1]
            B0 = np.cross(T, N)
            c, si = np.cos(ang * s)[:, None], np.sin(ang * s)[:, None]
            N = c * N + si * B0
            break
        N[b] = normalize(nb - T[b] * (nb @ T[b]))
    B = np.cross(T, N)
    return N, B


def sweep(P: np.ndarray, radius, closed: bool, sides: int = SIDES, cap0: str = "round", cap1: str = "round",
          head: tuple[float, float] | None = None, up_hint: np.ndarray | None = None,
          cap_rings: int = CAP_RINGS) -> tuple[np.ndarray, np.ndarray]:
    """Unindexed triangles (positions, unit normals) of a tube along P.

    cap0/cap1: 'round' (hemisphere), 'flat' or None (open). ``head`` = (base radius, length) adds a
    cone at the end of an open curve: the tube is trimmed by the head length so the apex lands on
    the last point of P.
    """
    P = np.asarray(P, float)
    tip = None
    if head is not None and not closed:
        rb, hl = head
        s = arclen(P)
        if s[-1] <= hl * 1.05:
            raise ValueError("arrow shorter than its head")
        tip, tdir = P[-1].copy(), None
        keep = s < s[-1] - hl
        base = np.array([np.interp(s[-1] - hl, s, P[:, k]) for k in range(3)])
        P = np.vstack([P[keep], base])
        tdir = normalize(tip - base)
        cap1 = None
    n = len(P)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    N, B = transport_frames(P, closed, up_hint)
    if tip is not None:   # align the last ring with the head axis so the cone sits square
        T_end = tdir
        N[-1] = normalize(N[-1] - T_end * (N[-1] @ T_end))
        B[-1] = np.cross(T_end, N[-1])
    ang = 2 * np.pi * np.arange(sides) / sides
    ring_n = np.cos(ang)[None, :, None] * N[:, None] + np.sin(ang)[None, :, None] * B[:, None]
    ring_p = P[:, None] + r[:, None, None] * ring_n
    V = [ring_p.reshape(-1, 3)]
    NV = [ring_n.reshape(-1, 3)]
    F = []
    rows = n if closed else n - 1
    j = np.arange(sides)
    for i in range(rows):
        i2 = (i + 1) % n
        a, b = i * sides + j, i * sides + (j + 1) % sides
        c, d = i2 * sides + j, i2 * sides + (j + 1) % sides
        F += [np.stack([a, b, c], 1), np.stack([b, d, c], 1)]
    count = n * sides
    T = tangents(P, closed)

    def add_cap(end: int, sign: float, kind: str | None):
        nonlocal count
        if kind is None:
            return
        ring0 = np.arange(sides) + end * sides
        tdir_ = T[end] * sign
        if tip is not None and end == n - 1:
            tdir_ = tdir
        if kind == "flat":
            V.append(P[end][None]); NV.append(tdir_[None])
            c0 = count; count += 1
            V.append(ring_p[end]); NV.append(np.repeat(tdir_[None], sides, 0))
            r0 = np.arange(sides) + count; count += sides
            tri = np.stack([np.full(sides, c0), r0, np.roll(r0, -1)], 1)
            F.append(tri if sign > 0 else tri[:, [0, 2, 1]])
            return
        prev = ring0
        for k in range(1, cap_rings + 1):
            phi = 0.5 * np.pi * k / (cap_rings + 1)
            rn = np.cos(phi) * ring_n[end] + np.sin(phi) * tdir_[None]
            V.append(P[end][None] + r[end] * rn); NV.append(rn)
            cur = np.arange(sides) + count; count += sides
            q = np.stack([prev, np.roll(prev, -1), cur], 1), np.stack([np.roll(prev, -1), np.roll(cur, -1), cur], 1)
            F.extend([x if sign > 0 else x[:, [0, 2, 1]] for x in q])
            prev = cur
        V.append((P[end] + r[end] * tdir_)[None]); NV.append(tdir_[None])
        pole = count; count += 1
        tri = np.stack([prev, np.roll(prev, -1), np.full(sides, pole)], 1)
        F.append(tri if sign > 0 else tri[:, [0, 2, 1]])

    if not closed:
        add_cap(0, -1.0, cap0)
        add_cap(n - 1, 1.0, cap1)
    if tip is not None:
        rb, hl = head
        base_c = P[-1]
        rn = ring_n[-1]
        # flange from the tube ring out to the cone base (faces backwards)
        V.append(base_c[None] + rb * rn); NV.append(np.repeat(-tdir[None], sides, 0))
        fl = np.arange(sides) + count; count += sides
        V.append(ring_p[-1]); NV.append(np.repeat(-tdir[None], sides, 0))
        inner = np.arange(sides) + count; count += sides
        F += [np.stack([inner, fl, np.roll(fl, -1)], 1), np.stack([inner, np.roll(fl, -1), np.roll(inner, -1)], 1)]
        # cone side: normals tilted toward the apex
        L = np.hypot(rb, hl)
        cn = rn * (hl / L) + tdir[None] * (rb / L)
        V.append(base_c[None] + rb * rn); NV.append(cn)
        cb = np.arange(sides) + count; count += sides
        mid = normalize(cn + np.roll(cn, -1, 0))
        V.append(np.repeat(tip[None], sides, 0)); NV.append(mid)
        ap = np.arange(sides) + count; count += sides
        F.append(np.stack([cb, np.roll(cb, -1), ap], 1))
    V = np.concatenate(V)
    NV = normalize(np.concatenate(NV))
    F = np.concatenate(F)
    # orient every face so its geometric normal agrees with its vertex normals
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum("ij,ij->i", fn, NV[F].sum(1)) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V[F.reshape(-1)], NV[F.reshape(-1)]


# ======================================================= imported geometry
def old_chains(m: dict) -> list[tuple[np.ndarray, bool]]:
    """Centrelines of an imported marking: per segment ``sides`` x (side quad, start cap, end cap)
    triangles; consecutive segments sharing an end point form a chain."""
    p, _ = decode(m)
    if len(p) % 12:
        raise ValueError(f"{m['id']}: not an imported prism-chain marking")
    blk = p.reshape(-1, 12, 3)
    A, B = blk[:, 6], blk[:, 9]
    same = np.r_[False, (np.abs(A[1:] - A[:-1]).max(1) < 1e-6) & (np.abs(B[1:] - B[:-1]).max(1) < 1e-6)]
    st = np.where(~same)[0]
    A, B = A[st], B[st]
    chains, cur = [], [0]
    for k in range(1, len(A)):
        if np.linalg.norm(A[k] - B[k - 1]) < 1e-5:
            cur.append(k)
        else:
            chains.append(cur)
            cur = [k]
    chains.append(cur)
    out = []
    for c in chains:
        closed = len(c) > 2 and np.linalg.norm(B[c[-1]] - A[c[0]]) < 1e-5
        pts = A[c] if closed else np.vstack([A[c], B[c[-1:]]])
        out.append((pts, closed))
    return out


def ellipse_yz(e: dict, n: int = 720, scale: float = 1.0) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([np.zeros(n), e["cy"] + scale * e["ry"] * np.cos(a), e["cz"] + scale * e["rz"] * np.sin(a)], 1)


def window_point(theta: float, q: float, e: dict = WINDOW) -> np.ndarray:
    return np.array([0.0, e["cy"] + q * e["ry"] * np.cos(theta), e["cz"] + q * e["rz"] * np.sin(theta)])


def window_angle(p: np.ndarray, e: dict = WINDOW) -> float:
    return float(np.arctan2((p[2] - e["cz"]) / e["rz"], (p[1] - e["cy"]) / e["ry"]))


# ================================================================ designs
def design_smooth(m_right: dict, sigma: float, chain: int = 0) -> tuple[np.ndarray, bool]:
    P, closed = old_chains(m_right)[chain]
    P = resample(P, 0.01, closed)
    return gsmooth(P, sigma, closed), closed


def flow_line(host: Host, start: np.ndarray, direction, length: float, stop=None, h: float = 0.02) -> np.ndarray:
    """March over a surface along the tangent part of ``direction(p)`` (re-projected each step)."""
    pts = [start]
    p = start
    for _ in range(int(length / h)):
        n = host.normal_field(p[None], 0.25)[0]
        d = direction(p)
        d = d - n * (d @ n)
        if np.linalg.norm(d) < 1e-9:
            break
        q = p + h * normalize(d)
        t, _ = host.cast(q[None] + 0.5 * n[None], -n[None], 1.0)
        if np.isfinite(t[0]):
            q = q + (0.5 - t[0]) * n
        if stop is not None and stop(q):
            break
        pts.append(q)
        p = q
    return np.array(pts)


def top_outline(m: dict, mirror: bool) -> np.ndarray:
    """Boundary loop of the outer (skin) surface of a scalp slab record."""
    p, n = decode(m)
    if mirror:
        p, n = mirror_x(p, n)
    T = p.reshape(-1, 3, 3)
    N = normalize(n.reshape(-1, 3, 3).sum(1))
    top = np.einsum("ij,ij->i", N, normalize(T.mean(1) - SHELL)) > 0.5
    V, F = weld(T[top].reshape(-1, 3), 1e-6)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    adj: dict[int, list[int]] = {}
    for a, b in u[c == 1]:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    start = min(adj)
    loop, prev, cur = [start], -1, start
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    if len(loop) != len(adj):
        raise RuntimeError(f"{m['id']}: skin outline is not a single loop")
    return V[loop]


# ================================================================ builders
class Side:
    """Host surfaces of one side, in the right-side (canonical) frame."""

    def __init__(self, records: list[dict], side: str):
        self.side, self.mirror = side, side == "l"
        self.records = records
        sfx = "" if side == "r" else "-l"
        osfx = "-l" if side == "r" else ""
        rec = lambda i: find(records, id=i)                       # noqa: E731
        bones = [find(records, native=b) for b in BONES]
        tmus = find(records, native="Temporalis muscle." + side)
        self.flap = rec("ILL-scalp-flap" + sfx)
        self._lazy = {
            "bone": lambda: Host(bones, mirror=self.mirror),
            "bonePeri": lambda: Host(bones + [rec(i + sfx) for i in PERI_STATIONARY], mirror=self.mirror),
            "periAll": lambda: Host(bones + [rec(i + s) for i in PERI for s in (sfx, osfx)], mirror=self.mirror),
            "bed": lambda: Host(bones + [rec(i + s) for i in PERI for s in (sfx, osfx)]
                                + [rec(i + sfx) for i in TEMPORAL] + [tmus], mirror=self.mirror),
            "skin": lambda: Host([rec(i + s) for i in SKIN for s in (sfx, osfx)], mirror=self.mirror),
            "galea": lambda: Host([rec(i + s) for i in GALEA for s in (sfx, osfx)], mirror=self.mirror),
            "dura": lambda: Host([rec("ILL-dura" + sfx)], mirror=self.mirror),
            "boneOp": lambda: Host(bones, facing=OP_AXIS, min_dot=0.05, mirror=self.mirror),
        }
        self._hosts: dict[str, Host] = {}

    def host(self, key: str) -> Host:
        if key not in self._hosts:
            self._hosts[key] = self._lazy[key]()
        return self._hosts[key]

    def incision(self) -> np.ndarray:
        P = np.asarray(self.flap["incisionLine"], float)
        return P * np.array([-1.0, 1, 1]) if self.mirror else P

    def incision_mark(self) -> np.ndarray:
        """Course of the incision marking: the incision without its first ``INCISION_TRIM`` (it starts on the
        scalp's free edge at the zygomatic arch, where the illustrative scalp and the field outline end)."""
        P = resample(self.incision(), 0.005)
        return P[np.searchsorted(arclen(P), INCISION_TRIM):]

    def flap_fields(self) -> dict:
        f = self.flap["flap"]
        s = np.array([-1.0, 1, 1]) if self.mirror else np.ones(3)
        return {k: np.asarray(f[k], float) * s for k in ("pivot", "axis", "away", "lift")}


# operative view axis (entry - focus of the renderer's Operative View, right side)
OP_AXIS = normalize(np.array([-3.05, 0.30, 1.25]) - np.array([-1.55, -0.3, 0.60]))
OUT_X = np.array([-1.0, 0.0, 0.0])    # outward along the window cylinder axis (right side)


def line(Q, closed, r=R_LINE, **kw):
    return sweep(Q, r, closed, **kw)


def lift_head(Q: np.ndarray, host: Host, r: float, head: tuple[float, float]) -> np.ndarray:
    """Raise the end of a draped arrow so its wider cone clears the surface (smooth ramp)."""
    rb, hl = head
    s = arclen(Q)
    w = np.clip((s - (s[-1] - hl - 0.15)) / 0.15, 0, 1)
    w = w * w * (3 - 2 * w)
    n = host.normal_field(Q, 0.2)
    return Q + n * ((rb - r) * w)[:, None]


def build_marker(base: str, S: Side, right: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Geometry (right-side frame) and info for one marking. ``right`` holds the right-side records
    (designs come from the canonical right side)."""
    info: dict = {}
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    lines: list[tuple[np.ndarray, bool]] = []

    def put(Q, closed, st=None, r=R_LINE, **kw):
        if r == R_DASH:
            kw = {"sides": DASH_SIDES, "cap_rings": 2, **kw}
        parts.append(line(Q, closed, r, **kw))
        lines.append((Q, closed))
        if st:
            for k, v in st.items():
                info[k] = min(info.get(k, v), v) if k.endswith("MinMm") else max(info.get(k, v), v)

    if base == "MARK-pterion":
        D, closed = design_smooth(right[base], 0.05)
        Q, st = drape(D, S.host("bone"), closed=True, r=R_LINE, normal_radius=0.2)
        put(Q, True, st)
        info["surface"] = "bone"
    elif base == "MARK-incision":
        Q, st = drape(S.incision_mark(), S.host("skin"), closed=False, r=R_LINE, sigma_line=0.02)
        put(Q, False, st)
        info["surface"] = "skin"
    elif base in ("MARK-scalp-sutures", GALEA_SUTURES):
        layer = "skin" if base == "MARK-scalp-sutures" else "galea"
        inc, st0 = drape(S.incision_mark(), S.host(layer), closed=False, r=R_DASH, sigma_line=0.02, step=0.01)
        host = S.host(layer)
        s = arclen(inc)
        T = tangents(inc, False)
        nrm = host.normal_field(inc, 0.2)
        for t in np.linspace(0.05, 0.95, 15) * s[-1]:
            i = int(np.searchsorted(s, t))
            c, d = inc[i], normalize(np.cross(nrm[i], T[i]))
            D = c[None] + np.linspace(-0.13, 0.13, 9)[:, None] * d[None]
            Q, st = drape(D, host, closed=False, r=R_DASH, sigma_line=0.03, step=DASH_STEP, up=0.3)
            put(Q, False, st, r=R_DASH)
        info["surface"] = layer
    elif base == "MARK-bone-outline":
        D = ellipse_yz(WINDOW)
        D[:, 0] = -6.0
        Q, st = drape(D, S.host("bonePeri"), closed=True, r=R_LINE, along=OUT_X, up=0.0, tmax=6.5, sigma_line=0.04)
        put(Q, True, st)
        info["surface"] = "bone (outer table; pericranium where it remains)"
    elif base in BURRS:
        cy, cz = BURRS[base]
        D = ellipse_yz(dict(cy=cy, cz=cz, ry=BURR_RING, rz=BURR_RING), n=240)
        D[:, 0] = -6.0
        Q, st = drape(D, S.host("bonePeri"), closed=True, r=R_LINE, along=OUT_X, up=0.0, tmax=6.5, sigma_line=0.04)
        put(Q, True, st)
        info["surface"] = "bone (outer table; pericranium where it remains)"
    elif base in ("MARK-dural-cut", "MARK-dural-low"):
        e = DURAL if base == "MARK-dural-cut" else DURAL_LOW
        D = ellipse_yz(e)
        D[:, 0] = -6.0
        Q, st = drape(D, S.host("dura"), closed=True, r=R_LINE, along=OUT_X, up=0.0, tmax=6.5, sigma_line=0.04)
        put(Q, True, st)
        info["surface"] = "dura"
    elif base == "MARK-tacks":
        for (P, _) in old_chains(right[base]):
            th = window_angle(P.mean(0))
            D = np.array([window_point(th, q) for q in np.linspace(0.78, 0.92, 7)])
            D[:, 0] = -6.0
            Q, st = drape(D, S.host("dura"), closed=False, r=R_DASH, along=OUT_X, up=0.0, tmax=6.5, step=DASH_STEP,
                          sigma_line=0.03)
            put(Q, False, st, r=R_DASH)
        info["surface"] = "dura"
    elif base == "MARK-fixation":
        for (P, _) in old_chains(right[base]):
            th = window_angle(P.mean(0))
            D = np.array([window_point(th, q) for q in np.linspace(0.88, 1.12, 9)])
            D[:, 0] = -6.0
            Q, st = drape(D, S.host("bonePeri"), closed=False, r=R_DASH, along=OUT_X, up=0.0, tmax=6.5, step=DASH_STEP,
                          sigma_line=0.03)
            put(Q, False, st, r=R_DASH)
        info["surface"] = "bone (outer table; pericranium where it remains)"
    elif base == "MARK-periosteum":
        D, closed = design_smooth(right[base], 0.12)
        Q, st = drape(D, S.host("periAll"), closed=True, r=R_LINE)
        put(Q, True, st)
        info["surface"] = "pericranium / bone"
    elif base == "MARK-sphenoid":
        D, closed = design_smooth(right[base], 0.08)
        Q, st = drape(D, S.host("boneOp"), closed=False, r=R_LINE, normal_radius=0.15, up=0.4)
        put(Q, False, st)
        info["surface"] = "bone (sphenoid ridge)"
    elif base in ("MARK-sylvian", "MARK-cistern"):
        P, closed = old_chains(right[base])[0]
        D = gsmooth(resample(P, 0.01, closed), 0.16 if base == "MARK-sylvian" else 0.1, closed)
        put(resample(D, STEP / 2, closed), closed)
        info["surface"] = "deep (smoothed, not draped)"
    elif base == "MARK-clinoid":
        P, _ = old_chains(right[base])[0]
        put(resample(clinoid_loop(P), STEP / 2, True), True)
        info["surface"] = "deep (smoothed, not draped)"
    elif base in ("MARK-drain", "MARK-muscle-sutures"):
        D, closed = design_smooth(right[base], 0.1 if base == "MARK-drain" else 0.06)
        Q, st = drape(D, S.host("bed"), closed=False, r=R_LINE)
        put(Q, False, st)
        info["surface"] = "bed under the scalp (pericranium, temporal fascia, bone)"
    elif base == "MARK-position":
        P, _ = old_chains(right[base])[0]
        D = gsmooth(resample(P, 0.01), 0.3)
        Q, st = drape(D, S.host("skin"), closed=False, r=R_LINE)
        put(lift_head(Q, S.host("skin"), R_LINE, ARROW_HEAD), False, st, head=ARROW_HEAD)
        info["surface"] = "skin"
    elif base == "MARK-field":
        Q, st = drape(field_design(S, right), S.host("skin"), closed=True, r=R_LINE, sigma_line=0.04)
        put(Q, True, st)
        info["surface"] = "skin"
    elif base == "MARK-flap":
        Q, st = drape(flap_margin_design(S, right), S.host("bed"), closed=True, r=R_LINE, sigma_line=0.04,
                      radial=True, up=1.0, tmax=2.0)
        put(Q, True, st)
        info["surface"] = "bed under the scalp flap (pericranium, temporal fascia, bone)"
    elif base == "MARK-flap-arrow":
        D = gsmooth(resample(flap_arrow_line(S, right), 0.01), 0.1)
        Q, st = drape(D, S.host("bed"), closed=False, r=R_LINE)
        put(lift_head(Q, S.host("bed"), R_LINE, ARROW_HEAD), False, st, head=ARROW_HEAD)
        info["surface"] = "bed under the scalp flap (temporal fascia)"
    else:
        raise KeyError(base)
    if not parts:
        raise RuntimeError(base)
    pos = np.concatenate([p for p, _ in parts])
    nrm = np.concatenate([n for _, n in parts])
    info["host"] = HOST_OF.get(base)
    info["lines"] = lines
    return pos, nrm, info


HOST_OF = {"MARK-pterion": "bone", "MARK-incision": "skin", "MARK-field": "skin", "MARK-scalp-sutures": "skin",
           GALEA_SUTURES: "galea",
           "MARK-position": "skin", "MARK-flap": "bed", "MARK-flap-arrow": "bed", "MARK-drain": "bed",
           "MARK-muscle-sutures": "bed", "MARK-periosteum": "periAll", "MARK-burr-1": "bonePeri",
           "MARK-burr-2": "bonePeri", "MARK-bone-outline": "bonePeri", "MARK-fixation": "bonePeri",
           "MARK-sphenoid": "bone", "MARK-tacks": "dura", "MARK-dural-cut": "dura", "MARK-dural-low": "dura"}


def clinoid_loop(P: np.ndarray) -> np.ndarray:
    """Clean outline of the imported clinoid marking: an ellipse in the points' best-fit plane (principal
    axes, 1.2 x the half extents), lifted onto a quadratic surface fitted through the points."""
    U = np.unique(np.round(P, 6), axis=0)
    c = U.mean(0)
    _, _, vt = np.linalg.svd(U - c)
    X = (U - c) @ vt.T
    ctr = 0.5 * (X[:, :2].max(0) + X[:, :2].min(0))
    half = 0.5 * np.ptp(X[:, :2], 0)
    ab = half * 1.2
    t = 2 * np.pi * np.arange(240) / 240
    u, v = ctr[0] + ab[0] * np.cos(t), ctr[1] + ab[1] * np.sin(t)
    A = np.stack([np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2], 1)
    coef = np.linalg.lstsq(A, X[:, 2], rcond=None)[0]
    w = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1) @ coef
    return c + np.stack([u, v, w], 1) @ vt


# ------------------------------------------------------- scalp-level designs
def chart_basis(dirs: np.ndarray):
    c0 = normalize(normalize(dirs).mean(0))
    e1 = normalize(np.cross(c0, [0.0, 0.0, 1.0]))
    e2 = np.cross(c0, e1)
    return c0, e1, e2


def to_chart(P: np.ndarray, basis) -> np.ndarray:
    """Azimuthal-equidistant chart of directions from the shell centre."""
    c0, e1, e2 = basis
    d = normalize(P - SHELL)
    ang = np.arccos(np.clip(d @ c0, -1, 1))
    u, v = d @ e1, d @ e2
    n = np.maximum(np.hypot(u, v), 1e-12)
    return np.stack([ang * u / n, ang * v / n], 1)


def from_chart(X: np.ndarray, basis) -> np.ndarray:
    c0, e1, e2 = basis
    ang = np.linalg.norm(X, axis=1)
    n = np.maximum(ang, 1e-12)
    return normalize(np.cos(ang)[:, None] * c0 + np.sin(ang)[:, None] * ((X[:, :1] / n[:, None]) * e1 + (X[:, 1:] / n[:, None]) * e2))


def polar_hit(center: np.ndarray, angles: np.ndarray, poly: np.ndarray, closed: bool, far: bool) -> np.ndarray:
    """Distance from ``center`` along each angle to the nearest (or farthest) crossing of a polyline."""
    A = poly
    B = np.roll(poly, -1, 0) if closed else poly[1:]
    A = A if closed else poly[:-1]
    d = np.stack([np.cos(angles), np.sin(angles)], 1)
    e = B - A
    w = A - center
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    den = np.where(np.abs(den) < 1e-15, 1e-15, den)
    t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / den
    u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / den
    ok = (t > 0) & (u >= 0) & (u <= 1)
    if far:
        return np.where(ok, t, -np.inf).max(1)
    return np.where(ok, t, np.inf).min(1)


def skin_free_edge(S: Side) -> np.ndarray:
    """Free edge of the illustrative scalp (outer boundary of the union of both sides' skin tops)."""
    tops = []
    for sfx in ("", "-l"):
        for i in SKIN:
            m = find(S.records, id=i + sfx)
            p, n = decode(m)
            if S.mirror:
                p, n = mirror_x(p, n)
            T = p.reshape(-1, 3, 3)
            N = normalize(n.reshape(-1, 3, 3).sum(1))
            tops.append(T[np.einsum("ij,ij->i", N, normalize(T.mean(1) - SHELL)) > 0.5])
    V, F = weld(np.concatenate(tops).reshape(-1, 3), 2e-3)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    E = E[E[:, 0] != E[:, 1]]
    u, c = np.unique(E, axis=0, return_counts=True)
    return V[np.unique(u[c == 1])]


def field_design(S: Side, right: dict) -> np.ndarray:
    """Prepared-field boundary: the imported field outline grown to enclose the scalp incision with a
    12 mm margin, kept 3 mm inside the scalp's free edge, smoothed (chart of directions from the
    shell centre, polar about the region's centre). The incision margin is re-imposed after the
    smoothing (which would otherwise erode the narrow buffer around the incision's ends). Where the
    margin would pass the free edge (the incision starts on it at the zygomatic arch) the boundary
    runs along the edge, ``FIELD_EDGE_INSET`` inside it."""
    inc = S.incision()
    old, _ = old_chains(right["MARK-field"])[0]
    basis = chart_basis(np.vstack([inc, old]) - SHELL)
    Xi, Xo = to_chart(resample(inc, 0.02), basis), to_chart(resample(old, 0.02, True), basis)
    Rk = np.median(np.linalg.norm(inc - SHELL, axis=1))          # skin radius (units) for mm -> chart
    edge = to_chart(skin_free_edge(S), basis)
    center = np.vstack([Xi, Xo]).mean(0)
    n = 720
    ang = 2 * np.pi * np.arange(n) / n
    rho_old = polar_hit(center, ang, Xo, True, far=True)
    # incision buffer of radius m, in polar form
    m = 12 * MM / Rk
    rel = Xi - center
    ri, ai = np.linalg.norm(rel, axis=1), np.arctan2(rel[:, 1], rel[:, 0])
    da = (ang[:, None] - ai[None] + np.pi) % (2 * np.pi) - np.pi
    perp = ri[None] * np.sin(np.abs(da))
    along = ri[None] * np.cos(da) + np.sqrt(np.maximum(m * m - perp ** 2, 0))
    rho_buf = np.where((perp <= m) & (np.cos(da) > 0), along, -np.inf).max(1)
    rho = np.maximum(rho_old, rho_buf)
    # free edge (nearest edge point in each angular bin, interpolated), 3 mm inset
    rel = edge - center
    re_, ae = np.linalg.norm(rel, axis=1), np.arctan2(rel[:, 1], rel[:, 0])
    bins = ((ae % (2 * np.pi)) / (2 * np.pi) * n).astype(int) % n
    rho_edge = np.full(n, np.inf)
    np.minimum.at(rho_edge, bins, re_)
    rho_edge = -maxfilter(-rho_edge, 3, True)             # bridge empty bins conservatively
    tight = smooth1d((rho_buf > rho_edge - 3 * MM / Rk).astype(float), 6, True) > 0.05
    lim = rho_edge - np.where(tight, FIELD_EDGE_INSET, 3 * MM) / Rk
    k = 12                                                 # 6 degree gaussian in angle
    for _ in range(3):
        rho = np.minimum(rho, lim)
        rho = smooth1d(rho, k, True)
    # re-impose the incision margin (bounded by the edge limit), smoothing gently without losing it
    need = np.minimum(rho_buf, lim)
    for _ in range(4):
        rho = smooth1d(np.maximum(rho, need), 3, True)
    rho = np.minimum(np.maximum(rho, need), np.maximum(lim, rho_edge - 1.5 * MM / Rk))
    X = center + rho[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    d = from_chart(X, basis)
    return SHELL + d * Rk


def inside_polygon(X: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test (2D points X against the closed polygon poly)."""
    A, B = poly, np.roll(poly, -1, 0)
    y, x = X[:, None, 1], X[:, None, 0]
    cross = (A[None, :, 1] > y) != (B[None, :, 1] > y)
    xi = A[None, :, 0] + (y - A[None, :, 1]) * (B[None, :, 0] - A[None, :, 0]) / np.where(
        np.abs(B[None, :, 1] - A[None, :, 1]) < 1e-15, 1e-15, B[None, :, 1] - A[None, :, 1])
    return ((cross & (x < xi)).sum(1) % 2) == 1


def field_margin(S: Side, field: np.ndarray, mark: np.ndarray) -> dict:
    """Signed margin (mm, positive inside) of the incision marking's centreline inside the field outline,
    measured in the chart of directions from the shell centre; also the incision's distance to the free edge."""
    basis = chart_basis(np.vstack([mark, field]) - SHELL)
    Rk = np.median(np.linalg.norm(mark - SHELL, axis=1))
    Xf, Xi = to_chart(field, basis), to_chart(mark, basis)
    d = cKDTree(Xf).query(Xi)[0] * Rk / MM
    sd = np.where(inside_polygon(Xi, Xf), d, -d)
    de = cKDTree(to_chart(skin_free_edge(S), basis)).query(Xi)[0] * Rk / MM
    away = de >= 15.0
    return {"fieldMarginMinMm": round(float(sd.min()), 2),
            "fieldMarginAwayFromEdgeMinMm": round(float(sd[away].min()), 2) if away.any() else None}


def flap_margin_design(S: Side, right: dict) -> np.ndarray:
    """Flap margin: the imported outline of the area exposed by the anterior flap reflection,
    smoothed and kept at least 4 mm in front of the scalp incision and 6 mm behind the flap's hinge
    line (so the whole loop lies on bed that the reflected flap uncovers)."""
    old, _ = old_chains(right["MARK-flap"])[0]
    old = gsmooth(resample(old, 0.01, True), 0.12, True)
    f = S.flap_fields()
    outline = top_outline(S.flap, S.mirror)
    hinge = outline[np.abs((outline - f["pivot"]) @ f["away"]) < 0.02]
    hinge = hinge[np.argsort(hinge @ f["axis"])]
    inc = resample(S.incision(), 0.02)
    basis = chart_basis(old - SHELL)
    Xo, Xi, Xh = to_chart(old, basis), to_chart(inc, basis), to_chart(hinge, basis)
    Rk = np.median(np.linalg.norm(old - SHELL, axis=1))
    center = Xo.mean(0)
    n = 720
    ang = 2 * np.pi * np.arange(n) / n
    rho = polar_hit(center, ang, Xo, True, far=True)
    rho = np.minimum(rho, polar_hit(center, ang, Xi, False, far=False) - 4 * MM / Rk)
    rho = np.minimum(rho, polar_hit(center, ang, Xh, False, far=False) - 6 * MM / Rk)
    rho = smooth1d(rho, 16, True)
    X = center + rho[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    d = from_chart(X, basis)
    return SHELL + d * Rk


def flap_arrow_line(S: Side, right: dict) -> np.ndarray:
    """Arrow on the exposed bed from the imported arrow's tail along the flap's reflection direction
    (the tangent part of -away), stopping before the hinge roll."""
    f = S.flap_fields()
    host = S.host("bed")
    P, _ = old_chains(right["MARK-flap-arrow"])[0]
    start = host.closest(P[:1])[0][0]
    old_len = arclen(P)[-1]
    stop = lambda q: (q - f["pivot"]) @ f["away"] < 0.55          # noqa: E731
    return flow_line(host, start, lambda p: -f["away"], old_len, stop)


def build_tool(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Instrument arrow: shaft from the far end to a cone whose apex is the record's tip (anchor)."""
    chains = old_chains(m)
    far = chains[0][0][0]
    tip = np.asarray(m["anchor"], float)
    P = np.array([far, tip])
    return sweep(P, TOOL_SHAFT, False, sides=TOOL_SIDES, cap0="round", head=TOOL_HEAD)


# ============================================================ reveal fields
def monotone_fraction(v: np.ndarray) -> float:
    d = np.diff(v)
    return float((d > 0).mean()) if len(d) else 1.0


def loop_reveal_centre(Q: np.ndarray) -> np.ndarray:
    """(y, z) about which the loop's angle increases monotonically (or as nearly as possible)."""
    yz = Q[:, 1:]
    c0 = yz.mean(0)
    span = np.ptp(yz, 0)
    best, bc = -1.0, c0
    for fy in np.linspace(-0.3, 0.3, 13):
        for fz in np.linspace(-0.3, 0.3, 13):
            c = c0 + np.array([fy, fz]) * span
            a = np.unwrap(np.arctan2(yz[:, 1] - c[1], yz[:, 0] - c[0]))
            sgn = np.sign(a[-1] - a[0])
            score = float((np.diff(a) * sgn > 0).mean()) - 1e-3 * np.hypot(fy, fz)
            if score > best:
                best, bc = score, c
    return bc


def reveal_fields(m: dict, lines: list[tuple[np.ndarray, bool]], r: float) -> dict:
    """markerCenter / markerAxis / markerRange for the new geometry (display coordinates)."""
    allp = np.vstack([Q for Q, _ in lines])
    out = {"markerRange": [float(allp[:, 1].min() - r), float(allp[:, 1].max() + r)], "markerAxis": 1}
    kind = m.get("markerKind", 0)
    if kind == 1:
        Q = max(lines, key=lambda x: len(x[0]))[0]
        c = loop_reveal_centre(Q)
        out["markerCenter"] = [float(Q[:, 0].mean()), float(c[0]), float(c[1])]
        a = np.unwrap(np.arctan2(Q[:, 2] - c[1], Q[:, 1] - c[0]))
        out["_monotone"] = max(monotone_fraction(a), monotone_fraction(-a))
    elif kind == 2:
        # keep the old sweep's start end; pick the axis along which the line rises most consistently
        Q = np.vstack([q for q, _ in lines]) if len(lines) > 1 else lines[0][0]
        old_axis = int(m.get("markerAxis", 1))
        if Q[0, old_axis] > Q[-1, old_axis]:
            Q = Q[::-1]
        if len(lines) > 1:   # dashes: order by their position along the old axis
            Q = np.array([q.mean(0) for q, _ in sorted(lines, key=lambda x: x[0].mean(0)[old_axis])])
        scores = [monotone_fraction(Q[:, a]) if Q[-1, a] > Q[0, a] else -1 for a in range(3)]
        axis = old_axis if scores[old_axis] >= max(scores) - 1e-9 else int(np.argmax(scores))
        out["markerAxis"] = axis
        out["markerRange"] = [float(allp[:, axis].min() - r), float(allp[:, axis].max() + r)]
        out["_monotone"] = scores[axis]
    return out


# ===================================================================== apply
DESC_KEEP = ("Schematic educational marking on reference anatomy; not a measured surgical landmark or "
             "patient-specific trajectory.")


DASHED = ("MARK-tacks", "MARK-fixation", "MARK-scalp-sutures", GALEA_SUTURES)
SURFACE_TEXT = {
    "skin": "the illustrative scalp skin",
    "galea": "the illustrative galea",
    "bed": "the tissue bed exposed under the reflected scalp flap (pericranium, temporal fascia or bone)",
    "periAll": "the pericranium, or the bone where there is none",
    "bonePeri": "the outer table of the bone, or the pericranium where it remains",
    "bone": "the bone",
    "dura": "the illustrative dura",
}
DESIGN_TEXT = {
    "MARK-incision": (" It follows the illustrative scalp incision and stops 5 mm above the scalp's free edge at the "
                      "zygomatic arch, where the illustrative scalp ends."),
    "MARK-scalp-sutures": " Skin stitches are evenly spaced across the illustrative scalp incision (skin closure).",
    GALEA_SUTURES: (" Buried galeal stitches are evenly spaced across the illustrative incision in the galea "
                    "(galea closure, drawn between the still-open skin edges)."),
    "MARK-field": (" The outline is the earlier marking widened to enclose the illustrative incision with a margin "
                   "of at least 8 mm; at the zygomatic arch, where the incision meets the scalp's free edge, it runs "
                   "along that edge."),
    "MARK-flap": " It is kept between the scalp incision and the scalp flap's hinge line.",
    "MARK-flap-arrow": " It points along the scalp flap's reflection direction.",
    "MARK-bone-outline": " It lies exactly over the renderer's bone-window cut.",
    "MARK-burr-1": " It circles the renderer's burr hole.",
    "MARK-burr-2": " It circles the renderer's burr hole.",
    "MARK-dural-cut": " It lies exactly over the renderer's dural opening.",
    "MARK-dural-low": " It lies exactly over the renderer's dural opening.",
    "MARK-tacks": " Ticks point radially toward the bone-window margin.",
    "MARK-fixation": " Plates cross the bone-window margin radially.",
}


def describe(base: str, info: dict, side: str) -> str:
    if info["surface"].startswith("deep"):
        how = (f"Smoothed from the original schematic marking (deep position kept; largest change "
               f"{info['deviationMm']:.1f} mm); drawn as a {R_LINE / MM:.1f} mm-radius line.")
    else:
        r = R_DASH if base in DASHED else R_LINE
        lo, med, hi = info["clearanceMinMm"], info["clearanceMedMm"], info["clearanceMaxMm"]
        how = (f"Drawn as a smooth line of constant {r / MM:.1f} mm radius whose underside lies "
               f"{max(lo, GAP / MM):.1f}–{hi:.1f} mm (median {med:.1f} mm) above "
               f"{SURFACE_TEXT[info['host']]}." + DESIGN_TEXT.get(base, ""))
        if side == "l":
            how += " Fitted on the native left anatomy."
    return DESC_KEEP + " " + how


def apply(records: list[dict]) -> dict:
    """Regenerate all marking and instrument geometry (both sides). Refuses to run twice."""
    marks = [m for m in records if m["group"] in ("annotations", "instruments")]
    if any("markerRadius" in m for m in marks):
        raise RuntimeError("markers already regenerated (records carry markerRadius); run on unprocessed records")
    missing = [i for i in ("ILL-scalp-flap", "ILL-scalp-flap-l", "ILL-dura", "ILL-superfascia") if not any(m["id"] == i for m in records)]
    if missing:
        raise RuntimeError(f"markers needs the scalp, dura and temporal_layers steps first (missing {missing})")
    # snapshot of the imported right-side records (designs are read from them on both passes)
    # galeal closure stitches: new records cloned from the skin closure stitches (same colour, reveal kind)
    for sfx in ("", "-l"):
        src = find(records, id="MARK-scalp-sutures" + sfx)
        new = {k: (list(v) if isinstance(v, list) else v) for k, v in src.items()}
        new["id"] = GALEA_SUTURES + sfx
        records.insert(records.index(src) + 1, new)
        marks.insert(marks.index(src) + 1, new)
    for m in marks:
        b = m["id"][:-2] if m["id"].endswith("-l") else m["id"]
        if b in STAGES:
            m["markerStages"] = list(STAGES[b])
        if b in NAMES:
            m["name"] = NAMES[b]
            if "nativeName" in m:
                m["nativeName"] = NAMES[b]
    right = {m["id"]: dict(m) for m in marks if m.get("procedureSide") == "right"}
    old_counts = {m["id"]: m["count"] for m in marks}
    summary: dict = {"records": [], "radiusMm": {"line": R_LINE / MM, "dash": R_DASH / MM, "tool": TOOL_SHAFT / MM},
                     "gapMm": GAP / MM, "sides": SIDES, "stepMm": STEP / MM}
    base_ids = sorted(i for i, m in right.items() if m.get("marker"))
    for side in ("r", "l"):
        S = Side(records, side)
        for base in base_ids:
            m = find(records, id=base if side == "r" else base + "-l")
            pos, nrm, info = build_marker(base, S, right)
            lines = info.pop("lines")
            deep = info["surface"].startswith("deep")
            if deep:
                old = np.vstack([P for P, _ in old_chains(right[base])])
                allq = np.vstack([q for q, _ in lines])
                info["deviationMm"] = float(cKDTree(allq).query(old)[0].max() / MM)
            if S.mirror:
                pos, nrm = mirror_x(pos, nrm)
                lines = [(Q * np.array([-1.0, 1, 1]), c) for Q, c in lines]
            r = R_DASH if base in DASHED else R_LINE
            old_anchor = np.asarray(m["anchor"], float)
            encode(m, pos, nrm)
            rv = reveal_fields(m, lines, r)
            mono = rv.pop("_monotone", None)
            allq = np.vstack([q for q, _ in lines])
            anchor = allq[np.argmin(np.linalg.norm(allq - old_anchor, axis=1))]
            m["anchor"] = [float(v) for v in anchor]
            if m.get("markerKind", 0) != 1:
                rv["markerCenter"] = (list(m["markerCenter"]) if m.get("markerKind", 0) == 0 and m.get("markerCenter")
                                      else [float(v) for v in anchor])
            m.update(rv)
            m["markerRadius"] = r
            m["markerSurface"] = info["surface"]
            m["description"] = describe(base, info, side)
            m["source"] = SOURCE
            if side == "l" and not deep:
                m.pop("mirroredFrom", None)
                m["independentlyFitted"] = True
            cs = [curve_stats(Q, c) for Q, c in lines]
            summary["records"].append({
                "id": m["id"], "verticesBefore": old_counts[m["id"]], "vertices": m["count"], "surface": info["surface"],
                **{k: v for k, v in info.items() if k.endswith("Mm") or k == "projectionMisses"},
                "maxTurnDegPerMm": round(max(c["maxTurnDegPerMm"] for c in cs), 2),
                "revealMonotone": None if mono is None else round(mono, 3), "markerAxis": m.get("markerAxis")})
    for m in marks:
        if not m.get("instrument"):
            continue
        anchor = list(m["anchor"])
        pos, nrm = build_tool(m)
        encode(m, pos, nrm)
        m["anchor"] = anchor
        m["markerRadius"] = TOOL_SHAFT
        m["source"] = TOOL_SOURCE
        summary["records"].append({"id": m["id"], "verticesBefore": old_counts[m["id"]], "vertices": m["count"]})
    # the incision marking lies inside the prepared field on both sides
    for side in ("r", "l"):
        S = Side(records, side)
        fm = field_margin(S, field_design(S, right), S.incision_mark())
        summary["field_" + side] = fm
        if fm["fieldMarginMinMm"] < 1.0 or (fm["fieldMarginAwayFromEdgeMinMm"] or 99) < FIELD_MIN_MARGIN / MM:
            raise RuntimeError(f"incision marking not inside the prepared field ({side}): {fm}")
    summary["verticesBefore"] = int(sum(old_counts.values()))
    summary["verticesAfter"] = int(sum(m["count"] for m in marks))
    return summary


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit("usage: python3 tools/anatomy_build/markers.py IN.json OUT.json")
    records = load_records(argv[1])
    summary = apply(records)
    save_records(records, argv[2])
    import json
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))
    for r in summary["records"]:
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv)
