"""BRAIN build step: smoother cortical parcels and a continuous cortex in the bone window.

Usage::

    python3 tools/anatomy_build/brain.py IN.json OUT.json            # build
    python3 tools/anatomy_build/brain.py IN.json --audit [right|left] # re-derive the visible set

Two problems in the base atlas records are addressed.

1. **Coarse parcels.** The Z-Anatomy cortical parcels are closed low-poly shells
   (for example the temporal pole has 274 vertices and a mean edge of about 4 mm).
   The Z-Anatomy ``Startup.blend`` was inspected with ``blender_export.py``: the
   cortical parcels carry *no* Subdivision Surface (or other) modifiers, so the
   coarse geometry is the source geometry and re-exporting with modifiers evaluated
   would not change it (only a few tiny brainstem nuclei carry Subdivision
   Surface). The parcels are therefore refined offline with **PN triangles**
   (curved point-normal triangles, Vlachos et al. 2001). PN triangles are an
   *interpolating* scheme: every original vertex keeps its exact position, and each
   new edge point depends only on the two end vertices and their normals, so the
   refined shell stays watertight and its silhouette does not shrink away from
   neighbouring parcels. Normals are recomputed from the refined surface. The
   refinement rounds the facets between source vertices; it adds no anatomical
   detail that is not in the source.

2. **Voids in the bone window.** Only six cortical parcels per side were
   "relevant" (shown outside Full Head Context), so the window showed floating
   chunks of cortex on a black background. Rays cast from the guide's operative
   cameras (lateral, basal and narrow-opening entries, plus the step 13/17/18
   default cameras) through the window ellipse were used to list every brain
   record that is first-hit (``--audit`` reproduces this). Those parcels lose
   ``contextOnly`` (``WINDOW_VISIBLE``), except the orbital gyri, the lateral
   orbital sulcus and the temporal pole: they lie in front of the internal carotid
   artery, M1, A1 and the optic nerve from the operative cameras, and brain
   retraction is not modelled, so showing them would close the opticocarotid
   corridor that steps 18-20 point at. They stay contextOnly, as in the base.
   The remaining seams between atlas parcels (about 1.5 % of window rays, mostly
   along the Sylvian line) look straight through the brain. They are closed with
   an illustrative *sulcal depth backing*: the Z-Anatomy white-matter surface of
   the same side, cropped to the lateral frontotemporal region, given a darker
   tone and labelled as illustration. Backing triangles that lie between an
   operative camera and a relevant artery or nerve of that side are removed (plus
   two rings of neighbours), so the backing never hides a labelled target.

The step is deterministic and needs only numpy.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (decode, encode, find, load_records, normalize, save_records,  # noqa: E402
                    vertex_normals, weld)

# Parcels first-hit by window rays from the operative cameras (see ``audit``). Both sides.
WINDOW_VISIBLE = [
    "Insula (Subcentral gyrus and ant. and post. sulci*)",
    "Middle temporal gyrus",
    "Orbital part of  inferior frontal gyrus",
    "Opercular part of inferior frontal gyrus",
    "Superior temporal gyrus (Lateral part)",
    "Triangular part of inferior frontal gyrus",
    # previously contextOnly; made relevant here:
    "Precentral gyrus",
    "Middle frontal gyrus",
    "Inferior frontal sulcus",
    "Postcentral gyrus",
    "Precentral sulcus (inferior part)*",
    "Central sulcus",
    # also first-hit, but they sit in front of the internal carotid artery, M1, A1 and optic nerve
    # from the operative cameras. Brain retraction is not modelled, so they stay contextOnly to
    # keep the opticocarotid corridor of steps 18-20 open (refined like the window parcels):
    "Orbital gyri",
    "Orbital sulci (Lateral Orbital sulcus*)",
    "Temporal pole",
]
KEEP_CONTEXT_ONLY = {"Orbital gyri", "Orbital sulci (Lateral Orbital sulcus*)", "Temporal pole"}
# Cortex seen in the basal frontal / temporal and Sylvian views (refined, contextOnly unchanged).
BASAL_SYLVIAN = [
    "Straight gyrus (Gyrus rectus)",
    "Olfactory sulcus",
    "Orbital sulci (H-shaped orbital sulci*)",
    "Orbital gyri (Frontomarginal gyrus and sulcus*)",
    "Transverse frontopolar gyrus and sulcus*",
    "Inferior temporal gyrus",
    "Superior temporal sulcus",
    "Inferior temporal sulcus",
    "Lateral occipitotemporal gyrus",
    "Medial occipitotemporal gyrus (Parahippocampal*)",
    "Temporal plane",
    "Transverse temporal gyri",
    "Lat_Fis-ant-Horizont",
    "Lat_Fis-ant-Vertical",
    "Lat_Fis-post",
    "Circular sulcus of insula",
]
# PN tessellation factor: each triangle -> LEVEL**2. Factor 2 halves the mean edge (~3.4 -> ~1.7 mm);
# factor 3 looked only marginally smoother at operative zoom and cost 2.7x the data, so 2 is used.
# Posterior and medial cortex (only seen in Full Head Context) is left as in the source.
LEVEL = 2

# Everything in the brain group that is not cortex (left untouched).
NOT_CORTEX = re.compile(
    r"nucle|quadrangular|biventral|gracile|semilunar|central lobule|vermis|culmen|declive|flocculus|tonsil|lingula|nodule|pons|midbrain|medulla|olive|"
    r"pyramid|peduncle|colliculus|geniculate|ventricle|aqueduct|commissure|corpus callosum|fornix|septum|"
    r"septal|stria|thalamus|habenula|putamen|globus|caudate|lentiform|white matter|optic tract|mamillary|"
    r"interpeduncular|amygdal|\bhippocamp", re.I)

# Sulcal depth backing: white-matter crop box for the right side (display units; mirrored for left).
BACKING_BOX = dict(x_max=-0.9, y=(-1.8, 2.0), z=(-0.8, 3.4))
BACKING_COLOR = "#8c6c64"


# ------------------------------------------------------------ PN triangles
def _tess_pattern(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric sample points (u, v) and sub-triangles for tessellation factor L."""
    pts, index = [], {}
    for i in range(L + 1):
        for j in range(L + 1 - i):
            index[i, j] = len(pts)
            pts.append((i / L, j / L))
    tris = []
    for i in range(L):
        for j in range(L - i):
            tris.append((index[i, j], index[i + 1, j], index[i, j + 1]))
            if i + j < L - 1:
                tris.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    return np.array(pts), np.array(tris)


def pn_refine(pos: np.ndarray, nrm: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Refine unindexed triangles with PN (curved point-normal) triangles.

    ``pos``/``nrm`` are per-corner arrays (3 rows per triangle). Original corners are
    reproduced exactly; edge points depend only on the edge's end vertices and normals.
    Returns unindexed (positions, normals) with smooth normals recomputed from the
    refined surface.
    """
    P = pos.reshape(-1, 3, 3)
    N = normalize(nrm.reshape(-1, 3, 3))
    P1, P2, P3 = P[:, 0], P[:, 1], P[:, 2]
    N1, N2, N3 = N[:, 0], N[:, 1], N[:, 2]

    def w(a, b, n):
        return np.einsum("ij,ij->i", b - a, n)[:, None]

    b210 = (2 * P1 + P2 - w(P1, P2, N1) * N1) / 3
    b120 = (2 * P2 + P1 - w(P2, P1, N2) * N2) / 3
    b021 = (2 * P2 + P3 - w(P2, P3, N2) * N2) / 3
    b012 = (2 * P3 + P2 - w(P3, P2, N3) * N3) / 3
    b102 = (2 * P3 + P1 - w(P3, P1, N3) * N3) / 3
    b201 = (2 * P1 + P3 - w(P1, P3, N1) * N1) / 3
    E = (b210 + b120 + b021 + b012 + b102 + b201) / 6
    Vc = (P1 + P2 + P3) / 3
    b111 = E + (E - Vc) / 2

    uv, sub = _tess_pattern(L)
    u = uv[:, 0][None, :, None]   # weight of P2
    v = uv[:, 1][None, :, None]   # weight of P3
    t = 1 - u - v                 # weight of P1
    e = lambda a: a[:, None, :]   # noqa: E731
    S = (e(P1) * t ** 3 + e(P2) * u ** 3 + e(P3) * v ** 3
         + e(b210) * 3 * t * t * u + e(b120) * 3 * t * u * u + e(b201) * 3 * t * t * v
         + e(b021) * 3 * u * u * v + e(b102) * 3 * t * v * v + e(b012) * 3 * u * v * v
         + e(b111) * 6 * t * u * v)
    # exact corners (guards against float drift so shared vertices weld)
    corner = {0: (0, 0), 1: (L, 0), 2: (0, L)}
    for k, (i, j) in corner.items():
        S[:, int(np.where((np.round(uv[:, 0] * L) == i) & (np.round(uv[:, 1] * L) == j))[0][0])] = P[:, k]
    # linearly interpolated normals, used only to keep the outward orientation
    Nl = normalize(e(N1) * t + e(N2) * u + e(N3) * v)

    F = (sub[None] + (np.arange(len(P)) * len(uv))[:, None, None]).reshape(-1, 3)
    Vs = S.reshape(-1, 3)
    Ns = Nl.reshape(-1, 3)
    Vw, Fw = weld_ids(Vs, F)
    nrm_new = vertex_normals(Vw, Fw)
    out_p = Vw[Fw.reshape(-1)]
    out_n = nrm_new[Fw.reshape(-1)]
    agree = np.einsum("ij,ij->i", out_n, Ns[F.reshape(-1)])
    if agree.mean() < 0.8:
        raise RuntimeError(f"PN refinement flipped orientation (mean agreement {agree.mean():.3f})")
    # where the recomputed normal disagrees badly with the interpolated source normal
    # (near-degenerate slivers on thin sulcal rims), fall back to the source normal
    bad = agree < 0.0
    out_n[bad] = Ns[F.reshape(-1)][bad]
    return out_p, normalize(out_n)


def weld_ids(V: np.ndarray, F: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Weld coincident points of an indexed mesh (returns compacted V and remapped F)."""
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.reshape(-1)[F]


# ------------------------------------------------------------ backing
OP_CAMERAS = [  # (entry, focus, distance) of the guide's operative view (right side; app.js setOperativeView)
    ([-3.05, .30, 1.25], [-1.55, -.3, .60], 10.5),
    ([-3.05, .05, 1.85], [-.85, -1.05, 1.35], 9.0),
    ([-2.9, -.62, 1.65], [-.85, -1.05, 1.35], 9.0),
]


def op_eyes(side: str) -> list[np.ndarray]:
    s = np.array([1.0 if side == "right" else -1.0, 1, 1])
    out = []
    for entry, focus, dist in OP_CAMERAS:
        e, f = np.array(entry), np.array(focus)
        out.append((e + dist * normalize(e - f)) * s)
    return out


STEP_CAMERAS = [  # (yaw, pitch, distance, target) of the default step-13, step-17 and step-18 views
    (1.0, .28, 14.0, [-1, 0, .7]),
    (1.0, .28, 11.0, [-1, 0, .7]),
    (1.0, .40, 9.0, [-1.9, -.5, 1]),
]


def step_eyes(side: str) -> list[np.ndarray]:
    s = np.array([1.0 if side == "right" else -1.0, 1, 1])
    out = []
    for yaw, pitch, dist, target in STEP_CAMERAS:
        c, sn, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        ry = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]]).T   # same matrices as the vertex shader
        rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]).T
        out.append((np.array(target, float) + (rx @ ry).T @ np.array([0, 0, dist])) * s)
    return out


def exposed_through_window(V: np.ndarray, F: np.ndarray, occluders: np.ndarray, side: str) -> np.ndarray:
    """Triangles of (V, F) seen through the bone window from any guide camera, not hidden by ``occluders``.

    ``occluders`` is an unindexed (3k, 3) triangle array of the cortex that is shown.
    """
    from common import ray_nearest
    s = 1.0 if side == "right" else -1.0
    Ov, Of = weld(occluders, 1e-6)
    c = V[F].mean(1)
    seen = np.zeros(len(F), bool)
    for eye in op_eyes(side) + step_eyes(side):
        d = c - eye
        L = np.linalg.norm(d, axis=1)
        d = d / L[:, None]
        # where the sight line crosses the window plane x = -2.3 (mirrored on the left)
        k = (-2.3 * s - eye[0]) / np.where(np.abs(d[:, 0]) > 1e-9, d[:, 0], 1e-9)
        q = eye + k[:, None] * d
        inside = ((q[:, 1] - .3) / 1.55) ** 2 + ((q[:, 2] - 1.25) / 1.65) ** 2 < 1.1 ** 2
        test = np.where(inside & (k > 0) & (k < L) & ~seen)[0]
        if not len(test):
            continue
        t, tri = ray_nearest(eye, d[test], Ov, Of, chunk=64)
        seen[test[(tri < 0) | (t > L[test] - 0.02)]] = True
    return seen


def corridor_occluders(V: np.ndarray, F: np.ndarray, targets: np.ndarray, side: str,
                       samples: int = 1500, rounds: int = 8) -> np.ndarray:
    """Boolean mask of triangles that lie between an operative camera and a target point."""
    from common import ray_nearest
    rng = np.random.default_rng(7)
    pts = targets[rng.choice(len(targets), min(samples, len(targets)), replace=False)]
    removed = np.zeros(len(F), bool)
    for eye in op_eyes(side):
        d = pts - eye
        L = np.linalg.norm(d, axis=1)
        d = d / L[:, None]
        active = np.ones(len(pts), bool)
        for _ in range(rounds):
            idx = np.where(~removed)[0]
            t, tri = ray_nearest(eye, d[active], V, F[idx], chunk=32)
            hit = (tri >= 0) & (t < L[active] - 0.01)
            if not hit.any():
                break
            removed[idx[tri[hit]]] = True
            sub = np.where(active)[0]
            active[sub[~hit]] = False
    return removed


def grow(F: np.ndarray, mask: np.ndarray, rings: int) -> np.ndarray:
    """Grow a triangle mask by vertex-adjacency rings."""
    for _ in range(rings):
        verts = np.zeros(F.max() + 1, bool)
        verts[F[mask].reshape(-1)] = True
        mask = mask | verts[F].any(1)
    return mask


def backing_record(wm: dict, side: str, targets: np.ndarray, cortex: np.ndarray) -> dict:
    s = 1.0 if side == "right" else -1.0
    pos, _ = decode(wm)
    T = pos.reshape(-1, 3, 3)
    c = T.mean(1)
    b = BACKING_BOX
    keep = ((c[:, 0] * s < b["x_max"]) & (c[:, 1] > b["y"][0]) & (c[:, 1] < b["y"][1])
            & (c[:, 2] > b["z"][0]) & (c[:, 2] < b["z"][1]))
    p = T[keep].reshape(-1, 3)
    V, F = weld(p, 1e-6)
    n = vertex_normals(V, F)
    # only where it shows through seams of the displayed cortex (plus ~4 mm margin)
    F = F[grow(F, exposed_through_window(V, F, cortex, side), 4)]
    # keep the opticocarotid / Sylvian corridor open: drop backing in front of target vessels/nerves
    if len(targets):
        F = F[~grow(F, corridor_occluders(V, F, targets, side), 2)]
    side_name = "Right" if side == "right" else "Left"
    rid = "ILL-sulcal-backing" if side == "right" else "ILL-sulcal-backing-l"
    m = {
        "name": f"{side_name} sulcal depth backing — illustration",
        "nativeName": f"{side_name} sulcal depth backing — illustration.{side[0]}",
        "id": rid,
        "fma": "Illustration",
        "procedureSide": side,
        "group": "brain",
        "color": BACKING_COLOR,
        "source": f"Derived from the Z-Anatomy {side} white-matter (telencephalon) surface, Startup.blend; CC BY-SA 4.0",
        "description": ("Illustrative backing that closes the seams between atlas cortical parcels inside the "
                        "pterional exposure, so the background does not show through. Geometry is the native "
                        f"{side} white-matter surface, kept only where it shows through those seams from the "
                        "guide's operative and step cameras, and shown in a darker sulcal tone. It is not a "
                        "separate structure, and sulcal depths are not measured."),
        "illustrative": True,
    }
    if side == "left":
        m["baseId"] = "ILL-sulcal-backing"
        m["independentlyFitted"] = True
    encode(m, V[F.reshape(-1)], n[F.reshape(-1)])
    return m


# ------------------------------------------------------------ apply
REFINE_NOTE = (" Cortical facets refined offline with interpolating PN triangles (factor {L}): source vertices "
               "are kept exactly and only the surface between them is rounded; no anatomical detail is added.")


def is_cortex(m: dict) -> bool:
    return m["group"] == "brain" and not NOT_CORTEX.search(m["nativeName"]) and not m.get("illustrative")


def apply(records: list[dict]) -> dict:
    t0 = time.time()
    visible = {f"{n}.{s}" for n in WINDOW_VISIBLE for s in "lr"}
    smooth = visible | {f"{n}.{s}" for n in BASAL_SYLVIAN for s in "lr"}
    summary = {"refined": {}, "uncontext": [], "added": [], "triangles_before": 0, "triangles_after": 0}
    shown_cortex = {"r": [], "l": []}   # unrefined geometry of the cortex shown in the window
    for m in records:
        if not is_cortex(m) or m["nativeName"] not in smooth:
            continue
        pos, nrm = decode(m)
        if m["nativeName"] in visible and m["nativeName"][:-2] not in KEEP_CONTEXT_ONLY:
            shown_cortex[m["nativeName"][-1]].append(pos)
            if m.pop("contextOnly", None):
                summary["uncontext"].append(m["id"])
                m["description"] = m.get("description", "").replace(
                    "Available head context; not part of the core pterional structure subset.",
                    "Shown in the pterional exposure: cortex seen through the bone window from the operative views.")
        p2, n2 = pn_refine(pos, nrm, LEVEL)
        summary["triangles_before"] += len(pos) // 3
        summary["triangles_after"] += len(p2) // 3
        encode(m, p2, n2)
        if "PN triangles" not in m.get("description", ""):
            m["description"] = m.get("description", "") + REFINE_NOTE.format(L=LEVEL)
        summary["refined"][m["id"]] = m["nativeName"]
    for side, tag in (("right", "r"), ("left", "l")):
        rid = "ILL-sulcal-backing" + ("" if side == "right" else "-l")
        records[:] = [m for m in records if m["id"] != rid]
        targets = np.concatenate([decode(m)[0] for m in records if m["group"] in ("arteries", "nerves")
                                  and not m.get("contextOnly") and not m.get("instrument")
                                  and not m.get("marker") and m["nativeName"].endswith("." + tag)])
        b = backing_record(find(records, native=f"White matter of telencephalon.{tag}"), side, targets,
                           np.concatenate(shown_cortex[tag]))
        records.append(b)
        summary["added"].append({"id": rid, "triangles": b["count"] // 3})
    added = sum(a["triangles"] for a in summary["added"])
    summary["triangles_added_total"] = summary["triangles_after"] - summary["triangles_before"] + added
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


# ------------------------------------------------------------ audit (optional, slow)
def audit(records: list[dict], side: str = "right", rays: int = 3000) -> dict:
    """List brain records first-hit by rays through the window ellipse from the guide cameras."""
    from collections import Counter
    from common import ray_nearest
    sgn = 1 if side == "right" else -1
    other = ".l" if side == "right" else ".r"
    B = [m for m in records if m["group"] == "brain" and not m["nativeName"].endswith(other)
         and not m.get("illustrative")]
    Vs, Fs, own, off = [], [], [], 0
    for i, m in enumerate(B):
        p, _ = decode(m)
        p[:, 0] *= sgn
        Vs.append(p)
        Fs.append(np.arange(len(p)).reshape(-1, 3) + off)
        own += [i] * (len(p) // 3)
        off += len(p)
    V, F, own = np.concatenate(Vs), np.concatenate(Fs), np.array(own)
    T = V[F]
    k = (T[:, :, 0] < -0.2).any(1) & (T[:, :, 1] < 1.8).any(1) & (T[:, :, 2] > -0.6).any(1)
    F, own = F[k], own[k]

    def op(entry, focus, dist):
        entry, focus = np.array(entry), np.array(focus)
        return entry + dist * normalize(entry - focus)

    def cam(yaw, pitch, dist, target):
        c, s, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        ry = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T
        rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]).T
        return np.array(target) + (rx @ ry).T @ np.array([0, 0, dist])

    eyes = {"op-lateral": op([-3.05, .30, 1.25], [-1.55, -.3, .60], 10.5),
            "op-basal": op([-3.05, .05, 1.85], [-.85, -1.05, 1.35], 9),
            "op-narrow": op([-2.9, -.62, 1.65], [-.85, -1.05, 1.35], 9),
            "step13": cam(1.0, .28, 14, [-1, 0, .7]), "step17": cam(1.0, .28, 11, [-1, 0, .7]),
            "step18": cam(1.0, .4, 9, [-1.9, -.5, 1])}
    rng = np.random.default_rng(0)
    total = Counter()
    for name, eye in eyes.items():
        th = rng.uniform(0, 2 * np.pi, rays)
        r = np.sqrt(rng.uniform(0, 1, rays)) * 1.05
        tgt = np.stack([np.full(rays, -2.3), .3 + 1.55 * r * np.cos(th), 1.25 + 1.65 * r * np.sin(th)], 1)
        _, tri = ray_nearest(eye, normalize(tgt - eye), V, F, chunk=16)
        total.update(B[own[t]]["nativeName"] for t in tri[tri >= 0])
        print(f"{name}: miss {np.mean(tri < 0):.3f}", file=sys.stderr)
    return dict(total.most_common())


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[1] == "--audit":
        side = argv[2] if len(argv) > 2 else "right"
        print(json.dumps(audit(load_records(argv[0]), side), indent=1))
        return
    if len(argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    records = load_records(argv[0])
    summary = apply(records)
    save_records(records, argv[1])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
