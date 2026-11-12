#!/usr/bin/env python3
"""Rebuild the shipped anatomy chunks from the original atlas import.

The browser loads ``assets/anatomy-0N.bin`` (one gzip JSON list of mesh
records split over six files) and ``data/manifest.json``. This script
regenerates both from the records of the original import commit, so the
derived geometry is reproducible::

    python3 tools/build_anatomy.py                 # build into the repo
    python3 tools/build_anatomy.py --check         # build and validate, write nothing
    python3 tools/build_anatomy.py --base base.json --save-records /tmp/final.json

Base records are read with ``git show 7961958:assets/anatomy-0N.bin`` (the
six chunks concatenated and gunzipped) or from ``--base`` (a JSON file as
written by ``tools/anatomy.py unpack``, or a gzip of one). The script refuses
to run on already-derived data: the base must match the known hash of the
import, unless ``--allow-other-base`` is given, and even then it must not
contain records or normal lengths written by a build step.

Steps (each module in ``tools/anatomy_build/`` exposes ``apply(records)``):

1. ``brain``            PN-triangle refinement of cortical parcels, window
                        parcels made relevant, illustrative sulcal backing.
2. ``dura``             dura sheets fitted under the inner table (reads brain).
3. ``temporal_layers``  temporal fascia laminae and fat pads over temporalis.
4. ``veins``            illustrative superficial veins and dural sinuses
                        (reads brain, bone, arteries).
5. ``arachnoid``        translucent arachnoid sheet (reads brain, veins).
6. ``scalp``            illustrative scalp (skin with subcutaneous tissue),
                        galea and pericranium over the vault, split into the
                        pterional scalp flap, pericranial flap, stationary
                        operative region and context-only vault; flap records
                        carry hinge fields (``flap``) and the scalp flap the
                        incision course (``incisionLine``). Reads bone,
                        temporalis, temporal layers and the incision marking.
7. ``markers``          clean schematic markings and instrument arrows (smooth
                        constant-radius tubes draped on the surface they
                        annotate); reads scalp, dura and temporal layers.
8. ``ao``               baked ambient occlusion in the normal length; also
                        flips inward-facing closed records outward. Last.

Then the records are validated, ``data/manifest.json`` generated fields are
rewritten (``structures``, ``structure_count``, ``vertex_count``,
``context_expansion`` counts, ``markings.items``, ``build`` and the prose
fields listed in ``PROSE``; every other field is kept) and the chunks are
packed. Requires numpy and scipy; runs in about 6 minutes on 16 cores. The default base
needs a full git clone (commit 7961958 must be present); otherwise pass ``--base``.

``study_plates`` (the two fallback images shown only without WebGL) is a hand-maintained
manifest field: the build keeps whatever the current manifest holds and does not render it.
``variants`` (a static snapshot of the guide steps for the 12 configurations) is regenerated
from the renderer with ``JSON.stringify(guideQA.variantSnapshot())`` in a browser after
``buildSteps`` changes; the build keeps it and only re-syncs legacy cranial group lists.
The manifest ``build`` record holds no timings, so rebuilding unchanged inputs rewrites the
manifest byte for byte; per-step timings are printed to stderr and saved with ``--summary``.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO / "tools" / "anatomy_build"
sys.path.insert(0, str(BUILD_DIR))

import common  # noqa: E402

BASE_COMMIT = "7961958"
# sha256 of the gunzipped base JSON at BASE_COMMIT (identical to the compact
# json.dumps of the records, so --base files written by `anatomy.py unpack`
# are checked the same way).
BASE_SHA256 = "a576e8f35111cb9feba9cf1e5d2a8e2492823ac099d6b6ac766afd65587ad6fa"
BASE_RECORDS = 614
STEPS = ["brain", "dura", "temporal_layers", "veins", "arachnoid", "scalp", "markers", "ao"]
DERIVED_ID_PREFIXES = ("ILL-vein-", "ILL-arachnoid", "ILL-sulcal-backing", "ILL-scalp", "ILL-galea",
                       "ILL-pericranium")
UNIT_NORMAL_GROUPS = {"annotations", "instruments"}
KNOWN_GROUPS = {"skin", "galea", "pericranium", "bone", "brain", "muscle", "fascia", "deepfascia", "fat",
                "arteries", "veins", "nerves", "dura", "arachnoid", "annotations", "instruments"}
FLAP_KEYS = {"scalp", "pericranium"}
SHELL_CENTRE = np.array([0.0, -0.4, -0.4])   # scalp shell centre (tools/anatomy_build/scalp.py C)
AO_MIN = 0.3
GZIP_BUDGET_MB = 58.0

STEP_INFO = {
    "brain": "Cortical parcels seen through the bone window and along the basal frontal, temporal and Sylvian "
             "surfaces are refined with interpolating PN triangles (each triangle -> 4; source vertices unchanged; "
             "new surface within about 1 mm of the source facets). Six window parcels per side lose contextOnly. "
             "An illustrative sulcal depth backing (native white-matter surface, cropped) closes seams between "
             "parcels.",
    "dura": "Illustrative dura sheets fitted independently per side between the native inner table and cortex, "
            "covering the bone window with a margin and the lateral sphenoid ridge; the sheet enters the bone slab "
            "only inside the window cylinder, so it never shows over the cut bone edge.",
    "temporal_layers": "Illustrative deep and superficial temporal fascial laminae and three fat pads, built as "
                       "offset surfaces over the native temporalis and skull; left side fitted on native left "
                       "anatomy.",
    "veins": "Illustrative superficial middle cerebral vein system, sphenoparietal and cavernous sinuses, "
             "superficial temporal vein and variant Trolard/Labbe anastomotic veins, with courses derived from "
             "native cortex, bone and arteries; the cavernous sinus is placed from the native oculomotor and "
             "abducens nerves (parasellar landmarks).",
    "arachnoid": "Illustrative translucent arachnoid sheet over the window region and Sylvian fissure, with a "
                 "schematic fissure line used by the renderer to open it; left out over the hidden orbital gyri "
                 "and temporal pole.",
    "scalp": "Illustrative scalp over the vault from the brow to the occiput and down to the zygomatic arch: skin "
             "and subcutaneous tissue as one closed slab, galea aponeurotica continuous with the temporoparietal "
             "fascia, an empty loose areolar gap and pericranium on the outer table (absent over the temporal "
             "fascia), built as offset shells over the native skull and temporalis and the illustrative temporal "
             "layers. Each side is split into the pterional scalp flap anterior to a curvilinear incision (upper "
             "course from the atlas schematic marking, lower course redrawn from the zygomatic arch behind the bone "
             "window), an anteriorly based pericranial flap (kept outside the bone window), the stationary operative "
             "region and the context-only vault; flap records carry hinge fields for the renderer. The atlas "
             "superficial temporal artery and frontal branch and the illustrative superficial temporal vein are "
             "re-seated radially into the subcutaneous layer where the scalp covers them.",
    "markers": "Schematic markings and instrument arrows redrawn as clean strokes: smooth centrelines resampled "
               "evenly by arc length and swept as constant-radius tubes (parallel-transport frames, smooth normals, "
               "round caps, cone arrowheads), with at least 0.1 mm of clearance above the surface they annotate "
               "(illustrative scalp skin or galea, the bed under the scalp flap, pericranium, bone or dura) "
               "enforced against the true mesh; each record's description gives its measured clearance range. "
               "Outlines of the renderer's bone window, burr holes and dural openings follow the "
               "shader's analytic cuts exactly; the incision and the galeal and skin closure stitches follow the "
               "illustrative scalp incision; the prepared-field outline encloses the incision marking; deep markings (Sylvian trace, clinoid, optico-carotid region) are smoothed in place. Left "
               "markings are fitted on native left anatomy. Reveal fields are recomputed from the new geometry.",
    "ao": "Per-vertex ambient occlusion baked into normal length (range 0.3-1) by voxel ray marching; closed "
          "records whose normals pointed inward are flipped outward first. Sheets the renderer moves (dura, "
          "fascial layers, fat pads, arachnoid, scalp skin, galea, pericranium) occlude only themselves.",
}

# Manifest prose that describes the geometry. Kept here so a rebuild from the
# original manifest gives the same text.
PROSE = {
    "revision": "Lifelike rendering: smoothed window cortex, conforming illustrative dura and temporal layers, "
                "illustrative veins and arachnoid, illustrative scalp, galea and pericranium with a reflectable "
                "pterional flap, schematic markings redrawn as clean strokes, baked ambient occlusion",
    "extraction": "Native meshes triangulated as fans; normals recomputed; negative-determinant winding corrected. "
                  "Vessel Bézier curves sampled with 12-sided tubes. Blender modifiers not evaluated; the cortical "
                  "parcels carry no modifiers in the source scene (only small brainstem nuclei carry Subdivision "
                  "Surface, midline structures Mirror and the corpus callosum Decimate). That was checked with "
                  "Blender 5.0.1 on a downloaded Startup.blend whose SHA-256 (9de642d9...) differs from "
                  "source_sha256; spot checks matched the records to 3e-6 display units (see "
                  "build.sourceInspection). Derived geometry is "
                  "rebuilt by tools/build_anatomy.py (see build): window and basal cortical parcels are refined "
                  "with interpolating PN triangles, which round the facets between the source vertices without "
                  "moving them and add no anatomical detail. Vertex normals point outward and their length "
                  "stores baked ambient occlusion (0.3-1; markings and instruments keep unit normals).",
    "limitations": [
        "Regional reference anatomy, not a patient model or independently validated operative simulation.",
        "Clinoid removal and bone opening are approximate illustrative spatial masks, not validated osteotomies. "
        "ACP is part of sphenoid; no marker object relabeled as tissue.",
        "Combinations are reading configurations, not clinically validated combinations. Intradural source "
        "describes wide fissure dissection; selecting No displays an evidence conflict.",
        "Native superficial fascia is retained as atlas context but superseded in the study view by illustrative "
        "laminae. The deep and superficial temporal laminae and all three fat pads are original illustrations "
        "built as offset surfaces over the native temporalis and skull; their thickness, extent, split height and "
        "attachments are illustrative values, not atlas segmentations.",
        "Separated-plane view exaggerates spacing to reveal layer order; it is not an operative tissue position.",
        "Six supplied PDFs reviewed; exact subject-specific 3D geometry cannot be recovered from 2D images.",
        "Song muscle-splitting branch depicts intact atlas muscle and describes its en-bloc exposure; an exact "
        "fiber split is not inferred.",
        "Cortex keeps its native atlas positions, except about 15 vertices per side at the bone-window rim "
        "moved inward by at most 2.8 mm so they stay under the illustrative dura; brain retraction and cortical "
        "deformation are not modeled. "
        "Window parcels are smoothed with PN triangles, which remove faceting but keep the coarse atlas gyral "
        "form; sulci read as seams between separate parcels. An illustrative backing (native white-matter "
        "surface) fills those seams. The orbital gyri, lateral orbital sulcus and temporal pole stay context-only "
        "so they do not cover the optico-carotid corridor; some dark gaps remain deep in the corridor.",
        "The Sylvian arachnoid and superficial veins are original illustrations: the arachnoid is a translucent "
        "sheet over the atlas cortex, drawn milkier over the fissure, that opens along a schematic fissure line. "
        "It is left out over the context-only orbital gyri and temporal pole. Opercular separation, the depth "
        "of the split and cisternal arachnoid are not reconstructed.",
        "Venous courses (superficial middle cerebral vein with tributaries, sphenoparietal and cavernous sinuses, "
        "superficial temporal vein, Trolard and Labbé veins) are derived from atlas cortex, bone and arteries, "
        "not segmented; one typical drainage pattern is shown and calibres are approximate. The cavernous sinus "
        "is placed from parasellar landmarks (native oculomotor and abducens nerves), not fitted to the atlas "
        "carotid, whose horizontal cavernous segment lies lower than typical; only the upper part of the "
        "carotid's ascending genu lies inside it, and the dural layers of its wall are not modeled.",
        "Local dura is a sheet fitted under the native inner table, covering the bone window with a margin and "
        "the lateral sphenoid ridge; its extent and single-surface thickness are illustrative. Where the atlas "
        "cortex crosses the atlas inner table the sheet runs inside the bone slab within the window; at the "
        "window rim it stays under the inner table, so atlas cortex can show through it in a few small spots "
        "at the cut edge.",
        "The scalp (skin with subcutaneous tissue, galea/temporoparietal fascia, pericranium) is an original "
        "illustration built as offset shells over the atlas skull, temporalis and illustrative temporal layers; "
        "thicknesses (about 7 mm in total over the vault), extent and cut lines are illustrative. The incision "
        "follows the upper part of the atlas schematic incision marking; its lower part is redrawn to begin on the "
        "zygomatic arch about 1 cm in front of the estimated tragus and to pass behind the guide's bone window. "
        "Hair, the auricle and the face are not modeled. Where the scalp covers them, the atlas superficial "
        "temporal artery and frontal branch and the illustrative superficial temporal vein are moved radially into "
        "the subcutaneous layer (their atlas course lies deeper, at the bone / temporal-fascia level); below the "
        "zygomatic arch they keep the atlas course. Atlas context "
        "sheets that coincide with the scalp (epicranial aponeurosis, frontalis, occipitalis, temporoparietalis, "
        "atlas superficial temporal fascia) are hidden while the scalp is drawn.",
        "Soft-tissue motion is schematic: in the renderer the scalp, pericranial, temporal fascial, temporalis and "
        "dural flaps peel from their free edge toward a straight illustrative hinge and roll over it, with the "
        "hinge height taken from the data; at the end of the fold the reflected scalp, galeal, fascial and "
        "pericranial flaps are laid toward the face about 9 mm above the facial skeleton or stationary scalp (a "
        "stand-in for the rolled sponge and the facial soft tissue the model lacks), and the dural leaf rests on "
        "the bone beyond the window rim. The incision edges part slightly more than in life for visibility, more "
        "so while the galea is closed. These are not soft-tissue simulations: the reflected temporalis can touch "
        "the rolled scalp near the lateral orbital rim, and hooks, sponges and drapes are not modeled.",
        "Patient positioning turns the whole atlas head rigidly about an estimated craniocervical pivot (between "
        "the occipital condyles). The default (about 20 degrees contralateral rotation and 20 degrees extension) "
        "follows the example in Rodriguez Rubio et al. and is target-dependent; the neck, body, table tilt and "
        "venous-drainage elevation are not modeled, and the head holder, its pin sites, the table edge and the "
        "orientation gizmo are schematic drawings, not measured equipment positions.",
        "Schematic markings are drawn as smooth tubes about 0.7 mm in radius at least 0.1 mm above the surface "
        "they annotate (skin markings typically 0.1-0.3 mm; outlines projected onto bone and dura ride an envelope "
        "typically 0.4-0.6 mm above it, up to about 1.8 mm at dash ends); they mark regions, trajectories and "
        "sequences for teaching, not measured landmarks. The prepared-field outline was widened to enclose the "
        "illustrative incision marking (at least 8 mm away from the scalp's free edge; at the zygomatic arch it "
        "runs along that edge and the incision marking stops 5 mm above it), and stitch, tack-up and fixation "
        "marks are evenly spaced or radial by design; galeal stitches are drawn on the galea, skin stitches on "
        "the skin. Deep markings are smoothed only and can pass behind vessels or partly into bone.",
        "Ambient occlusion is precomputed per vertex in the rest pose; reflected flaps keep rest-pose shading, and "
        "the scalp layers, dura and fascial sheets occlude only themselves. "
        "Surgical-palette surface texture is procedural decoration, not anatomy: it draws no dural or pial "
        "vessels, and the layered tones of the cut bone edge are estimated in the renderer, not modeled.",
        "Instrument directions illustrate a qualitative surface entry only. They are not exact surgical "
        "placements, collision-checked paths, measured angles or patient-specific trajectories.",
        "Right, Left and Both contexts are selectable; Both applies the procedure on the right. Dura, temporal "
        "layers, veins, arachnoid, sulcal backing and scalp layers are fitted independently to native left anatomy "
        "(the atlas left side is close to a mirror of the right; the left scalp reuses the right side's cut "
        "layout); left markings are redrawn by the same code on native left anatomy (the deep Sylvian, clinoid "
        "and optico-carotid markings and the instruments mirror the right).",
    ],
    "laterality": {
        "native": "Paired atlas anatomy retains the saved native left and right geometry.",
        "procedure": "Masks act on the selected side; Both operates on the right. The left dura, temporal fascial "
                     "layers and fat pads, illustrative veins, arachnoid and sulcal backing are built by the same "
                     "code on native left bone, muscle, cortex and arteries. Left scalp layers are built on native "
                     "left anatomy with the right side's cut layout mirrored. Left markings are redrawn by the same "
                     "code on native left anatomy; the deep Sylvian, clinoid and optico-carotid markings and the "
                     "instruments mirror the right illustrations.",
    },
}

CONTEXT_SCOPE = ("Available cranial/head atlas structures within the head bounding region, plus the context-only "
                 "illustrative superficial temporal vein, variant anastomotic veins and the illustrative scalp, "
                 "galea and pericranium of the vault; All does not imply a complete human atlas or a complete "
                 "segmentation of every tissue.")

# Group lists in manifest.variants (a static snapshot of the guide steps) are
# brought in line with assets/app.js buildSteps.
VARIANT_GROUPS = {
    ("bone", "brain", "dura", "arteries", "nerves"): ["bone", "brain", "dura", "arteries", "veins", "nerves",
                                                      "arachnoid"],
    ("bone", "brain", "arteries", "nerves"): ["bone", "brain", "arteries", "veins", "nerves", "arachnoid"],
    ("brain", "arteries", "nerves", "bone"): ["brain", "arteries", "veins", "nerves", "arachnoid", "bone"],
}
FISSURE_NOTE = ("The translucent arachnoid and superficial Sylvian veins are original illustrations; the arachnoid "
                "opens along a schematic fissure line on native cortex. Opercular separation, small veins and the "
                "depth of the split are not reconstructed.")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------- base
def read_base(path: Path | None) -> tuple[list[dict], dict]:
    if path is None:
        try:
            raw = b"".join(subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:assets/anatomy-{i:02}.bin"],
                                          check=True, capture_output=True).stdout for i in range(1, 7))
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = exc.stderr.decode(errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else exc
            sys.exit(f"Cannot read the base records with git show {BASE_COMMIT} ({detail}).\n"
                     f"This needs a full clone that contains commit {BASE_COMMIT} (not a ZIP download or shallow "
                     f"clone). Alternatively pass --base with the records of the original release, e.g. from "
                     f"`python3 tools/anatomy.py unpack` run on a checkout of {BASE_COMMIT}.")
        text = gzip.decompress(raw)
        origin = f"git show {BASE_COMMIT}:assets/anatomy-0[1-6].bin"
    else:
        raw = path.read_bytes()
        text = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        origin = str(path)
    records = json.loads(text)
    digest = sha256(json.dumps(records, separators=(",", ":")).encode())
    return records, {"origin": origin, "sha256": digest}


def check_base(records: list[dict], info: dict, allow_other: bool) -> None:
    problems = []
    if info["sha256"] != BASE_SHA256:
        problems.append(f"base hash {info['sha256'][:12]}… differs from the {BASE_COMMIT} import "
                        f"({BASE_SHA256[:12]}…)")
    derived = [m["id"] for m in records if m["id"].startswith(DERIVED_ID_PREFIXES)]
    if derived:
        problems.append(f"base already contains derived records ({', '.join(derived[:4])}…)")
    if any(m.get("markerRadius") is not None for m in records):
        problems.append("base already contains regenerated markings (markerRadius)")
    nonunit = []
    for m in records:
        if m["group"] in UNIT_NORMAL_GROUPS:
            continue
        _, n = common.decode(m)
        length = np.linalg.norm(n, axis=1)
        length = length[np.isfinite(length) & (length > 1e-6)]
        if len(length) and (np.abs(length - 1) > 1e-3).mean() > 0.01:
            nonunit.append(m["id"])
    if nonunit:
        problems.append(f"{len(nonunit)} base records have non-unit normals (AO already baked?), "
                        f"e.g. {nonunit[:3]}")
    if not problems:
        return
    hard = [p for p in problems if not p.startswith("base hash")]
    if hard or not allow_other:
        sys.exit("Refusing to build on this base:\n  - " + "\n  - ".join(problems))
    log("WARNING: " + problems[0])


# --------------------------------------------------------------- validation
def _vec(v, n: int = 3) -> np.ndarray | None:
    a = np.asarray(v, dtype=float) if isinstance(v, (list, tuple)) else None
    return a if a is not None and a.shape == (n,) and np.isfinite(a).all() else None


def validate_flap(rid: str, flap) -> list[str]:
    """FLAP CONTRACT fields: finite pivot, unit and mutually orthogonal axis/away/lift, radius > 0, finite angle."""
    if not isinstance(flap, dict):
        return [f"{rid}: flap must be an object"]
    errors = []
    if flap.get("key") not in FLAP_KEYS:
        errors.append(f"{rid}: flap key {flap.get('key')!r} not in {sorted(FLAP_KEYS)}")
    vec = {k: _vec(flap.get(k)) for k in ("pivot", "axis", "away", "lift")}
    for k, v in vec.items():
        if v is None:
            errors.append(f"{rid}: flap {k} must be a finite [x, y, z]")
    units = [k for k in ("axis", "away", "lift") if vec[k] is not None]
    for k in units:
        if abs(np.linalg.norm(vec[k]) - 1) > 1e-4:
            errors.append(f"{rid}: flap {k} is not unit length ({np.linalg.norm(vec[k]):.6f})")
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if abs(float(vec[a] @ vec[b])) > 1e-4:
                errors.append(f"{rid}: flap {a} and {b} are not orthogonal (dot {float(vec[a] @ vec[b]):.6f})")
    r, ang = flap.get("radius"), flap.get("angle")
    if not (isinstance(r, (int, float)) and not isinstance(r, bool) and np.isfinite(r) and r > 0):
        errors.append(f"{rid}: flap radius must be a positive number")
    if not (isinstance(ang, (int, float)) and not isinstance(ang, bool) and np.isfinite(ang) and 0 < ang <= 2 * np.pi):
        errors.append(f"{rid}: flap angle must be in (0, 2 pi]")
    hp = flap.get("hingeProfile")
    if hp is not None and (_vec(hp.get("c") if isinstance(hp, dict) else None) is None
                           or _vec(hp.get("range"), 2) is None):
        errors.append(f"{rid}: flap hingeProfile needs finite c [3] and range [2]")
    elif hp is not None and not _vec(hp.get("range"), 2)[1] > _vec(hp.get("range"), 2)[0]:
        errors.append(f"{rid}: flap hingeProfile range must be increasing")
    return errors


def validate_flap_geometry(rid: str, flap: dict, p: np.ndarray, n: np.ndarray) -> list[str]:
    """What the flap fields mean, checked on the record's vertices: the flap lies on the ``away`` side of the
    hinge (min d >= -1e-3, median d > 0) and ``lift`` is its outer side (the outward-facing vertex normals point
    along lift on average)."""
    pivot, away, lift = (np.asarray(flap[k], dtype=float) for k in ("pivot", "away", "lift"))
    d = (p.astype(np.float64) - pivot) @ away
    errors = []
    if d.min() < -1e-3 or not np.median(d) > 0:
        errors.append(f"{rid}: flap vertices are not on the away side of the hinge (min d {d.min():.4f}, "
                      f"median {np.median(d):.4f})")
    nn = n.astype(np.float64) / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    outward = nn[(nn * (p - SHELL_CENTRE)).sum(1) > 0]
    if not len(outward) or not float((outward @ lift).mean()) > 0:
        errors.append(f"{rid}: flap lift does not point to the flap's outer side")
    return errors


def validate(records: list[dict]) -> dict:
    errors: list[str] = []
    ids = Counter(m["id"] for m in records)
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        errors.append(f"duplicate ids: {dup[:5]}")
    idset = set(ids)
    refs = idset | {m["nativeName"] for m in records} | {m["name"] for m in records}
    ao_min, ao_max = 1.0, 0.0
    vertices = 0
    for m in records:
        rid = m["id"]
        if m.get("group") not in KNOWN_GROUPS:
            errors.append(f"{rid}: unknown group {m.get('group')!r}")
        try:
            buf = base64.b64decode(m["data"], validate=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rid}: data does not decode ({exc})")
            continue
        if len(buf) != m["count"] * 24:
            errors.append(f"{rid}: {len(buf)} bytes for count {m['count']}")
            continue
        if m["count"] == 0 or m["count"] % 3:
            errors.append(f"{rid}: count {m['count']} is not a positive multiple of 3")
        vertices += m["count"]
        f = np.frombuffer(buf, dtype="<f4").reshape(-1, 6)
        p, n = f[:, :3], f[:, 3:]
        if not np.isfinite(f).all():
            errors.append(f"{rid}: non-finite position or normal")
            continue
        lo, hi = p.min(0), p.max(0)
        if not (np.allclose(lo, m["bounds"][0], atol=1e-4) and np.allclose(hi, m["bounds"][1], atol=1e-4)):
            errors.append(f"{rid}: bounds do not match vertex data")
        length = np.linalg.norm(n.astype(np.float64), axis=1)
        if m["group"] in UNIT_NORMAL_GROUPS:
            if np.abs(length - 1).max() > 1e-3:
                errors.append(f"{rid}: annotation/instrument normals are not unit length")
        else:
            a, b = float(length.min()), float(length.max())
            ao_min, ao_max = min(ao_min, a), max(ao_max, b)
            if a < AO_MIN - 1e-3 or b > 1 + 1e-3:
                errors.append(f"{rid}: normal length (AO) outside [{AO_MIN}, 1]: {a:.4f}..{b:.4f}")
        if "baseId" in m and m["baseId"] not in idset:
            errors.append(f"{rid}: baseId {m['baseId']} does not exist")
        if "follow" in m and m["follow"] and m["follow"] not in refs:
            errors.append(f"{rid}: follow target {m['follow']} does not exist")
        if "opacity" in m and not (0 < float(m["opacity"]) <= 1):
            errors.append(f"{rid}: opacity {m['opacity']} outside (0, 1]")
        if "fissureLine" in m:
            fl = np.asarray(m["fissureLine"], dtype=float)
            if fl.ndim != 2 or fl.shape[1] != 3 or len(fl) < 2 or not np.isfinite(fl).all():
                errors.append(f"{rid}: fissureLine must be >= 2 finite [x,y,z] points")
        if "isolateKeep" in m:
            k = m["isolateKeep"]
            c = np.asarray(k.get("center", []), dtype=float) if isinstance(k, dict) else np.zeros(0)
            if c.shape != (3,) or not np.isfinite(c).all() or not float(k.get("radius", 0)) > 0:
                errors.append(f"{rid}: isolateKeep needs a finite [x,y,z] center and a positive radius")
        if m.get("procedureSide") not in (None, "right", "left"):
            errors.append(f"{rid}: procedureSide {m.get('procedureSide')!r}")
        if "flap" in m:
            ferr = validate_flap(rid, m["flap"])
            errors += ferr or validate_flap_geometry(rid, m["flap"], p, n)
        if "incisionLine" in m:
            il = np.asarray(m["incisionLine"], dtype=float)
            if il.ndim != 2 or il.shape[1] != 3 or len(il) < 2 or not np.isfinite(il).all():
                errors.append(f"{rid}: incisionLine must be >= 2 finite [x,y,z] points")
        for key in ("color", "innerColor"):
            if key in m and not (isinstance(m[key], str) and len(m[key]) == 7 and m[key][0] == "#"
                                 and all(c in "0123456789abcdefABCDEF" for c in m[key][1:])):
                errors.append(f"{rid}: {key} {m[key]!r} is not #rrggbb")
        if m.get("markerRadius") is not None:
            r = m["markerRadius"]
            if not (isinstance(r, (int, float)) and np.isfinite(r) and r > 0):
                errors.append(f"{rid}: markerRadius must be a positive number")
            kind = m.get("markerKind", 0)
            if m.get("marker") and kind == 1 and _vec(m.get("markerCenter")) is None:
                errors.append(f"{rid}: markerKind 1 needs a finite markerCenter [x, y, z]")
            if m.get("marker") and kind == 2:
                rng = _vec(m.get("markerRange"), 2)
                if m.get("markerAxis") not in (0, 1, 2) or rng is None or not rng[1] > rng[0]:
                    errors.append(f"{rid}: markerKind 2 needs markerAxis in 0..2 and an increasing markerRange")
    if errors:
        raise SystemExit("Validation failed:\n  - " + "\n  - ".join(errors[:40]))
    return {"records": len(records), "vertices": vertices, "triangles": vertices // 3,
            "normalLength": [round(ao_min, 4), round(ao_max, 4)]}


# ------------------------------------------------------------------ manifest
def sync_variants(manifest: dict) -> int:
    changed = 0
    for variant in manifest.get("variants", []):
        for step in variant.get("steps", []):
            new = VARIANT_GROUPS.get(tuple(step.get("groups", [])))
            if new:
                step["groups"] = list(new)
                changed += 1
            if step.get("id") == "fissure" and step.get("note") != FISSURE_NOTE:
                step["note"] = FISSURE_NOTE
                changed += 1
    return changed


def module_provenance(name: str) -> dict:
    path = BUILD_DIR / f"{name}.py"
    return {"module": f"tools/anatomy_build/{name}.py", "sha256": sha256(path.read_bytes())}


def tooling_provenance() -> list[dict]:
    """Hashes of the shared code every step depends on."""
    return [{"file": str(p.relative_to(REPO)), "sha256": sha256(p.read_bytes())}
            for p in (BUILD_DIR / "common.py", Path(__file__).resolve())]


# Where the claim in PROSE["extraction"] about modifiers was checked (tools/anatomy_build/blender_export.py).
SOURCE_INSPECTION = {
    "tool": "tools/anatomy_build/blender_export.py",
    "blender": "5.0.1",
    "scene": "Startup.blend from the Z-Anatomy application template (Z-Anatomy_Template.zip, file dated 2023-05-02)",
    "sceneSha256": "9de642d94eb00ffe19c2f1e1ae27f2042269951d86e6689a7eb925369b32a86d",
    "note": "Differs from source_sha256. Spot checks: Frontal bone reproduces triangle for triangle and Temporal "
            "pole.r has the same vertex set, both to 3e-6 display units, so the geometry is taken to be the same. "
            "No cortical parcel carries a modifier in this scene.",
}


def update_manifest(manifest: dict, records: list[dict], build: dict) -> dict:
    structures = [common.structure_entry(m) for m in records]
    manifest["structures"] = structures
    manifest["structure_count"] = len(records)
    manifest["vertex_count"] = int(sum(m["count"] for m in records))
    ctx = [m for m in records if m.get("contextOnly")]
    ce = manifest.setdefault("context_expansion", {})
    ce["scope"] = CONTEXT_SCOPE
    ce["structures"] = len(ctx)
    ce["groups"] = dict(Counter(m["group"] for m in ctx))
    ce.setdefault("omitted", [])
    if "markings" in manifest:
        manifest["markings"]["items"] = [s for s in structures if s.get("marker") and s.get("procedureSide") == "right"]
    for key, value in PROSE.items():
        manifest[key] = value
    sync_variants(manifest)
    # record the resulting state, not a change count, so a rebuild of unchanged inputs is byte-identical
    build["variants"] = ("snapshot of assets/app.js buildSteps (guideQA.variantSnapshot()); legacy cranial group "
                         "lists and the fissure-step note are re-synced by the build")
    manifest["build"] = build
    return manifest


# ---------------------------------------------------------------------- main
def concise(name: str, summary: dict) -> dict:
    """A short, stable digest of a step summary for the manifest."""
    if name == "brain":
        return {"refinedRecords": len(summary.get("refined", {})), "contextOnlyRemoved": summary.get("uncontext"),
                "added": summary.get("added"), "trianglesBefore": summary.get("triangles_before"),
                "trianglesAfter": summary.get("triangles_after")}
    if name == "dura":
        return {"replaced": sorted(summary)}
    if name == "temporal_layers":
        return {"replaced": summary.get("replaced")}
    if name in ("veins", "arachnoid"):
        return {"added": [{"id": r["id"], **({"triangles": r["triangles"]} if "triangles" in r else {})}
                          for r in summary.get("records", [])]}
    if name == "scalp":
        return {"added": [{"id": r["id"], "triangles": r["triangles"]} for r in summary.get("records", [])],
                "thicknessMm": {k: round(v, 3) for k, v in (summary.get("thicknessMm") or {}).items()},
                "flapRight": summary.get("flap_r"),
                "superficialVesselsReseated": summary.get("superficialVessels"),
                "pericraniumTrianglesInWindow": [summary.get("periInWindow"), summary.get("periInWindow-l")]}
    if name == "markers":
        return {"verticesBefore": summary.get("verticesBefore"), "verticesAfter": summary.get("verticesAfter"),
                "radiusMm": {k: round(v, 3) for k, v in (summary.get("radiusMm") or {}).items()},
                "gapMm": round(summary.get("gapMm") or 0, 3),
                "incisionInsideFieldMm": {"right": summary.get("field_r"), "left": summary.get("field_l")}}
    if name == "ao":
        hist = summary.get("histogram", {})
        return {"normalsFlippedOutward": summary.get("normalsFlippedOutward", {}).get("count"),
                "meanAoByGroup": {g: v["mean"] for g, v in hist.items() if isinstance(v, dict) and "mean" in v}}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, help="base records JSON (or gzip) instead of git show " + BASE_COMMIT)
    ap.add_argument("--allow-other-base", action="store_true",
                    help="accept a base whose hash differs from the import (still refuses derived data)")
    ap.add_argument("--check", action="store_true", help="build and validate only; write nothing to the repo")
    ap.add_argument("--save-records", type=Path, help="also write the final records JSON here")
    ap.add_argument("--summary", type=Path, help="write the full per-step summaries (JSON) here")
    args = ap.parse_args()

    t_all = time.time()
    records, base_info = read_base(args.base)
    log(f"base: {len(records)} records from {base_info['origin']} (sha256 {base_info['sha256'][:12]}…)")
    check_base(records, base_info, args.allow_other_base)

    steps, full = [], {}
    for name in STEPS:
        t0 = time.time()
        log(f"[{name}] running…")
        module = importlib.import_module(name)
        before = {m["id"] for m in records}
        summary = module.apply(records)   # ao.apply keeps the anchors set by earlier steps itself
        secs = round(time.time() - t0, 1)
        after = {m["id"] for m in records}
        full[name] = {"seconds": secs, "summary": summary}
        steps.append({"step": name, **module_provenance(name), "description": STEP_INFO[name],
                      "added": sorted(after - before), "removed": sorted(before - after),
                      "result": concise(name, summary)})
        log(f"[{name}] done in {secs:.1f} s; {len(records)} records")

    if args.save_records:
        common.save_records(records, args.save_records)
        log(f"records written to {args.save_records}")
    if args.summary:
        args.summary.write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    stats = validate(records)
    log(f"validated: {stats}")

    payload = gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    gz_mb = len(payload) / 1e6
    log(f"gzip size: {gz_mb:.2f} MB (budget {GZIP_BUDGET_MB} MB); sha256 {sha256(payload)}")
    if gz_mb > GZIP_BUDGET_MB:
        raise SystemExit(f"gzip data {gz_mb:.2f} MB exceeds the {GZIP_BUDGET_MB} MB budget")

    build = {
        "tool": "tools/build_anatomy.py",
        "base": {"commit": BASE_COMMIT, "source": base_info["origin"], "sha256": base_info["sha256"],
                 "records": BASE_RECORDS},
        "steps": steps,
        "tooling": tooling_provenance(),
        "sourceInspection": SOURCE_INSPECTION,
        "studyPlates": "hand-maintained fallback images; kept from the existing manifest, not rendered by this build",
        "result": {**stats, "gzipBytes": len(payload), "sha256": sha256(payload)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "notes": "Deterministic: every step seeds its randomness. Record descriptions and source fields state "
                 "which geometry is illustrative.",
    }
    try:
        import scipy
        build["environment"]["scipy"] = scipy.__version__
    except ImportError:
        pass

    if args.check:
        log("--check: repo left unchanged")
    else:
        manifest_path = REPO / "data" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        update_manifest(manifest, records, build)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for i, chunk in enumerate(common.CHUNKS):
            chunk.write_bytes(payload[len(payload) * i // len(common.CHUNKS):
                                      len(payload) * (i + 1) // len(common.CHUNKS)])
        log(f"wrote {len(common.CHUNKS)} chunks and {manifest_path.relative_to(REPO)}")
    log(f"total {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
