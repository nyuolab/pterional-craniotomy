"""Inspect the Z-Anatomy source scene (developer tool; not needed at build time).

Run headless with Blender (tested with Blender 5.0.1 on the Z-Anatomy ``Startup.blend``
from the ``Z-Anatomy_Template.zip`` application template, file dated 2023-05-02)::

    blender -b Startup.blend --python tools/anatomy_build/blender_export.py -- \
        --records RECORDS.json --report report.json [--export OUT.json.gz NAME ...]

It does three things:

1. **Transform check.** Re-derives display coordinates for a few unmodified records
   (default ``Frontal bone``) from the scene with the documented transform
   (world matrix, minus native origin [0, 0, 1.635], x50, display axes =
   (native X, native Z, -native Y)) and reports the maximum error against the record.
2. **Modifier report.** Lists every object behind a brain-group record that carries
   modifiers (type, levels, viewport flag).
3. **Optional evaluated export.** With ``--export``, writes depsgraph-evaluated
   (modifiers applied) triangles for the named objects, in display coordinates, as gzip
   JSON ``{nativeName: {"pos": [...], "tris": [...]}}``. Z-Anatomy geometry is
   CC BY-SA 4.0; keep that licence and this provenance with any exported file.

Findings for the base records (2026-11, recorded in brain.py):
* The downloaded ``Startup.blend`` has SHA-256
  9de642d94eb00ffe19c2f1e1ae27f2042269951d86e6689a7eb925369b32a86d, not the manifest's
  9f08a17e...; the geometry nevertheless matches (``Frontal bone`` reproduces
  triangle-for-triangle to 3e-6 display units; ``Temporal pole.r`` has the same vertex
  set to 3e-6, triangulated in a different order), so the scene content is the same.
* No cortical parcel carries a modifier. Subdivision Surface appears only on a few
  tiny brainstem nuclei; Mirror on midline structures; Decimate on the corpus callosum.
  Evaluated export would therefore not smooth the cortex, and no exported data file
  is stored in the repository.
"""
import base64
import gzip
import json
import sys

import bpy
import numpy as np

ORIGIN = np.array([0.0, 0.0, 1.635])
SCALE = 50.0


def to_display(co: np.ndarray) -> np.ndarray:
    v = (co - ORIGIN) * SCALE
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def mesh_triangles(ob, evaluated: bool = False):
    if evaluated:
        dg = bpy.context.evaluated_depsgraph_get()
        ob_eval = ob.evaluated_get(dg)
        me = ob_eval.to_mesh()
        mw = ob_eval.matrix_world
    else:
        me, mw = ob.data, ob.matrix_world
    co = np.array([(mw @ v.co)[:] for v in me.vertices], dtype=np.float64)
    tris = []
    for p in me.polygons:  # fan triangulation, as in the original extraction
        vs = list(p.vertices)
        for k in range(1, len(vs) - 1):
            tris.append((vs[0], vs[k], vs[k + 1]))
    if evaluated:
        ob_eval.to_mesh_clear()
    return to_display(co), np.array(tris, dtype=np.int64)


def record_positions(m: dict) -> np.ndarray:
    f = np.frombuffer(base64.b64decode(m["data"]), dtype="<f4").reshape(-1, 6)
    return f[:, :3].astype(np.float64)


def check_transform(records: dict, names: list[str]) -> dict:
    out = {}
    for name in names:
        V, F = mesh_triangles(bpy.data.objects[name])
        P = V[F.reshape(-1)]
        R = record_positions(records[name])
        res = {"scene_vertices": len(P), "record_vertices": len(R)}
        if P.shape == R.shape:
            res["max_abs_error_ordered"] = float(np.abs(P - R).max())
        # order-independent: distance from every record vertex to the nearest scene vertex
        from mathutils.kdtree import KDTree
        kd = KDTree(len(V))
        for k, co in enumerate(V):
            kd.insert(co, k)
        kd.balance()
        res["max_nearest_vertex_error"] = float(max(kd.find(r)[2] for r in R))
        out[name] = res
    return out


def modifier_report(records: dict) -> dict:
    rep = {}
    for name, m in records.items():
        if m.get("group") != "brain":
            continue
        ob = bpy.data.objects.get(name)
        if ob is None or not ob.modifiers:
            continue
        rep[name] = [{"type": md.type, "levels": getattr(md, "levels", None),
                      "render_levels": getattr(md, "render_levels", None),
                      "show_viewport": md.show_viewport} for md in ob.modifiers]
    return rep


def main(argv: list[str]) -> None:
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    opts = {"--records": None, "--report": "blender_report.json"}
    export_path, export_names, check = None, [], ["Frontal bone"]
    i = 0
    while i < len(args):
        if args[i] in opts:
            opts[args[i]] = args[i + 1]
            i += 2
        elif args[i] == "--check":
            check = args[i + 1].split(",")
            i += 2
        elif args[i] == "--export":
            export_path, export_names = args[i + 1], args[i + 2:]
            break
        else:
            sys.exit(f"unknown argument {args[i]}")
    records = {m["nativeName"]: m for m in json.load(open(opts["--records"]))}
    report = {
        "blend_file": bpy.data.filepath,
        "transform_check": check_transform(records, check),
        "brain_modifiers": modifier_report(records),
        "brain_records_missing_in_scene": sorted(n for n, m in records.items()
                                                 if m.get("group") == "brain" and n not in bpy.data.objects),
    }
    if export_path:
        data = {}
        for name in export_names:
            V, F = mesh_triangles(bpy.data.objects[name], evaluated=True)
            data[name] = {"pos": V.round(6).tolist(), "tris": F.tolist()}
        with gzip.open(export_path, "wt") as fh:
            json.dump({"source": "Z-Anatomy Startup.blend (CC BY-SA 4.0), depsgraph-evaluated", "objects": data}, fh)
        report["exported"] = export_names
    json.dump(report, open(opts["--report"], "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "brain_modifiers"}, indent=1))
    print("brain objects with modifiers:", len(report["brain_modifiers"]))


main(sys.argv)
