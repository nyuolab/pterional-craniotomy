# Pterional craniotomy

A browser-based anatomy guide for the pterional approach. The model follows the procedure from positioning through closure, with options for temporalis dissection, clinoidectomy, and Sylvian fissure splitting.

## Local preview

Run from the repository root, then open [localhost:8000](http://localhost:8000/).

```sh
python3 -m http.server 8000
```

## Source

- [index.html](index.html) and [styles.css](assets/styles.css) contain the page layout and controls.
- [app.js](assets/app.js) contains the procedure steps (`buildSteps`), camera controls, and WebGL renderer.
- [manifest.json](data/manifest.json) contains paper references, atlas metadata, mesh provenance, the build record (`build`) and fallback study plates.
- [loader.js](assets/loader.js) loads the manifest and six compressed anatomy chunks. It checks that the number of mesh records equals the manifest's `structure_count`.

## Rendering

- **Colors** (under **Layers & View**). *Surgical (lifelike)* is the default. It uses a per-tissue material shader with wet highlights, soft subsurface-style diffuse light, light falloff toward the edge of the field, cut bone edges with pale tables and pinkish diploë (the layering is estimated in the renderer), and tone mapping. *Atlas (teaching colors)* restores the manifest colors with simple shading. The choice is remembered in the browser.
- **Surface texture is decoration.** The fine patterns in the Surgical palette (bone porosity, muscle fibres, fascial streaks, fat lobules, very faint branching on the cortex) are procedural. They do not show real vessels or measured tissue structure; no dural vessels are drawn. On GPUs without high-precision fragment shaders the procedural texture is switched off and only the lit shading remains.
- **Ambient occlusion.** Crease shading is precomputed per vertex from the mesh geometry and applies in both palettes.
- **Layers.** The layers are scalp skin, galea, pericranium, bone, brain, muscles, superficial fascia, deep fascia, fat pads, arteries, veins, nerves, dura and arachnoid. Translucent meshes (the arachnoid, the cavernous sinus and the suprafascial fat) are drawn after the opaque meshes. The arachnoid is skipped when picking, so clicks select the anatomy beneath it; the other translucent meshes can be clicked. The arachnoid is drawn milkier over the Sylvian fissure, is shown only where its side's dura is open (or no dura is shown), and in the Sylvian step opens along a schematic fissure line. Atlas sheets that coincide with the illustrative scalp (epicranial aponeurosis, frontalis, occipitalis, temporoparietalis, atlas superficial temporal fascia) are hidden while the scalp is drawn.
- **Scalp and flaps.** The scalp is an illustrative layered shell: skin with subcutaneous tissue (a closed slab whose cut walls show dermis over fat), galea and pericranium. In the incision step the skin and galea edges part behind a knife point that travels along the scalp record's `incisionLine`. Flaps move with a curl/peel deformation in the vertex shader, mirrored in JavaScript (`curlPoint`) for labels, picking and depth sorting: each flap peels from its free edge toward a straight hinge and rolls over it, following the FLAP CONTRACT fields on the flap records (`flap = {key, pivot, axis, away, lift, radius, angle}`) with the hinge height taken from the data. The scalp flap rolls over the orbital margin; the pericranial flap, the fascial flap (subfascial technique) and the temporalis follow, and the dural leaf is first cut along a C-shaped groove and then reflected toward the sphenoid ridge. In the muscle-splitting technique scalp, fascia and temporalis fold together. At the end of the fold a drape table (built at load from the data) lays the straight part of the reflected scalp, galeal, fascial and pericranial flaps toward the face, about 9 mm above the facial skeleton or stationary scalp (a stand-in for the rolled sponge and the facial soft tissue the model lacks), instead of letting the large curved flap stand off like a sail; the dural leaf is laid on the bone beyond the window rim the same way. Shading normals of curled meshes are derived from the deformation itself (finite differences of the curl in the vertex shader). The superficial temporal artery, its frontal branch and the superficial temporal vein lie in the subcutaneous layer; their parts in front of the incision travel with the scalp flap. Cut edges are exact because the moving and stationary parts are drawn in two passes split on rest-pose coordinates. These motions are schematic, not soft-tissue simulation. *Separate Fascial Layers* explodes the temporal laminae and fat pads sideways in their rest pose and hides the scalp layers meanwhile.
- **Markings.** Colored markings are smooth tubes of constant radius (about 0.7 mm) with at least 0.1 mm of clearance above the surface they annotate (skin markings typically 0.1–0.3 mm; outlines projected onto bone and dura typically 0.4–0.6 mm, up to about 1.8 mm at dash ends; each record's description gives its range). They use a flat line shader with a soft key light and a dark silhouette, are widened in the vertex shader so they never draw thinner than about 2.6 device pixels, and are pulled 0.6 mm toward the camera in depth so they do not flicker against the surface beneath. They draw in with the stage progress; the incision line and the closure stitches draw in by arc length along the incision (behind the knife), and the dural line draws in with the C-shaped dural cut. Closure is layered: in *Close the galea* buried galeal stitches draw in on the galea while the skin edges stay slightly apart, and the skin stitches appear only in *Close the skin*.

## Anatomy data

The six chunks `assets/anatomy-01.bin` … `anatomy-06.bin` are one gzip-compressed JSON list of mesh records split into equal parts. Each record stores a base64 vertex buffer of little-endian float32 values: XYZ position and XYZ normal per vertex, and every three vertices form a triangle. Coordinates are display units (1 unit = 2 cm; +X is the patient's left, +Y superior, +Z anterior).

**The normal's length carries baked ambient occlusion,** from 0.3 (occluded) to 1 (open). Normals point outward; normalise them before lighting. Markings and instruments keep unit normals.

Besides geometry, some records carry fields the renderer reads: `flap` (hinge fields of the scalp and pericranial flaps; left records carry the x-mirrored vectors), `incisionLine` (the incision course on the skin, on the scalp flap records), `innerColor` (tint of the skin slab's cut walls and inner surface), `fissureLine` (arachnoid opening) and, on markings and instruments, `markerRadius` and `markerSurface`.

### Rebuilding the geometry

[build_anatomy.py](tools/build_anatomy.py) regenerates the shipped chunks and the generated parts of the manifest from the original atlas import (commit `7961958`):

```sh
python3 tools/build_anatomy.py            # rebuild assets/anatomy-0N.bin and data/manifest.json
python3 tools/build_anatomy.py --check    # build and validate only; write nothing
```

It reads the base records with `git show 7961958:assets/anatomy-0N.bin`, which needs a full clone containing that commit (not a ZIP download or a shallow clone), or from `--base FILE`. It refuses to run on data that a build has already derived. It needs Python 3 with numpy and scipy and takes about 6 minutes on a 16-core machine. The output is deterministic: rebuilding unchanged inputs reproduces the chunks and `data/manifest.json` byte for byte (per-step timings go to stderr, or to `--summary FILE`). The manifest's `study_plates` (fallback images shown only without WebGL) are maintained by hand; the build keeps them.

The build applies these steps from [tools/anatomy_build](tools/anatomy_build), in order. Each one is also a standalone `IN.json OUT.json` command, and running them in this order by hand gives the same records as the build:

1. `brain.py` refines the cortical parcels seen through the bone window, and along the basal frontal, temporal and Sylvian surfaces, with interpolating PN triangles. Source vertices do not move, and no anatomical detail is added. Six window parcels per side become relevant, and an illustrative sulcal backing (cropped native white-matter surface) fills the seams between parcels. The Z-Anatomy cortex carries no Blender modifiers, so re-exporting with modifiers evaluated would not change it ([blender_export.py](tools/anatomy_build/blender_export.py) checks this).
2. `dura.py` fits the illustrative dura under the inner table on each side, covering the bone window with a margin and the lateral sphenoid ridge. It enters the bone slab (where atlas cortex crosses the atlas inner table) only inside the window cylinder, so it never shows over the cut bone edge. Where the sheet then cannot clear the atlas cortex at the window rim, it moves those few cortex vertices (about 15 per side) inward by at most 2.8 mm; the affected records' descriptions say so.
3. `temporal_layers.py` builds the illustrative temporal fascial laminae and the three fat pads as offset surfaces over the native temporalis and skull.
4. `veins.py` adds illustrative veins: the superficial middle cerebral vein with tributaries, the sphenoparietal and cavernous sinuses, the superficial temporal vein, and variant Trolard and Labbé veins. Their courses are derived from the atlas cortex, bone and arteries. The cavernous sinus is placed from the atlas oculomotor and abducens nerves, because the atlas carotid's cavernous segment lies lower than typical.
5. `arachnoid.py` adds the illustrative translucent arachnoid sheet, with the schematic `fissureLine` the renderer uses to open it. It is left out over the context-only orbital gyri and temporal pole.
6. `scalp.py` adds the illustrative scalp over the vault, from the brow to the occiput and down to the zygomatic arch: skin with subcutaneous tissue as one closed slab, the galea (continuous with the temporoparietal fascia), an empty loose areolar gap, and pericranium on the outer table (absent over the temporal fascia). The layers are offset shells over the atlas skull and temporalis and the illustrative temporal layers, about 7 mm thick in total over the vault. Each side is split into the pterional scalp flap in front of the incision, an anteriorly based pericranial flap, the stationary operative region and the context-only vault. The incision follows the upper part of the atlas schematic incision marking; its lower part is redrawn to start on the zygomatic arch about 1 cm in front of the estimated tragus and to pass behind the bone window. The pericranial cut is kept at least 1.5 mm outside the renderer's bone window. The flap records carry the hinge fields (`flap`) and the incision course (`incisionLine`). The left scalp uses the right side's cut layout on native left anatomy. Where the scalp covers them, the atlas superficial temporal artery and frontal branch and the illustrative superficial temporal vein are moved radially into the subcutaneous layer (their atlas course lies at the bone / temporal-fascia level); the records' descriptions say so.
7. `markers.py` redraws every schematic marking and instrument arrow as a clean stroke: smooth centrelines resampled evenly by arc length, swept as constant-radius tubes with round caps and cone arrowheads, at least 0.1 mm above the surface they annotate (the scalp skin or galea, the bed under the scalp flap, pericranium, bone or dura). Outlines of the bone window, burr holes and dural openings follow the renderer's analytic cuts, and the incision and the galeal and skin stitches follow `incisionLine` (the incision marking stops 5 mm above the scalp's free edge at the zygomatic arch). The prepared-field outline encloses the incision marking (by at least 8 mm away from the free edge; the step checks this and fails otherwise). It adds the galeal-closure marking `MARK-galea-sutures`. Deep markings (Sylvian trace, clinoid, optico-carotid region) are only smoothed. Left markings are fitted on native left anatomy; the deep markings and instruments are mirrored.
8. `ao.py` bakes ambient occlusion into the normal length. Sheets that the renderer moves (dura, fascial layers, fat pads, arachnoid and the scalp layers) occlude only themselves. It first flips closed records whose normals pointed inward and keeps the anchors set by earlier steps. It must run last.

The build then validates every record: data decodes, counts and bounds match, normals are finite, AO is in range, ids are unique, `baseId` targets exist, flap fields are finite with unit, mutually orthogonal hinge vectors and a positive radius, the flap vertices lie on the `away` side of the hinge with `lift` pointing to the flap's outer side, and marking reveal fields are complete. It writes `structures`, `structure_count`, `vertex_count`, the `context_expansion` counts, `markings.items`, the geometry prose and a `build` entry (steps, module hashes, hashes of `common.py` and the build script, base hash, where the source scene was inspected, output size and hash) into the manifest, and packs the chunks.

All left-side additions are fitted on the native left anatomy rather than mirrored (the left scalp reuses the right cut layout; the deep markings and the instruments are mirrored). The scalp layers, dura, fascial layers, fat pads, veins, arachnoid, sulcal backing and markings are original illustrations; their records say so in `source` and `description`.

The manifest's `variants` field is a static snapshot of the guide steps for all 12 configurations. The build keeps it; after changing `buildSteps`, regenerate it in the browser console with `JSON.stringify(guideQA.variantSnapshot())`.

### Inspecting or hand-editing records

Use [anatomy.py](tools/anatomy.py) to unpack the mesh records for inspection and repack them after editing.

```sh
python3 tools/anatomy.py unpack /tmp/pterional-meshes.json
python3 tools/anatomy.py pack /tmp/pterional-meshes.json
```

`pack` checks the record count against the manifest's `structure_count`. If you add or remove meshes, update the manifest; a rebuild does this for you. Hand edits are overwritten by the next rebuild, so put lasting changes in a build step.

## Camera and patient positioning

- **Camera controls.** A normal one-finger or mouse drag tilts vertically but does not yaw the front view left/right. Hold Alt/Option and drag horizontally to turn side-to-side. Shift/right-drag pans; two-finger movement pans and zooms, two-finger twist rolls, Q/E roll, and arrow keys rotate. Wheel or pinch zooms. Anatomical presets (Anterior, Right, Left, Top) follow the head.
- **Smooth interaction.** During pointer movement the renderer skips the picking-buffer pass and synchronous pixel readback, uses device-pixel ratio 1, and reduces one procedural texture octave. It restores full display quality and refreshes label visibility after 140 ms without movement. Picking still runs immediately on a click.
- **Positioning.** The Position step turns the head 20° contralaterally and extends it 20° (the example in Rodríguez Rubio et al.; the amount is target-dependent) about an estimated craniocervical pivot, and later steps keep the pose. Adjust it under **Layers & View › Patient Position** (a collapsible section, opened automatically on the Position step; Anatomical/Operative toggle, rotation, extension and lateral flexion sliders, **Default 20° / 20°**). Neck, body and table tilt are not modeled.
- **Surgeon's View** and the Position step use a room-fixed camera with a schematic Mayfield-type head holder and the head end of the table; an orientation gizmo in the corner shows the patient axes and gravity. Leaving the Position step returns the camera to the next step's head-framed view unless Surgeon's View was chosen.

## GitHub Pages

Push the repository with `index.html` at the root. Under **Settings → Pages**, choose **Deploy from a branch**, then select **main** and **/ (root)**. The site needs no build step; the anatomy chunks are committed.

## Anatomy credits

Z-Anatomy (CC BY-SA 4.0), derived in part from BodyParts3D (DBCLS, CC BY-SA 2.1 Japan). Original illustrative meshes are released under the same CC BY-SA 4.0 terms. See the manifest and the guide’s **About** panel for provenance and limits of the illustrative geometry.
