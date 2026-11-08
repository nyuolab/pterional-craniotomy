# Pterional craniotomy

A browser-based anatomy guide for the pterional approach. The model follows the procedure from positioning through closure, with options for temporalis dissection, clinoidectomy, and Sylvian fissure splitting.

## Local preview

Run from the repository root, then open [localhost:8000](http://localhost:8000/).

```sh
python3 -m http.server 8000
```

## Source

- [index.html](index.html) and [styles.css](assets/styles.css) contain the page layout and controls.
- [app.js](assets/app.js) contains the procedure steps (`buildSteps`), camera presets (`setOperativeView`), and WebGL renderer.
- [manifest.json](data/manifest.json) contains paper references, atlas metadata, and fallback study plates.
- [loader.js](assets/loader.js) loads the manifest and six compressed anatomy chunks.

Use [anatomy.py](tools/anatomy.py) to unpack the mesh records for inspection and repack them after editing.

```sh
python3 tools/anatomy.py unpack /tmp/pterional-meshes.json
python3 tools/anatomy.py pack /tmp/pterional-meshes.json
```

Each record stores a base64 vertex buffer of little-endian float32 values, with XYZ position and XYZ normal per vertex. Every three vertices form a triangle. If you change the number of meshes, update the manifest and the count check in the tool.

## Operative View

The view follows the field orientation in Rodríguez Rubio et al., Figures 10C–D and 11. The skull-base side is at the top of the frame. The camera is aimed from atlas landmarks in the window toward the exposed field, with a separate aim for basal steps. The angles are illustrative because the paper does not provide calibrated camera poses. The patient head rotation and extension described in the positioning section are separate from the camera. [Immersive Surgical Anatomy of the Pterional Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC6759424/)

## GitHub Pages

Push the repository with `index.html` at the root. Under **Settings → Pages**, choose **Deploy from a branch**, then select **main** and **/ (root)**. There is no build step.

## Anatomy credits

Z-Anatomy (CC BY-SA 4.0), derived in part from BodyParts3D (DBCLS, CC BY-SA 2.1 Japan). See the manifest and the guide’s **About** panel for provenance and limits of the illustrative geometry.
