"""Unpack or repack the anatomy chunks used by the browser.

To regenerate the shipped geometry from the original atlas import, use
``tools/build_anatomy.py`` instead; this tool only converts between the
chunks and an editable JSON file.
"""
import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MANIFEST = ROOT / "data" / "manifest.json"
CHUNKS = [ASSETS / f"anatomy-{i:02}.bin" for i in range(1, 7)]


def expected_count() -> int:
    """Number of mesh records the browser loader expects (manifest structure_count)."""
    return int(json.loads(MANIFEST.read_text(encoding="utf-8"))["structure_count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("unpack", "pack"))
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    if args.action == "unpack":
        data = gzip.decompress(b"".join(chunk.read_bytes() for chunk in CHUNKS))
        args.file.write_text(json.dumps(json.loads(data), indent=2), encoding="utf-8")
        return

    records = json.loads(args.file.read_text(encoding="utf-8"))
    count = expected_count()
    if not isinstance(records, list) or len(records) != count:
        parser.error(f"The browser loader expects {count} mesh records (manifest structure_count); "
                     f"update data/manifest.json when adding or removing meshes.")
    data = gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    for i, chunk in enumerate(CHUNKS):
        start = len(data) * i // len(CHUNKS)
        end = len(data) * (i + 1) // len(CHUNKS)
        chunk.write_bytes(data[start:end])


if __name__ == "__main__":
    main()
