"""Unpack or repack the anatomy chunks used by the browser."""
import argparse
import gzip
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CHUNKS = [ASSETS / f"anatomy-{i:02}.bin" for i in range(1, 7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("unpack", "pack"))
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    if args.action == "unpack":
        data = gzip.decompress(b"".join(chunk.read_bytes() for chunk in CHUNKS))
        args.file.write_text(json.dumps(json.loads(data), indent=2), encoding="utf-8")
        return

    records = json.loads(args.file.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 614:
        parser.error("The browser loader expects 614 mesh records.")
    data = gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    for i, chunk in enumerate(CHUNKS):
        start = len(data) * i // len(CHUNKS)
        end = len(data) * (i + 1) // len(CHUNKS)
        chunk.write_bytes(data[start:end])


if __name__ == "__main__":
    main()
