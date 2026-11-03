"""Verify every code/data member listed in the supplied transfer manifest."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "TRANSFER_MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError("Unsafe manifest path: " + name)
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        if digest.hexdigest() != info["sha256"] or size != info["size_bytes"]:
            raise ValueError("Transfer mismatch: " + name)
    print("TRANSFER VERIFIED: %d files, including %d subject MAT files." %
          (len(manifest["files"]), sum(name.endswith(".mat") for name in manifest["files"])))


if __name__ == "__main__":
    main()
