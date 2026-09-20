#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "data_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file() or path in (MANIFEST, RESULTS / "README.md"):
            continue
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "schema_version": 1,
        "scope": "public machine-readable evidence for DualPath-CEM",
        "files": files,
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST)
    print(json.dumps({"manifest": str(MANIFEST), "files": len(files)}))


if __name__ == "__main__":
    main()
