from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = root / "evidence_manifest.json"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == target or "__pycache__" in path.parts:
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    target.write_text(
        json.dumps({"schema_version": 1, "files": files}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
