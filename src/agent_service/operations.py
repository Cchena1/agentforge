from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .backup import BackupManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentForge verified offline backup and restore")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup = subparsers.add_parser("backup", help="create a verified SQLite state backup")
    backup.add_argument("--state-dir", type=Path, required=True)
    backup.add_argument("--output-dir", type=Path, required=True)
    backup.add_argument("--vector-backend", choices=("sqlite", "qdrant"), default="sqlite")

    verify = subparsers.add_parser("verify", help="verify hashes and SQLite integrity")
    verify.add_argument("--backup-dir", type=Path, required=True)

    restore = subparsers.add_parser("restore", help="restore into an isolated or explicit target")
    restore.add_argument("--backup-dir", type=Path, required=True)
    restore.add_argument("--target-state-dir", type=Path, required=True)
    restore.add_argument("--replace-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: dict[str, Any]
    if args.command == "backup":
        path = BackupManager(args.state_dir, vector_backend=args.vector_backend).create(args.output_dir)
        verification = BackupManager.verify(path)
        result = {"backup_dir": str(path), "verification": verification.model_dump(mode="json")}
    elif args.command == "verify":
        verification = BackupManager.verify(args.backup_dir)
        result = verification.model_dump(mode="json")
    else:
        path = BackupManager.restore(
            args.backup_dir,
            args.target_state_dir,
            replace_existing=args.replace_existing,
        )
        result = {"restored_state_dir": str(path), "verified": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", True) and result.get("verification", {}).get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
