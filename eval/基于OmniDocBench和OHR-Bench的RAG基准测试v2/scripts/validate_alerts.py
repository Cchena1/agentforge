from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(__file__).resolve().parents[3] / "deploy" / "observability"


def _run(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promtool", type=Path, required=True)
    parser.add_argument("--amtool", type=Path, required=True)
    args = parser.parse_args()
    checks = {
        "prometheus_config": _run(
            [str(args.promtool), "check", "config", "prometheus.yml"], DEPLOY
        ),
        "alert_rules": _run([str(args.promtool), "check", "rules", "alert_rules.yml"], DEPLOY),
        "alert_rule_tests": _run([str(args.promtool), "test", "rules", "rule_tests.yml"], DEPLOY),
        "alertmanager_config": _run(
            [str(args.amtool), "check-config", "alertmanager.example.yml"], DEPLOY
        ),
    }
    result = {
        "schema_version": 1,
        "all_passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }
    output = ROOT / "results" / "alert_validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
