from __future__ import annotations

import argparse
import json
from pathlib import Path

from cua_jepa.observability import validate_run_artifacts


ROOT = Path(__file__).parents[1]


def main(run_id: str, expected_steps: int | None) -> None:
    report = validate_run_artifacts(ROOT / "artifacts" / run_id, expected_steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-steps", type=int)
    arguments = parser.parse_args()
    main(arguments.run_id, arguments.expected_steps)
