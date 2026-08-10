from __future__ import annotations

import argparse
import json
from pathlib import Path

from cua_jepa.sft_compare import paired_bootstrap


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired Model 2 and Model 4 predictions")
    parser.add_argument("model2_predictions", type=Path)
    parser.add_argument("model4_predictions", type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = paired_bootstrap(
        read_jsonl(args.model2_predictions),
        read_jsonl(args.model4_predictions),
        samples=args.samples,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
