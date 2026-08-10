from __future__ import annotations

import argparse
import json
from pathlib import Path

from cua_jepa.sft_compare import audit_controlled_run_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a controlled Model 2 and Model 4 SFT pair")
    parser.add_argument("model2_run", type=Path)
    parser.add_argument("model4_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_controlled_run_pair(args.model2_run, args.model4_run)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
