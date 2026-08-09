from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


def bundle_records(archive: tarfile.TarFile) -> dict[str, tuple[dict[str, Any], str]]:
    records: dict[str, tuple[dict[str, Any], str]] = {}
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith("/bundle.json"):
            payload = archive.extractfile(member)
            if payload is None:
                raise RuntimeError(f"Cannot read {member.name}")
            record = json.load(payload)
            records[record["bundle_id"]] = (record, member.name.rsplit("/", 1)[0])
    return records


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    payload = archive.extractfile(name)
    if payload is None:
        raise RuntimeError(f"Cannot read {name}")
    return payload.read()


def text_category(hint: str) -> str:
    value = hint.lower()
    if "formula" in value:
        return "formula"
    if "email" in value or "@" in value or "recipient" in value:
        return "email"
    if "url" in value or "website" in value:
        return "url"
    if "phone" in value or "mobile" in value or "tel" in value:
        return "phone"
    if "search" in value or "filter" in value:
        return "search_or_filter"
    if any(
        word in value for word in ("comment", "message", "description", "reply", "note", "body")
    ):
        return "message_or_note"
    if any(word in value for word in ("name", "title", "subject", "company", "product", "project")):
        return "name_or_title"
    return "generic"


def verify(before_root: Path, after_root: Path) -> dict[str, Any]:
    before_tars = sorted(before_root.rglob("*.tar"))
    after_tars = sorted(after_root.rglob("*.tar"))
    before_relative = {path.relative_to(before_root): path for path in before_tars}
    after_relative = {path.relative_to(after_root): path for path in after_tars}
    if before_relative.keys() != after_relative.keys():
        raise RuntimeError("The before and after shard sets differ")

    type_count = 0
    non_type_count = 0
    current_count = 0
    changed_type_images = 0
    retargeted_type_actions = 0
    unchanged_type_images: list[str] = []
    texts: list[str] = []
    categories: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()

    for relative in sorted(before_relative):
        with (
            tarfile.open(before_relative[relative], "r") as before,
            tarfile.open(after_relative[relative], "r") as after,
        ):
            old_records = bundle_records(before)
            new_records = bundle_records(after)
            if old_records.keys() != new_records.keys():
                raise RuntimeError(f"Bundle set changed in {relative}")
            for bundle_id in old_records:
                old, old_prefix = old_records[bundle_id]
                new, new_prefix = new_records[bundle_id]
                old_outer = {key: value for key, value in old.items() if key != "branches"}
                new_outer = {key: value for key, value in new.items() if key != "branches"}
                if old_outer != new_outer:
                    raise RuntimeError(f"Non-branch bundle metadata changed: {bundle_id}")
                old_current = member_bytes(before, f"{old_prefix}/{old['current_file']}")
                new_current = member_bytes(after, f"{new_prefix}/{new['current_file']}")
                if old_current != new_current:
                    raise RuntimeError(f"Current screenshot changed: {bundle_id}")
                current_count += 1
                if len(old["branches"]) != len(new["branches"]):
                    raise RuntimeError(f"Branch count changed: {bundle_id}")
                for index, (old_branch, new_branch) in enumerate(
                    zip(old["branches"], new["branches"], strict=True)
                ):
                    if "bundle_id" in new_branch:
                        raise RuntimeError(
                            f"Patch lookup metadata leaked into branch: {bundle_id}/{index}"
                        )
                    kind = old_branch["action"]["kind"]
                    kind_counts[kind] += 1
                    old_after = member_bytes(before, f"{old_prefix}/{old_branch['after_file']}")
                    new_after = member_bytes(after, f"{new_prefix}/{new_branch['after_file']}")
                    if kind != "type":
                        if old_branch != new_branch or old_after != new_after:
                            raise RuntimeError(
                                f"Non-typing branch changed: {bundle_id}/{index}/{kind}"
                            )
                        non_type_count += 1
                        continue

                    old_action = old_branch["action"]
                    new_action = new_branch["action"]
                    old_text = old_action["text"]
                    new_text = new_action["text"]
                    if new_action["kind"] != "type":
                        raise RuntimeError(f"Typing action kind changed: {bundle_id}/{index}")
                    if (old_action.get("x"), old_action.get("y")) != (
                        new_action.get("x"),
                        new_action.get("y"),
                    ):
                        retargeted_type_actions += 1
                    if old_text == new_text or new_text.startswith("Synthetic "):
                        raise RuntimeError(
                            f"Typing text was not properly replaced: {bundle_id}/{index}"
                        )
                    if old_after == new_after:
                        unchanged_type_images.append(f"{bundle_id}/{index}")
                    else:
                        changed_type_images += 1
                    type_count += 1
                    texts.append(new_text)
                    hint = str(new_branch.get("qa", {}).get("element_hint") or "")
                    categories[text_category(hint)] += 1
                    split_counts[new["split"]] += 1

    result = {
        "shards": len(before_tars),
        "current_screens_byte_identical": current_count,
        "non_typing_branches_byte_and_metadata_identical": non_type_count,
        "typing_branches_replaced": type_count,
        "typing_future_images_changed": changed_type_images,
        "typing_future_images_unchanged": len(unchanged_type_images),
        "typing_actions_retargeted_to_unobscured_fields": retargeted_type_actions,
        "unchanged_typing_examples": unchanged_type_images[:20],
        "action_kinds": dict(kind_counts),
        "typing_by_split": dict(split_counts),
        "typing_categories": dict(categories),
        "typing_text": {
            "unique": len(set(texts)),
            "minimum_length": min(map(len, texts)),
            "maximum_length": max(map(len, texts)),
            "distinct_lengths": len({len(value) for value in texts}),
            "synthetic_prefix_count": sum(value.startswith("Synthetic ") for value in texts),
        },
    }
    result["passed"] = not unchanged_type_images
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a targeted typing-only regeneration")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.before.resolve(), args.after.resolve())
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
