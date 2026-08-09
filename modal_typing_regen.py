from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from cua_jepa.shards import add_bytes, file_sha256
from modal_app import (
    CONFIG,
    EXPECTED_PACKAGE_NAMES,
    REMOTE_DATA_ROOT,
    _download_volume_file,
    _ensure_app_state_compatibility,
    _load_remote_catalog,
    _local_bytes,
    _reserve_local_port,
    _stop_process_group,
    _wait_for_server,
    image,
    volume,
)

typing_app = modal.App("cua-jepa-typing-regeneration")
if modal.is_local():
    typing_image = image.add_local_file(
        Path(__file__).parent / "modal_app.py", "/root/modal_app.py", copy=True
    )
else:
    typing_image = image


@typing_app.function(
    image=typing_image,
    volumes={REMOTE_DATA_ROOT: volume},
    cpu=1.0,
    memory=2048,
    max_containers=24,
    timeout=20 * 60,
    retries=0,
)
def regenerate_typing_shard(job: dict[str, Any], run_id: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    from cua_jepa.actions import varied_type_text
    from cua_jepa.collector import BundleCollector
    from cua_jepa.state_variants import make_state_variant, normalize_state_for_app

    split = job["split"]
    app_name = job["app"]
    shard_name = job["shard_name"]
    records = job["records"]
    patch_name = shard_name.replace(".tar", ".typing.tar")
    remote_dir = REMOTE_DATA_ROOT / run_id / "typing" / split / app_name
    remote_tar = remote_dir / patch_name
    remote_json = remote_tar.with_suffix(".json")
    remote_sha = remote_tar.with_suffix(".sha256")
    if remote_tar.exists() and remote_json.exists() and remote_sha.exists():
        existing = json.loads(remote_json.read_text(encoding="utf-8"))
        existing["volume_path"] = str(remote_tar.relative_to(REMOTE_DATA_ROOT))
        existing["resumed"] = True
        return existing

    app_dir = Path("/opt/cua-jepa/apps") / app_name
    _ensure_app_state_compatibility(app_dir, app_name)
    package_name = json.loads((app_dir / "package.json").read_text(encoding="utf-8")).get("name")
    if package_name != EXPECTED_PACKAGE_NAMES[app_name]:
        raise RuntimeError(f"Wrong app package for {app_name}: {package_name!r}")
    states = {entry["source_task_id"]: entry for entry in _load_remote_catalog(app_name)}

    with tempfile.TemporaryDirectory(prefix="cua-jepa-typing-") as temporary:
        temp_root = Path(temporary)
        server_log = temp_root / "vite.log"
        marker_dir = app_dir / "public"
        marker_dir.mkdir(exist_ok=True)
        (marker_dir / "__cua_jepa_app_id.txt").write_text(app_name, encoding="utf-8")
        port = _reserve_local_port()
        base_url = f"http://127.0.0.1:{port}"
        with server_log.open("wb") as log_handle:
            server = subprocess.Popen(
                [
                    "node",
                    "node_modules/vite/bin/vite.js",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=app_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            _wait_for_server(base_url, server, server_log, app_name)
            temporary_tar = temp_root / f"{patch_name}.partial"
            final_tar = temp_root / patch_name
            patches: list[dict[str, Any]] = []
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                collector = BundleCollector(
                    browser=browser,
                    base_url=base_url,
                    app=app_name,
                    split=split,
                    viewport_width=CONFIG.viewport.width,
                    viewport_height=CONFIG.viewport.height,
                    actions_per_bundle=CONFIG.actions_per_bundle,
                    maximum_action_candidates=(CONFIG.quality.maximum_action_candidates_per_bundle),
                    maximum_branch_reset_attempts=(CONFIG.quality.maximum_branch_reset_attempts),
                    maximum_reset_changed_fraction=(
                        CONFIG.quality.maximum_reset_changed_pixel_fraction
                    ),
                    allow_warmups=(app_name not in CONFIG.quality.warmup_disabled_apps),
                    minimum_changed_fraction=CONFIG.quality.minimum_changed_pixel_fraction,
                    maximum_changed_fraction=CONFIG.quality.maximum_changed_pixel_fraction,
                )
                with tarfile.open(temporary_tar, mode="w") as patch_tar:
                    for record in records:
                        state_entry = states.get(record["source_task_id"])
                        if state_entry is None:
                            raise RuntimeError(f"Missing source state: {record['source_task_id']}")
                        initial_state = normalize_state_for_app(
                            app_name,
                            make_state_variant(state_entry["state"], int(record["seed"])),
                        )
                        for branch_index, stored_branch in enumerate(record["branches"]):
                            if stored_branch["action"]["kind"] != "type":
                                continue
                            try:
                                artifact = collector.regenerate_type_branch(
                                    record,
                                    initial_state,
                                    branch_index,
                                    lambda target_hint: varied_type_text(
                                        app_name,
                                        target_hint,
                                        int(record["seed"]),
                                        record["bundle_id"],
                                        branch_index,
                                    ),
                                    record["_current_webp"],
                                )
                            except Exception as exc:
                                raise RuntimeError(
                                    f"{split}/{app_name}/{shard_name}/"
                                    f"{record['bundle_id']}/{branch_index}: {exc}"
                                ) from exc
                            member_name = f"{record['bundle_id']}/{stored_branch['after_file']}"
                            add_bytes(patch_tar, member_name, artifact.after_webp)
                            patches.append(
                                {
                                    "bundle_id": record["bundle_id"],
                                    "branch_index": branch_index,
                                    "action": artifact.action,
                                    "after_file": stored_branch["after_file"],
                                    "after_sha256": artifact.after_sha256,
                                    "changed_pixel_fraction": artifact.changed_pixel_fraction,
                                    "qa": {
                                        "element_hint": artifact.element_hint,
                                        "branch_start_render_sha256": (
                                            artifact.branch_start_render_sha256
                                        ),
                                        "branch_start_changed_pixel_fraction": (
                                            artifact.branch_start_changed_pixel_fraction
                                        ),
                                        "after_render_sha256": (artifact.after_render_sha256),
                                        "state_diff_paths": list(artifact.state_diff_paths),
                                        "state_diff_bytes": artifact.state_diff_bytes,
                                    },
                                }
                            )
                    patch_manifest = {
                        "schema_version": 1,
                        "source_shard": shard_name,
                        "split": split,
                        "app": app_name,
                        "bundle_count": len(records),
                        "typing_patches": len(patches),
                        "patches": patches,
                    }
                    add_bytes(
                        patch_tar,
                        "patches.json",
                        json.dumps(patch_manifest, separators=(",", ":")).encode("utf-8"),
                    )
                browser.close()
            os.replace(temporary_tar, final_tar)
            result = {key: value for key, value in patch_manifest.items() if key != "patches"}
            result["sha256"] = file_sha256(final_tar)
            result["bytes"] = final_tar.stat().st_size
            final_tar.with_suffix(".json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            final_tar.with_suffix(".sha256").write_text(
                f"{result['sha256']}  {final_tar.name}\n", encoding="ascii"
            )
            remote_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_tar, remote_tar)
            shutil.copy2(final_tar.with_suffix(".json"), remote_json)
            shutil.copy2(final_tar.with_suffix(".sha256"), remote_sha)
            volume.commit()
            result["volume_path"] = str(remote_tar.relative_to(REMOTE_DATA_ROOT))
            result["resumed"] = False
            return result
        finally:
            _stop_process_group(server)


def load_jobs(
    source_root: Path,
    only_app: str = "",
    only_shard: str = "",
    shard_limit: int = 0,
) -> list[dict]:
    jobs: list[dict[str, Any]] = []
    for source_tar in sorted(source_root.rglob("*.tar")):
        relative = source_tar.relative_to(source_root)
        if len(relative.parts) != 3:
            continue
        split, app_name, shard_name = relative.parts
        if only_app and app_name != only_app:
            continue
        if only_shard and shard_name != only_shard:
            continue
        records = []
        with tarfile.open(source_tar, mode="r") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith("/bundle.json"):
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise RuntimeError(f"Cannot read {member.name}")
                    record = json.load(payload)
                    if any(branch["action"]["kind"] == "type" for branch in record["branches"]):
                        prefix = str(PurePosixPath(member.name).parent)
                        current_payload = archive.extractfile(f"{prefix}/{record['current_file']}")
                        if current_payload is None:
                            raise RuntimeError(
                                f"Cannot read current image for {record['bundle_id']}"
                            )
                        record["_current_webp"] = current_payload.read()
                        records.append(record)
        if records:
            jobs.append(
                {
                    "split": split,
                    "app": app_name,
                    "shard_name": shard_name,
                    "records": records,
                }
            )
        if shard_limit and len(jobs) >= shard_limit:
            break
    return jobs


def run_jobs(
    jobs: list[dict[str, Any]], run_id: str, local_root: Path, deadline: float
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    batch_size = CONFIG.maximum_remote_workers * 4
    for offset in range(0, len(jobs), batch_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("Ten-hour typing-regeneration limit reached")
        batch = jobs[offset : offset + batch_size]
        for result in regenerate_typing_shard.starmap(
            [(job, run_id) for job in batch],
            order_outputs=False,
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                errors.append(repr(result))
            else:
                results.append(result)
        if errors:
            error_path = local_root / run_id / "typing_errors.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
            raise RuntimeError(f"{len(errors)} typing shard(s) failed; see {error_path}")
        print(
            f"Typing regeneration batch {offset // batch_size + 1}/"
            f"{math.ceil(len(jobs) / batch_size)} completed"
        )

    for index, result in enumerate(results, start=1):
        relative = Path(result["volume_path"])
        local_tar = local_root / relative
        for suffix in (".json", ".sha256", ".tar"):
            remote_file = str(relative.with_suffix(suffix)).replace("\\", "/")
            local_file = local_tar.with_suffix(suffix)
            _download_volume_file(remote_file, local_file)
        if file_sha256(local_tar) != result["sha256"]:
            raise RuntimeError(f"Checksum mismatch after download: {relative}")
        used = _local_bytes(local_root / run_id)
        if used > CONFIG.maximum_local_bytes:
            raise RuntimeError("Local data limit exceeded")
        progress = {
            "run_id": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_shards": index,
            "total_shards": len(results),
            "typing_patches": sum(item["typing_patches"] for item in results[:index]),
            "local_bytes": used,
        }
        (local_root / run_id / "typing_progress.json").write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{index}/{len(results)}] downloaded {relative.name}")
    return results


@typing_app.local_entrypoint()
def main(
    source_root: str = "data/synthetic/clean-20260808-v5/full",
    run_id: str = "typing-20260808-v1",
    local_root: str = "data/synthetic",
    only_app: str = "",
    only_shard: str = "",
    shard_limit: int = 0,
) -> None:
    source = Path(source_root).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Missing source dataset: {source}")
    jobs = load_jobs(
        source,
        only_app=only_app,
        only_shard=only_shard,
        shard_limit=shard_limit,
    )
    if not jobs:
        raise RuntimeError("No typing jobs found")
    deadline = time.monotonic() + CONFIG.maximum_runtime_hours * 3600
    results = run_jobs(jobs, run_id, Path(local_root).resolve(), deadline)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "typing_shards": len(results),
                "typing_patches": sum(item["typing_patches"] for item in results),
            },
            indent=2,
        )
    )
