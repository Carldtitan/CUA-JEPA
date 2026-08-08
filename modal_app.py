from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from cua_jepa.config import load_config
from cua_jepa.shards import (
    ShardSpec,
    add_bundle,
    build_full_specs,
    build_pilot_specs,
    deterministic_index,
    file_sha256,
    finalize_shard,
)


ROOT = Path(__file__).parent
LOCAL_CONFIG_PATH = ROOT / "configs" / "dataset_v1.json"
REMOTE_CONFIG_FILE = "/opt/cua-jepa/dataset_v1.json"
CATALOG_PATH = ROOT / ".cache" / "state_catalog.json"
HUB_APPS_PATH = ROOT / ".vendor" / "CUA-Gym" / "hub" / "websites"
CONFIG = load_config(
    LOCAL_CONFIG_PATH if modal.is_local() else Path(REMOTE_CONFIG_FILE)
)
ALL_APPS = tuple(
    sorted({app for split in CONFIG.splits.values() for app in split.apps})
)

APP_NAME = "cua-jepa-synthetic-data"
VOLUME_NAME = "cua-jepa-synthetic-v1"
REMOTE_DATA_ROOT = Path("/dataset")


def _require_local_inputs() -> None:
    missing = [path for path in (CATALOG_PATH, HUB_APPS_PATH) if not path.exists()]
    missing.extend(HUB_APPS_PATH / app for app in ALL_APPS if not (HUB_APPS_PATH / app).exists())
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Missing collection inputs:\n{joined}")


if modal.is_local():
    _require_local_inputs()
    image = (
        modal.Image.from_registry(
            "mcr.microsoft.com/playwright:v1.59.0-noble", add_python="3.12"
        )
        .pip_install("pillow>=10,<13", "playwright==1.59.0", "requests>=2.31,<3")
    )
    for app_name in ALL_APPS:
        image = image.add_local_dir(
            HUB_APPS_PATH / app_name,
            f"/opt/cua-jepa/apps/{app_name}",
            copy=True,
            ignore=["node_modules", ".mock-states", "dist"],
        )

    image = image.run_commands(
        *[
            f"cd /opt/cua-jepa/apps/{app_name} && npm ci --no-audit --no-fund"
            for app_name in ALL_APPS
        ]
    )
    image = (
        image.add_local_python_source("cua_jepa", copy=True)
        .add_local_file(LOCAL_CONFIG_PATH, REMOTE_CONFIG_FILE, copy=True)
        .add_local_file(CATALOG_PATH, "/opt/cua-jepa/state_catalog.json", copy=True)
    )
else:
    image = modal.Image.debian_slim()

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def _wait_for_server(url: str, process: subprocess.Popen, log_path: Path) -> None:
    import requests

    for _ in range(90):
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Mock server exited early:\n{tail}")
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Mock server did not become ready: {url}")


def _load_remote_catalog(app_name: str) -> list[dict[str, Any]]:
    catalog = json.loads(
        Path("/opt/cua-jepa/state_catalog.json").read_text(encoding="utf-8")
    )
    states = catalog.get(app_name, [])
    if not states:
        raise RuntimeError(f"No states available for {app_name}")
    return states


@app.function(
    image=image,
    volumes={REMOTE_DATA_ROOT: volume},
    cpu=1.0,
    memory=2048,
    max_containers=12,
    timeout=20 * 60,
    retries=0,
)
def generate_shard(spec_value: dict[str, Any], run_id: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    from cua_jepa.collector import BundleCollector, BundleRejected
    from cua_jepa.state_variants import make_state_variant

    spec = ShardSpec(**spec_value)
    remote_dir = REMOTE_DATA_ROOT / run_id / spec.phase / spec.split / spec.app
    remote_tar = remote_dir / spec.name
    remote_json = remote_tar.with_suffix(".json")
    remote_sha = remote_tar.with_suffix(".sha256")
    if remote_tar.exists() and remote_json.exists() and remote_sha.exists():
        existing = json.loads(remote_json.read_text(encoding="utf-8"))
        existing["volume_path"] = str(remote_tar.relative_to(REMOTE_DATA_ROOT))
        existing["resumed"] = True
        return existing

    app_dir = Path("/opt/cua-jepa/apps") / spec.app
    states = _load_remote_catalog(spec.app)
    with tempfile.TemporaryDirectory(prefix="cua-jepa-") as temporary:
        temp_root = Path(temporary)
        server_log = temp_root / "vite.log"
        with server_log.open("wb") as log_handle:
            server = subprocess.Popen(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "5173",
                ],
                cwd=app_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_for_server("http://127.0.0.1:5173", server, server_log)
            temporary_tar = temp_root / f"{spec.name}.partial"
            final_tar = temp_root / spec.name
            metadata: list[dict[str, Any]] = []
            failures: Counter[str] = Counter()
            attempts = 0
            maximum_attempts = (
                spec.bundle_count * CONFIG.quality.maximum_generation_attempts_per_bundle
            )

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                collector = BundleCollector(
                    browser=browser,
                    base_url="http://127.0.0.1:5173",
                    app=spec.app,
                    split=spec.split,
                    viewport_width=CONFIG.viewport.width,
                    viewport_height=CONFIG.viewport.height,
                    actions_per_bundle=CONFIG.actions_per_bundle,
                    minimum_changed_fraction=CONFIG.quality.minimum_changed_pixel_fraction,
                )
                with tarfile.open(temporary_tar, mode="w") as tar:
                    while len(metadata) < spec.bundle_count and attempts < maximum_attempts:
                        accepted_index = spec.start_index + len(metadata)
                        candidate_seed = (
                            CONFIG.seed
                            + spec.shard_index * 1_000_003
                            + accepted_index * 101
                            + attempts
                        )
                        state_index = deterministic_index(
                            CONFIG.seed, spec.app, candidate_seed, len(states)
                        )
                        state_entry = states[state_index]
                        state = make_state_variant(state_entry["state"], candidate_seed)
                        bundle_id = (
                            f"{spec.phase}-{spec.split}-{spec.app}-"
                            f"{accepted_index:06d}"
                        )
                        attempts += 1
                        try:
                            artifact = collector.generate(
                                bundle_id=bundle_id,
                                seed=candidate_seed,
                                source_task_id=state_entry["source_task_id"],
                                initial_state=state,
                            )
                        except BundleRejected as exc:
                            failures[str(exc)] += 1
                            continue
                        metadata.append(add_bundle(tar, artifact))
                browser.close()

            if len(metadata) != spec.bundle_count:
                raise RuntimeError(
                    f"{spec.phase}/{spec.app}/shard-{spec.shard_index:04d}: "
                    f"accepted {len(metadata)}/{spec.bundle_count} bundles; "
                    f"failures={dict(failures)}"
                )

            manifest = finalize_shard(
                temporary_path=temporary_tar,
                final_path=final_tar,
                spec=spec,
                metadata=metadata,
                failures=failures,
                attempts=attempts,
            )
            remote_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_tar, remote_tar)
            shutil.copy2(final_tar.with_suffix(".json"), remote_json)
            shutil.copy2(final_tar.with_suffix(".sha256"), remote_sha)
            volume.commit()
            manifest["volume_path"] = str(remote_tar.relative_to(REMOTE_DATA_ROOT))
            manifest["resumed"] = False
            return manifest
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


def _download_volume_file(
    remote_path: str, local_path: Path, maximum_attempts: int = 6
) -> None:
    temporary_path = local_path.with_suffix(local_path.suffix + ".partial")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, maximum_attempts + 1):
        try:
            with temporary_path.open("wb") as handle:
                for chunk in volume.read_file(remote_path):
                    handle.write(chunk)
            temporary_path.replace(local_path)
            return
        except Exception:
            temporary_path.unlink(missing_ok=True)
            if attempt == maximum_attempts:
                raise
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"Download failed for {remote_path}; retrying in {delay}s "
                f"({attempt}/{maximum_attempts})"
            )
            time.sleep(delay)


def _local_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _run_specs(
    specs: list[ShardSpec], run_id: str, local_root: Path, deadline: float
) -> list[dict]:
    results: list[dict] = []
    errors: list[str] = []
    batch_size = CONFIG.maximum_remote_workers
    for offset in range(0, len(specs), batch_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("Ten-hour generation limit reached")
        batch = specs[offset : offset + batch_size]
        jobs = [(asdict(spec), run_id) for spec in batch]
        for result in generate_shard.starmap(
            jobs, order_outputs=False, return_exceptions=True
        ):
            if isinstance(result, BaseException):
                errors.append(repr(result))
                continue
            remote_tar = result["volume_path"]
            relative = Path(remote_tar)
            local_tar = local_root / relative
            for suffix in (".json", ".sha256"):
                remote_file = str(relative.with_suffix(suffix)).replace("\\", "/")
                local_file = local_tar.with_suffix(suffix)
                _download_volume_file(remote_file, local_file)
            if not local_tar.exists() or file_sha256(local_tar) != result["sha256"]:
                _download_volume_file(remote_tar, local_tar)
            actual_sha256 = file_sha256(local_tar)
            if actual_sha256 != result["sha256"]:
                local_tar.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Checksum mismatch for {relative}: "
                    f"{actual_sha256} != {result['sha256']}"
                )
            results.append(result)

            run_dir = local_root / run_id
            used = _local_bytes(run_dir)
            if used > CONFIG.maximum_local_bytes:
                raise RuntimeError(
                    f"Local data limit exceeded: {used} > {CONFIG.maximum_local_bytes}"
                )
            report = {
                "run_id": run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed_shards": len(results),
                "accepted_bundles": sum(item["accepted_bundles"] for item in results),
                "transitions": sum(item["transitions"] for item in results),
                "local_bytes": used,
                "shards": results,
            }
            report_path = run_dir / "progress.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                f"[{len(results)}/{len(specs)}] {relative.name}: "
                f"{report['accepted_bundles']} bundles, {used / 2**30:.2f} GiB local"
            )
        if errors:
            run_dir = local_root / run_id
            error_report = {
                "run_id": run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "failed_shards": len(errors),
                "errors": errors,
            }
            error_path = run_dir / "errors.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps(error_report, indent=2), encoding="utf-8")
            raise RuntimeError(
                f"{len(errors)} shard(s) failed; see {error_path}: "
                + " | ".join(errors)
            )
    return results


def _pilot_passes(results: list[dict]) -> tuple[bool, dict[str, Any]]:
    accepted = sum(result["accepted_bundles"] for result in results)
    attempts = sum(result["attempts"] for result in results)
    failures = Counter()
    for result in results:
        failures.update(result["failures"])
    acceptance_rate = accepted / attempts if attempts else 0.0
    reset_failures = failures.get("nonidentical_branch_start", 0) + failures.get(
        "branch_start_hash_mismatch", 0
    )
    identical_reset_rate = 1.0 - (reset_failures / attempts if attempts else 1.0)
    report = {
        "accepted_bundles": accepted,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "identical_reset_rate": identical_reset_rate,
        "failures": dict(failures),
    }
    passed = (
        accepted == CONFIG.pilot_bundles
        and acceptance_rate >= CONFIG.quality.minimum_bundle_acceptance_rate
        and identical_reset_rate >= CONFIG.quality.minimum_identical_reset_rate
    )
    return passed, report


@app.local_entrypoint()
def main(
    phase: str = "pilot",
    auto_continue: bool = False,
    run_id: str = "",
    local_root: str = "data/synthetic",
) -> None:
    if phase not in {"pilot", "full"}:
        raise ValueError("phase must be pilot or full")
    if not run_id:
        run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    destination = Path(local_root).resolve()
    started = time.monotonic()
    deadline = started + CONFIG.maximum_runtime_hours * 3600

    if phase == "pilot":
        pilot_results = _run_specs(build_pilot_specs(CONFIG), run_id, destination, deadline)
        passed, gate = _pilot_passes(pilot_results)
        gate_path = destination / run_id / "pilot_gate.json"
        gate_path.write_text(json.dumps({"passed": passed, **gate}, indent=2), encoding="utf-8")
        print(json.dumps({"pilot_passed": passed, **gate}, indent=2))
        if not passed:
            raise RuntimeError("Pilot quality gate failed; full generation was not launched")
        if auto_continue:
            elapsed_hours = (time.monotonic() - started) / 3600
            if elapsed_hours >= CONFIG.maximum_runtime_hours:
                raise TimeoutError("Runtime limit reached before full generation")
            _run_specs(build_full_specs(CONFIG), run_id, destination, deadline)
    else:
        _run_specs(build_full_specs(CONFIG), run_id, destination, deadline)
