#!/usr/bin/env python3
"""Pull all wandb run logs into mainrun/logs/ as per-run files.

Default (no args): reconcile every run in `mainrun-sandbox` against the local
log directory. Each run lands as `{sanitized_name}-{run_id}.log`. Already-fetched
runs are skipped; runs renamed on the wandb web UI get their local file renamed
to match (run_id is the stable key).

Use `task fetch-checkpoint -- <run_name>` to download a run's best.pt.
"""
import os
import re
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = "mainrun-sandbox"
LOG_REL_IN_RUN = "logs/mainrun.log"
LOG_DIR = REPO_ROOT / "mainrun" / "logs"
DOWNLOAD_ROOT = REPO_ROOT / "mainrun"  # so wandb writes to LOG_DIR/mainrun.log


def fail(msg):
    print(f"fetch_run: {msg}", file=sys.stderr)
    sys.exit(1)


def sanitize_name(name: str) -> str:
    if not name:
        return "run"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "run"


def index_local_logs() -> dict[str, Path]:
    """Map run_id -> local log path for files matching `*-<run_id>.log`."""
    index: dict[str, Path] = {}
    if not LOG_DIR.exists():
        return index
    for path in LOG_DIR.glob("*-*.log"):
        # run_id is the substring after the final '-' before '.log'
        run_id = path.stem.rsplit("-", 1)[-1]
        if run_id:
            index[run_id] = path
    return index


def main():
    if not os.environ.get("WANDB_API_KEY"):
        fail("WANDB_API_KEY not set — cannot fetch from wandb.")

    api = wandb.Api()
    entity = api.default_entity
    if not entity:
        fail("could not determine wandb entity. Run `wandb login` first.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    local = index_local_logs()
    runs = api.runs(f"{entity}/{PROJECT}", order="-created_at")

    counts = {"fetched": 0, "renamed": 0, "up_to_date": 0, "skipped": 0}

    for run in runs:
        target_name = f"{sanitize_name(run.name)}-{run.id}.log"
        target = LOG_DIR / target_name
        existing = local.get(run.id)

        if existing is not None:
            if existing.name == target_name:
                print(f"up-to-date: {target_name}")
                counts["up_to_date"] += 1
            else:
                existing.rename(target)
                print(f"renamed: {existing.name} → {target_name}")
                counts["renamed"] += 1
            continue

        try:
            run.file(LOG_REL_IN_RUN).download(root=str(DOWNLOAD_ROOT), replace=True)
        except Exception as exc:
            print(f"skip {run.id} ({run.name}): {exc}", file=sys.stderr)
            counts["skipped"] += 1
            continue

        downloaded = LOG_DIR / "mainrun.log"
        if not downloaded.exists():
            print(f"skip {run.id} ({run.name}): downloaded file missing", file=sys.stderr)
            counts["skipped"] += 1
            continue

        downloaded.replace(target)
        print(f"fetched: {target_name}")
        counts["fetched"] += 1

    total = sum(counts.values())
    print(
        f"\nsummary: {total} runs — "
        f"{counts['fetched']} fetched, "
        f"{counts['renamed']} renamed, "
        f"{counts['up_to_date']} up-to-date, "
        f"{counts['skipped']} skipped"
    )


if __name__ == "__main__":
    main()
