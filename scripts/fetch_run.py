#!/usr/bin/env python3
"""Pull a training run's structlog file + best checkpoint from wandb to local.

Usage:
    python3 scripts/fetch_run.py            # latest run in mainrun-sandbox
    python3 scripts/fetch_run.py <run_id>   # specific run id

Writes:
    mainrun/logs/mainrun.log
    mainrun/checkpoints/best.pt

Designed to be run on the local Mac before `task submit`. wandb is the
canonical store for both files; they are gitignored locally.
"""
import argparse
import os
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = "mainrun-sandbox"
LOG_REL_IN_RUN = "logs/mainrun.log"
LOG_DEST = REPO_ROOT / "mainrun" / "logs" / "mainrun.log"
CKPT_DEST_DIR = REPO_ROOT / "mainrun" / "checkpoints"


def fail(msg):
    print(f"fetch_run: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", nargs="?", default=None,
                        help="wandb run id (default: latest run in mainrun-sandbox)")
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        fail("WANDB_API_KEY not set — cannot fetch from wandb.")

    api = wandb.Api()
    entity = api.default_entity
    if not entity:
        fail("could not determine wandb entity. Run `wandb login` first.")

    if args.run_id:
        run = api.run(f"{entity}/{PROJECT}/{args.run_id}")
    else:
        runs = api.runs(f"{entity}/{PROJECT}", per_page=1, order="-created_at")
        run = next(iter(runs), None)
        if run is None:
            fail(f"no runs found in {entity}/{PROJECT}.")

    print(f"run: {run.name} ({run.id})")
    print(f"url: {run.url}")
    print(f"state: {run.state}")

    LOG_DEST.parent.mkdir(parents=True, exist_ok=True)
    log_root = REPO_ROOT / "mainrun"
    try:
        run.file(LOG_REL_IN_RUN).download(root=str(log_root), replace=True)
        print(f"log → {LOG_DEST.relative_to(REPO_ROOT)}")
    except Exception as exc:
        print(f"warn: could not download log file '{LOG_REL_IN_RUN}': {exc}", file=sys.stderr)

    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not model_artifacts:
        print("warn: no model artifact attached to this run.", file=sys.stderr)
        return
    artifact = model_artifacts[0]
    CKPT_DEST_DIR.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(CKPT_DEST_DIR))
    print(f"checkpoint → {CKPT_DEST_DIR.relative_to(REPO_ROOT)}/best.pt (artifact: {artifact.name})")


if __name__ == "__main__":
    main()
