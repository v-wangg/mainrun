#!/usr/bin/env python3
"""Download a wandb run's best checkpoint by run name.

Usage:
    python3 scripts/fetch_checkpoint.py "<run_name>"

Writes `mainrun/checkpoints/best.pt` (overwrites). On multiple matches, the
most recent run wins and the others are listed as a warning. Run names come
from the `--name` flag passed to `task train` (e.g. "lock-hp(batch_size=256)").
"""
import argparse
import os
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = "mainrun-sandbox"
CKPT_DEST_DIR = REPO_ROOT / "mainrun" / "checkpoints"


def fail(msg):
    print(f"fetch_checkpoint: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="wandb run name to fetch checkpoint for")
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        fail("WANDB_API_KEY not set — cannot fetch from wandb.")

    api = wandb.Api()
    entity = api.default_entity
    if not entity:
        fail("could not determine wandb entity. Run `wandb login` first.")

    matches = [r for r in api.runs(f"{entity}/{PROJECT}", order="-created_at")
               if r.name == args.name]

    if not matches:
        fail(f"no runs in {PROJECT} match name {args.name!r}.")

    run = matches[0]
    if len(matches) > 1:
        others = ", ".join(f"{r.id} ({r.created_at})" for r in matches[1:])
        print(
            f"warn: {len(matches)} runs match name {args.name!r}; "
            f"using most recent ({run.id}, {run.created_at}). "
            f"Older: {others}",
            file=sys.stderr,
        )

    print(f"run: {run.name} ({run.id})")
    print(f"url: {run.url}")
    print(f"state: {run.state}")

    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not model_artifacts:
        fail(f"run {run.id} has no model artifact.")
    artifact = model_artifacts[0]

    CKPT_DEST_DIR.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(CKPT_DEST_DIR))

    val_loss = artifact.metadata.get("val_loss") if artifact.metadata else None
    step = artifact.metadata.get("step") if artifact.metadata else None
    print(
        f"checkpoint → {CKPT_DEST_DIR.relative_to(REPO_ROOT)}/best.pt "
        f"(artifact: {artifact.name}, val_loss={val_loss}, step={step})"
    )


if __name__ == "__main__":
    main()
