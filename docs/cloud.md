# Remote training executor (vast.ai)

The vast.ai GPU box is the **remote training executor** for `mainrun`. Claude runs locally on the Mac and edits code there; the box only pulls and runs `task train`. This doc covers how the image is built, how to bring up an instance, and how the wandb integration works.

## Why this exists

A full 7-epoch training run on the canonical CPU dev container takes ~80 min. On the vast.ai GPU box, the same run takes ~2 min. The cloud path exists purely for iteration speed — the canonical CPU dev container in `.devcontainer/Dockerfile` is still the assessment-side runtime.

Claude used to run inside the box too, but tmux/iterm buffer issues on remote sessions made interactive work painful. Now Claude runs locally and uses git as the one-way sync.

## Workflow

```
[ local Mac (Claude) ]                       [ vast.ai box (run-only) ]
        edit
         │
         ▼
       commit ───────  git push ───────▶  git pull
                                              │
                                              ▼
                                          task train
                                              │
                                              ▼
                                          wandb cloud
                                              │
   task fetch-run  ◀──────────────────────────┘
        │
        ▼
   analyze, report.pdf, task submit
```

- **Edit only on local.** Never edit on the box. `scripts/checkpoint.mjs` runs `git add . && git commit` before every `task train`; if you've drifted on the box, that drift gets committed locally on the box and is invisible to local Claude until pushed back. After a clean `git pull` with no on-box edits, `checkpoint.mjs` is a no-op.
- **wandb is the canonical run store.** `mainrun/logs/*.log`, `mainrun/checkpoints/best.pt`, and `wandb/` are all gitignored. The structlog file ships via `wandb.save(..., policy="live")`; the best checkpoint ships via `wandb.Artifact`. Without wandb (no `WANDB_API_KEY` on the box, or import failure), the run produces nothing recoverable to local.
- **Pull before analyzing or submitting.** `task fetch-run` (locally) downloads the latest run's log + checkpoint into the gitignored locations train.py would have written them to, so `report.pdf` writers, `jq` queries, and `task submit` all see them.

## Components

| File | Role |
|------|------|
| `.devcontainer/Dockerfile.cloud` | Sibling of the canonical Dockerfile. Adds `openssh-server`, `gnupg`, `gh`; configures sshd for key-only root login; CMD is `cloud-start.sh`. |
| `.devcontainer/cloud-start.sh` | Container entrypoint. Injects `$PUBLIC_KEY` into `/root/.ssh/authorized_keys`, starts sshd, then `sleep infinity`. No auto-clone, no auto-train. |
| `.devcontainer/requirements.txt` | Adds `wandb==0.26.1` on top of the canonical stack. Both Dockerfiles install from this file, so wandb is present in both images. |
| `.dockerignore` | Trims build context (`.git`, `node_modules`, `mainrun/data`, `mainrun/logs`, `*.log`, `submission.zip`, `.DS_Store`). |
| `.gitignore` | Excludes `mainrun/checkpoints/`, `mainrun/logs/*.log`, and `wandb/` from commits. |
| Docker Hub | `vwang78/mainrun-cloud` — https://hub.docker.com/r/vwang78/mainrun-cloud |

`Dockerfile.cloud` mirrors the canonical `Dockerfile` line-for-line (the canonical one is owned by Maincode and frozen), then adds the cloud delta on top. If the canonical Dockerfile changes, mirror the change in `Dockerfile.cloud` and republish.

## Bring-up flow on vast.ai

1. Launch a vast.ai instance from the `vwang78/mainrun-cloud` image.
2. Pass two runtime env vars in vast.ai's "Docker options" / template config:
   - `PUBLIC_KEY=<your ssh public key>` — for SSH access.
   - `WANDB_API_KEY=<your wandb key>` — required for run telemetry. Without it, `train.py` will log a `wandb_disabled` event and Claude will not see the run.
3. SSH in as `root` on the port vast.ai forwards to 22.
4. First time only:
   ```bash
   git clone <repo-url> mainrun
   cd mainrun
   ```
5. Every subsequent run:
   ```bash
   cd mainrun
   git pull
   task train
   ```

The container does not auto-clone or auto-train. Everything runs from your SSH session. **Do not edit code on the box** — push from local instead.

## wandb integration

`mainrun/train.py` initializes wandb conditionally and uses it as the canonical run store:

- Import is wrapped in `try/except ImportError` — training runs even if wandb isn't installed.
- `wandb.init(mode="online" if WANDB_API_KEY else "disabled", project="mainrun-sandbox", config=hyperparams)`.
- When online, `wandb.save("./logs/mainrun.log", base_path=".", policy="live")` ships the structlog file continuously, so even on crash the partial log is recoverable.
- A structlog event announces the result on every run:
  - `wandb_init` with `mode`, `project`, `run_id`, `run_url` when the key is set.
  - `wandb_disabled` with reason and impact when it isn't. Treat this as a hard-stop signal — the run produces nothing recoverable.
- Per-step: `train/loss`, `train/lr`.
- Per-eval: `val/loss`, `val/epoch`.
- Best checkpoint logged as a `model` artifact with metadata (val loss, step, hyperparams) and aliases `["latest", "best"]`.
- `wandb.finish()` runs in the `finally` block so sessions close cleanly.

The frozen `evaluate()` function is not touched — wandb only consumes the loss values it returns.

## Pulling a run back to local

`scripts/fetch_run.py` pulls the canonical artifacts of a run into the gitignored local paths `train.py` would have written them to:

```bash
task fetch-run                  # latest run in mainrun-sandbox
task fetch-run -- <run_id>      # specific run
```

It writes:
- `mainrun/logs/mainrun.log` — from the wandb run file `logs/mainrun.log`.
- `mainrun/checkpoints/best.pt` — from the `model` artifact (alias `latest` or `best`).

Requires `WANDB_API_KEY` on the local Mac too. Run this before `task submit`, before writing `report.pdf`, or any time you need the raw log to grep/jq locally.

## Submission flow

```bash
# On the box (after edits + push from local):
ssh box; cd mainrun; git pull; task train

# Back on local:
task fetch-run        # populate mainrun/logs and mainrun/checkpoints from wandb
# write report.pdf from the now-local log + wandb plots, place at mainrun/report.pdf
task submit           # zips the repo with logs and checkpoint included
```

`scripts/submit.mjs` zips the repo excluding only `node_modules/` and `mainrun/data/`. Whatever else is on the local filesystem at submit time goes in — including the freshly-fetched log, checkpoint, and `report.pdf`.

## Differences from the local dev container

- Access is SSH-only; no VS Code Remote-Containers integration.
- `/root/.mainrun` is still written, so `mainrun/utils.py`'s dev-container check is satisfied.
- Root login is enabled with public-key auth only (`PermitRootLogin yes`, `PasswordAuthentication no`).
- The image bundles the GitHub CLI (`gh`).

## Rebuilding and pushing the image

When `requirements.txt` changes (new dep) or the canonical `Dockerfile` changes (mirror it):

```bash
docker build -f .devcontainer/Dockerfile.cloud -t vwang78/mainrun-cloud:latest .
docker push vwang78/mainrun-cloud:latest
```

Then update vast.ai templates to pull the new tag.
