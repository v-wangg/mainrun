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
       commit  ──────  git push  ──────▶  git pull
                                              │
                                              ▼
                                          task train
                                              │
                                              ▼
                                          wandb cloud  ◀── Claude reads here
```

- **Edit only on local.** Never edit on the box. `scripts/checkpoint.mjs` runs `git add . && git commit` before every `task train`; if you've drifted on the box, that drift gets committed locally on the box and is invisible to local Claude until pushed back. After a clean `git pull` with no on-box edits, `checkpoint.mjs` is a no-op.
- **wandb is the feedback channel.** `mainrun/logs/*.log`, `mainrun/checkpoints/best.pt`, and `wandb/` are all gitignored. Without wandb, the run is invisible to Claude.

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

`mainrun/train.py` initializes wandb conditionally:

- Import is wrapped in `try/except ImportError` — training runs even if wandb isn't installed.
- `wandb.init(mode="online" if WANDB_API_KEY else "disabled", project="mainrun-sandbox", config=hyperparams)`.
- A structlog event announces the result on every run:
  - `wandb_init` with `mode`, `project`, `run_id`, `run_url` when the key is set.
  - `wandb_disabled` with reason and impact when it isn't. Treat this as a hard signal that Claude has no visibility into the run.
- Per-step: `train/loss`, `train/lr`.
- Per-eval: `val/loss`, `val/epoch`.
- Best checkpoint logged as a `model` artifact with metadata (val loss, step, hyperparams) and aliases `["latest", "best"]`.
- `wandb.finish()` runs in the `finally` block so sessions close cleanly.

The frozen `evaluate()` function is not touched — wandb only consumes the loss values it returns.

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
