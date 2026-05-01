# Cloud / vast.ai runtime

The default place to run `mainrun` is a vast.ai GPU box built from `.devcontainer/Dockerfile.cloud`. This doc describes how that image is built, how to bring up an instance, and how the wandb integration works.

## Why this exists

A full 7-epoch training run on the canonical CPU dev container takes ~80 min. On the vast.ai GPU box, the same run takes ~2 min. The cloud path exists purely for iteration speed — the canonical CPU dev container in `.devcontainer/Dockerfile` is still the assessment-side runtime; this is where development happens.

## Components

| File | Role |
|------|------|
| `.devcontainer/Dockerfile.cloud` | Sibling of the canonical Dockerfile. Adds `openssh-server`, `gnupg`, `gh`, the Claude Code CLI; configures sshd for key-only root login; CMD is `cloud-start.sh`. |
| `.devcontainer/cloud-start.sh` | Container entrypoint. Injects `$PUBLIC_KEY` into `/root/.ssh/authorized_keys`, starts sshd, then `sleep infinity`. No auto-clone, no auto-train. |
| `.devcontainer/requirements.txt` | Adds `wandb==0.26.1` on top of the canonical stack. Both Dockerfiles install from this file, so wandb is present in both images. |
| `.dockerignore` | Trims build context (`.git`, `node_modules`, `mainrun/data`, `mainrun/logs`, `*.log`, `submission.zip`, `.DS_Store`). |
| `.gitignore` | Excludes `mainrun/checkpoints/` and `wandb/` from commits. |
| Docker Hub | `vwang78/mainrun-cloud` — https://hub.docker.com/r/vwang78/mainrun-cloud |

`Dockerfile.cloud` mirrors the canonical `Dockerfile` line-for-line (the canonical one is owned by Maincode and frozen), then adds the cloud delta on top. If the canonical Dockerfile changes, mirror the change in `Dockerfile.cloud` and republish.

## Bring-up flow on vast.ai

1. Launch a vast.ai instance from the `vwang78/mainrun-cloud` image.
2. Pass `PUBLIC_KEY=<your ssh public key>` as a runtime env var (vast.ai's "Docker options" / template config).
3. SSH in as `root` on the port vast.ai forwards to 22.
4. Inside the container:
   ```bash
   git clone <repo-url> mainrun
   cd mainrun
   export WANDB_API_KEY=...   # optional, enables wandb logging
   task train
   ```

The container does not auto-clone or auto-train. Everything runs from your SSH session.

## wandb integration

`mainrun/train.py` initializes wandb conditionally:

- Import is wrapped in `try/except ImportError` — training runs even if wandb isn't installed.
- `wandb.init(mode="online" if WANDB_API_KEY else "disabled", project="mainrun-sandbox", config=hyperparams)`.
- Per-step: `train/loss`, `train/lr`.
- Per-eval: `val/loss`, `val/epoch`.
- Best checkpoint logged as a `model` artifact with metadata (val loss, step, hyperparams) and aliases `["latest", "best"]`.
- `wandb.finish()` runs in the `finally` block so sessions close cleanly.

The frozen `evaluate()` function is not touched — wandb only consumes the loss values it returns.

## Differences from the local dev container

- Access is SSH-only; no VS Code Remote-Containers integration.
- `/root/.mainrun` is still written (line 50 of `Dockerfile.cloud`), so `mainrun/utils.py`'s dev-container check is satisfied.
- Root login is enabled with public-key auth only (`PermitRootLogin yes`, `PasswordAuthentication no`).
- The image bundles the GitHub CLI (`gh`) and the Claude Code CLI (`claude`).

## Rebuilding and pushing the image

When `requirements.txt` changes (new dep) or the canonical `Dockerfile` changes (mirror it):

```bash
docker build -f .devcontainer/Dockerfile.cloud -t vwang78/mainrun-cloud:latest .
docker push vwang78/mainrun-cloud:latest
```

Then update vast.ai templates to pull the new tag.
