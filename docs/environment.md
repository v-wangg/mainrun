# Environment

## Dev container — required

`mainrun/utils.py` aborts on startup unless `/root/.mainrun` exists. The dev container (`.devcontainer/`) is the only supported runtime. From the task spec:

> Running outside devcontainer = broken submission & metrics

Launch via VS Code "Reopen in Container" (see `README.md`).

## Task commands

The Taskfile (`Taskfile.yml`) is the canonical entry point. Python and `npx zx` are called internally — don't invoke `train.py` directly.

| Command | What it does |
|---------|--------------|
| `task download` | Download the HN dataset to `./data/` (HuggingFace cache). |
| `task checkpoint` | `git add . && git commit -m "Mainrun auto checkpoint"`. Dep of `train` and `submit`. |
| `task train` | Full pipeline: checkpoint → download → `python3 train.py`. |
| `task submit` | Checkpoint → `npx zx submit.mjs` (zip + upload). |

## Python stack (`.devcontainer/requirements.txt`)

- Python 3.10
- `torch==2.7.1`, `datasets==4.0.0`, `tokenizers==0.21.2`
- `tqdm==4.67.1`, `structlog==24.4.0`, `matplotlib==3.10.3`

## GPU

`.devcontainer/devcontainer.json` declares `"gpu": "optional"`. The baseline run in `mainrun/logs/baseline.log` was on CPU (`device: "cpu"`) — ~30 min end-to-end. With GPU, the same run is minutes.

`train.py` picks `"cuda" if torch.cuda.is_available() else "cpu"`. No MPS branch. The dev container images are Linux + optional CUDA, so Apple GPU isn't relevant inside the container.

For the remote training executor (vast.ai + `vwang78/mainrun-cloud` + wandb), see [`docs/cloud.md`](./cloud.md). That box runs `task train` only; Claude edits locally and pushes via git. This file documents the canonical CPU dev container that the assessment uses.

## Logs

- `mainrun/logs/baseline.log` — Maincode's reference baseline. **Committed.**
- `mainrun/logs/mainrun.log` — current run's output. Rewritten each run, **not committed** (`.gitignore` excludes `mainrun*.log`).
- `scripts/checkpoint.mjs` moves the previous `mainrun.log` to a timestamped sibling before each new run, so intermediate runs are preserved between checkpoints (locally, not in git).

## Inspecting a run

Each line in `mainrun.log` is one JSON event. Useful queries:

```bash
# Final validation loss
cat mainrun/logs/mainrun.log | jq -c 'select(.event=="validation_step")' | tail -5

# Training curve (step, loss) for plotting
cat mainrun/logs/mainrun.log | jq -r 'select(.event=="training_step") | [.step, .loss] | @csv' > /tmp/train.csv
```

## Dataset cache

HuggingFace `datasets` caches to `mainrun/data/` (covered by `.gitignore`). `task download` runs idempotently. If a run fails with a download error, re-run `task download` and check network.
