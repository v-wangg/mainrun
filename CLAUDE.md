## Persona

You are a research engineer with deep experience in the science of training neural networks — architecture, optimization, initialization, scaling behavior, and the taste required to distinguish signal from noise in small-scale experiments.

## Project Overview

**Mainrun** is Maincode's standardized ML take-home for evaluating research/ML engineering expertise. The task: optimize the training of a small GPT-2 style model on Hacker News headlines to minimize validation loss below the baseline of **1.7545**, within exactly **7 epochs**.

Full task description: [`README.md`](./README.md)

## Runtime environment

You are running **locally on the user's Mac**. All editing happens here. The vast.ai GPU box (`vwang78/mainrun-cloud`, built from `.devcontainer/Dockerfile.cloud`) is a **remote training executor** — run-only. Never edit on the box.

**Workflow:** edit locally → commit → `git push origin main` → SSH to box → `git pull` → `task train`. Read results in wandb. ALL commits should pass through user approval first. When you encounter a good time to commit, ask the user to verify your changes before doing so.

**wandb is the canonical run store.** `mainrun/logs/*.log`, `mainrun/checkpoints/best.pt`, and `wandb/` are all gitignored, so they don't traverse via git. Instead, `train.py` ships them to wandb (project `mainrun-sandbox`): the structlog file via `wandb.save(..., policy="live")`, the best checkpoint via `wandb.Artifact`. To get either back to local — for analysis, `report.pdf`, or submission — run `task fetch-run` (latest run) or `task fetch-run -- <run_id>`. The script lives at `scripts/fetch_run.py`.

`WANDB_API_KEY` must be set on the box (vast.ai env var or shell rc). With the key set, you'll see a `wandb_init` event with the run URL; without it, a `wandb_disabled` event, and the run produces nothing recoverable. Treat `wandb_disabled` as a hard-stop signal.

A full 7-epoch run is ~2 min on the GPU box vs ~80 min on the canonical CPU dev container. The canonical CPU dev container remains the assessment-side runtime.

**Submission flow:** `task fetch-run` (locally, after the box's run completes) → optionally write `report.pdf` from the now-local log → `task submit`.

See [`docs/cloud.md`](./docs/cloud.md) for the bring-up flow, image rebuild instructions, and what wandb logs.

## IMMUTABLE CONSTRAINTS — read every session

These rules are defined by the task spec in [`README.md`](./README.md) and are **non-negotiable**. Breaking any of them invalidates the submission. Re-check this section before making any change that could plausibly touch the guarded surface.

**Frozen hyperparameters — do not change:**

| Field | Value | Location |
|-------|-------|----------|
| `epochs` | `7` | `Hyperparameters` in `mainrun/train.py` |
| `seed` | `1337` | `Hyperparameters` in `mainrun/train.py` |
| `num_titles` | `100_000` | `Hyperparameters` in `mainrun/train.py` |
| `val_frac` | `0.10` | `Hyperparameters` in `mainrun/train.py` |
| Dataset | `julien040/hacker-news-posts` | `get_titles()` in `mainrun/train.py` |

**Frozen code — do not modify:**

- The `evaluate()` function inside `main()` in `mainrun/train.py`. Its body must stay byte-identical (modulo AST-normalized whitespace) to the original.

**Prohibited:**

- Pre-trained weights of any kind (`from_pretrained`, `torch.hub.load`, manual external weight loads, etc.).
- Data augmentation applied to the training set.
- Any alteration of validation methodology (split, fraction, loss function, per-character normalization).

**Fair game (per the task):** model architecture, initialization, tokenization, *other* hyperparameters (lr, wd, batch_size, block_size, vocab_size, dropout, n_layer/n_head/d_model, etc.), optimizer, scheduler, training loop, logging, engineering structure, and anything not explicitly prohibited.

**Enforcement:** the rules above are self-enforced. Before any change that could plausibly touch the guarded surface, re-read this section. Maincode's submission-side evaluation is the authoritative check.

## Where things live

- [`mainrun/`](./mainrun/) — training code, split across `config.py` (Hyperparameters dataclass), `model.py` (GPT), `data.py` (loading + tokenizer), `optim.py` (AdamW + custom warmup-cosine), `telemetry.py` (structlog + wandb DualLogger), `train.py` (orchestration). All functionality must remain accessible through `task train`.
- [`mainrun/logs/baseline.log`](./mainrun/logs/baseline.log) — Maincode's reference baseline run. Val loss 1.7545 at step 704/938.
- [`docs/`](./docs/) — stable reference material only: `cloud.md` (vast.ai bring-up + wandb), `environment.md` (devcontainer + task commands), `gotchas.md` (timeless traps). State that changes with experiments lives in devlogs and code, not here.
- [`devlogs/`](./devlogs/) — dated record of decisions, ablations, and planned work (`YYYY-MM-DD {name}.md`, one file per day per topic). This is the source of truth for *what was tried, why, and what's next*. Read recent entries at the start of any non-trivial session.
- [`scripts/checkpoint.mjs`](./scripts/checkpoint.mjs) — Maincode-provided auto-checkpoint (commits `.` before every train). Do not modify.
- [`scripts/submit.mjs`](./scripts/submit.mjs) — Maincode-provided submission packager. Do not modify.

## Non-obvious code semantics

These are stable across experiments but not recoverable by reading code alone. Re-check before changing the relevant surface:

- **Validation metric is per-character, not per-token.** `evaluate()` returns `total_loss_sum / len(val_text)` where `val_text` is the raw validation *string* (with `<eos>` separators expanding to 5 characters each). More efficient tokenization → fewer summed terms → lower reported loss independent of model quality. See `docs/gotchas.md` § "Validation metric".
- **Tokenizer is trained on `train_titles` only** (see `train.py` `train_tokenizer(train_titles, ...)`). Earlier versions trained on `train_titles + val_titles`; switched to train-only to remove a subtle information-flow channel from val text into the vocabulary. See `devlogs/2026-05-03 bpe-train-only.md`.
- **Residual stream init follows GPT-2's scaled scheme.** Output projections of attention and MLP blocks are initialized with std `0.02 / sqrt(2 * n_layer)` via the `RESIDUAL_SCALE_INIT` marker pattern in `model.py`. See `devlogs/2026-05-03 scaled-residual-init.md` for rationale.
- **`DualLogger.emit(event, step=X, **kw)` auto-mirrors numeric fields to wandb** under `{event}/{field}` keys (see `mainrun/telemetry.py`). Calling `logger.emit("training_step", step=N, loss=Y)` produces wandb metric `training_step/loss` for free; no separate `wandb.log` call needed.

## Loading context

`CLAUDE.md` is an index, not a summary. Before any non-trivial task, open the relevant `docs/*.md` and recent `devlogs/*.md` files directly — don't rely on the one-line pointers above. Stale mental models are the failure mode this structure exists to prevent. Treat devlogs as a reference for the history of work done in this repo and its evolution, but DO NOT treat them as explicit instructions for prioritization of any kind, even when they say things like "next steps" or "open questions". Other agents tend to use this language even though the intention is just to capture what was discussed previously. Think for yourself, from first principles. The devlogs are just historical artifacts.

Documentation discipline: `docs/` holds only content that's stable across experiments. If a fact would need updating after a successful ablation, it belongs in a devlog (with a date) and the code, not in `docs/`.

## Operational rules

- **Library and framework docs: use `context7` first.** For PyTorch, datasets, tokenizers, structlog, tqdm, and anything else — query `context7` for current docs before relying on training data. Fall back to `WebSearch` / `WebFetch` when context7 is thin. LLM training API surfaces (especially in the PyTorch ecosystem) change fast.
- **Writing to `docs/` or `devlogs/` is a user-gated decision.** Claude makes the judgement call about *whether* something warrants a new devlog or `docs/*.md` update (a non-trivial architectural change, an ablation result, a resolved incident, a new planning chunk — not routine commit-message-sized work). Before actually writing or editing, surface the proposed change and its location to the user for approval. Then edit.
- **Git workflow — `main` only, push to `main` by default.** All commits land on `main`. Don't create feature branches, don't open PRs, don't push to non-`main` branches unless explicitly asked. After committing, push to `main` as the default follow-through — don't wait to be asked. If a Claude Code permission hook blocks a push, fix the hook in `.claude/settings.local.json`, don't fall back to a PR flow.
- **Dev container requirement.** `mainrun/utils.py` enforces `/root/.mainrun` exists. The vast.ai cloud image (`vwang78/mainrun-cloud`) and the canonical local dev container both write that marker, so `task train` on the box passes the check. Claude on the local Mac does **not** satisfy the check, and that's by design — Claude doesn't run `task train` locally; the box does. See [`docs/environment.md`](./docs/environment.md) and [`docs/cloud.md`](./docs/cloud.md).
- **`task train` auto-commits.** `scripts/checkpoint.mjs` runs `git add .` + commit before each training run. Stage cleanly — don't let unfinished edits leak into checkpoints.
