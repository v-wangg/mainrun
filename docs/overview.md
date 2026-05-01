# Overview

## The task

Maincode's take-home for the Research Engineer role. Given a small GPT-2 style training pipeline on Hacker News headlines, minimize validation loss as far below the baseline as possible — within a fixed 7-epoch budget.

**Baseline:** `val_loss = 1.7545` at step 704/938 (see `mainrun/logs/baseline.log`).

**Target:** Beat 1.7545 by a margin large enough to signal deliberate methodology rather than stochastic noise.

## Non-negotiables

See [`../CLAUDE.md`](../CLAUDE.md) § *Immutable Constraints* for the authoritative list. The short version:

- `epochs=7`, `seed=1337`, `num_titles=100_000`, `val_frac=0.10`, dataset unchanged.
- `evaluate()` body unchanged.
- No pre-trained weights. No training-data augmentation.

## Fair game

- Model architecture (attention variant, FFN variant, normalization, positional encoding, init scheme, weight tying, depth/width).
- Tokenization (BPE choices, vocab size, pre-tokenization, special tokens).
- Training recipe (optimizer, scheduler, warmup, weight decay, gradient clipping, loss function, label smoothing, dropout, batch_size, block_size).
- Engineering (mixed precision, compilation, data loader, logging, checkpointing).

## Deliverables

1. **Working code** — `task train` reproduces the final run end-to-end.
2. **`mainrun/report.pdf`** — short write-up covering:
   - What was changed and why.
   - What each change did to the training curve.
   - The reasoning trail: what was tried, what worked, what didn't.
3. **Submission** — `task submit` packages + uploads.

## Evaluation criteria (per `README.md`)

In order of importance:

1. **Model performance** — val loss improvement vs. baseline.
2. **Code quality** — clean, maintainable, well-documented.
3. **Innovation** — creative or insightful approaches.
4. **Documentation** — the report.

## How we're approaching this

Strategy details in [`strategy.md`](./strategy.md). Running principle: **many small, well-logged ablations beat a single big change.** Every modification needs an independent val-loss delta attributable to it; otherwise the report can't tell the story.
