# Strategy

This document tracks the **hypotheses, priorities, and ablation plan** for minimizing val loss. Populated as the project progresses; the devlogs hold per-run details.

## Framing

With 7 epochs fixed and ~938 steps total, the regime is **compute-constrained, not capacity-constrained.** Baseline under-trains the model. The highest-leverage changes are likely:

1. Better optimizer / learning-rate schedule (SGD-no-momentum + no warmup + cosine-from-step-1 is almost certainly suboptimal).
2. Better regularization balance (dropout 0.1 may or may not be right at this scale; weight decay is 0 which is unusual for a transformer).
3. Tokenization / representation efficiency (remember: loss is per-character of `val_text`, so tokens-per-character matters).
4. Model shape (depth/width/head-count) under the fixed compute budget.
5. Init scheme (GPT-2-style scaled residual init, μP, etc.).

## Hypotheses — initial ranking

Ordered roughly by expected-impact × ease. To be revised as evidence accrues. Each row will be linked to the devlog entry documenting the run.

| # | Hypothesis | Expected impact | Cost | Status |
|---|------------|-----------------|------|--------|
| 1 | AdamW (β₁=0.9, β₂=0.95, wd≈0.1) replaces SGD | Large | Trivial code change | Not started |
| 2 | LR warmup (e.g. 5–10% of steps) + cosine decay to ~10% peak | Medium-large | Trivial | Not started |
| 3 | Sweep peak LR on top of AdamW (grid around 1e-3, 3e-4, 6e-4) | Medium | Small | Not started |
| 4 | Tune weight decay independently | Medium | Small | Not started |
| 5 | Longer `block_size` (256 or 512) to capture more title-context | Unknown | Need to re-batch | Not started |
| 6 | Tokenizer rebuild — train only on train split; vocab ablation | Medium | Small | Not started |
| 7 | Drop dropout (or reduce) given training is the bottleneck, not overfitting | Medium | Trivial | Not started |
| 8 | Scaled residual init (GPT-2 style: `std=0.02/sqrt(2*n_layer)` for residual projs) | Small-medium | Small | Not started |
| 9 | Layer-norm placement / RMSNorm | Small | Small | Not started |
| 10 | Wider-but-shallower (or deeper-but-narrower) under ~same param budget | Unknown | Medium | Not started |
| 11 | Label smoothing | Small | Trivial | Not started |
| 12 | Mixed precision (bf16) — speed only, enables larger sweeps | Neutral on loss | Small | Not started |

## Ablation protocol

- **One variable per run** when attributing deltas. Occasional bundled changes are fine but must be flagged in the devlog as "bundled, not ablated."
- **Record:** full hyperparameters, git commit hash, val loss at every eval step, wall-clock time.
- **Plot:** every run's val curve overlaid on the baseline; deltas are the signal.
- **Honesty bar:** run-to-run noise from seed effects is zero here (seed is frozen) — but tokenizer retraining or data-loader changes can still introduce non-reproducible noise. Flag these.

## Subtle trap — the loss denominator

`evaluate()` returns `total_loss_sum / len(val_text)`. `val_text` is the character-length of the raw validation string, which is **fixed across runs** (the split is deterministic and the string is the same). So the denominator is a constant — a good thing for comparability.

But: the numerator is a sum over **token predictions made** during evaluation. `iter_full_split` only yields full `batch_size × block_size + 1` windows, dropping the tail. If tokenizer changes alter `len(val_ids)`, the number of windows iterated changes, and the number of tokens summed changes. A smaller vocab → longer token sequences → more windows → more terms summed → apparent loss ↑. A larger or more efficient vocab → fewer windows → fewer terms summed → apparent loss ↓.

**Implication:** tokenizer efficiency is implicitly rewarded even if per-token NLL is identical. Frame tokenizer choices with this in mind.

## Report-writing parallel track

The `report.pdf` is a graded deliverable. Writing it as we go (not at the end) means:

- Every devlog entry is a draft paragraph.
- Plots are generated as part of each run, not reconstructed from logs later.
- The final narrative is composed, not invented.

## Where the research comes in

External work we want to consult — literature review, Karpathy's training tricks, known small-LLM recipes — lands in `source-info/` via vault handoff. See [`../source-info/README.md`](../source-info/README.md) and the paper-ingest skill in the vault.
