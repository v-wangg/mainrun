# Architecture

Current state of the training pipeline in `mainrun/train.py`. This is the starting point — changes will be tracked in [`strategy.md`](./strategy.md) and in `devlogs/`.

## Model — GPT (`mainrun/train.py`)

**Config (`GPTConfig` + `Hyperparameters` defaults):**

| Field | Value |
|-------|-------|
| `vocab_size` | 16 000 (BPE-trained, see below) |
| `block_size` | 128 |
| `n_layer` | 6 |
| `n_head` | 8 |
| `d_model` | 512 (head_dim = 64) |
| `dropout` | 0.1 |
| Params | ~27.17 M |

**Components:**

- **Embeddings** — learned token + learned positional (`nn.Parameter(1, block_size, d_model)`, zero-init).
- **Blocks (×`n_layer`)** — pre-norm GELU MLP with 4× hidden; single-head-grouped QKV projection, causal mask via buffered `tril`, attention-dropout + residual-dropout.
- **Head** — `nn.LayerNorm` → `nn.Linear(d_model, vocab_size, bias=False)`, **tied to `token_emb.weight`**.
- **Init** — Normal(0, 0.02) for Linear/Embedding weights; zero for Linear biases.

## Tokenization

- **Trainer:** HuggingFace `tokenizers` BPE with `ByteLevel` pre-tokenizer and decoder.
- **Vocab:** 16 000. Special tokens: `<pad>`, `<eos>`, `<unk>`.
- **Corpus:** trained on the full set of 100k titles (train + val concatenated) — this means validation titles influence the tokenizer vocabulary. Note for strategy purposes.
- **Formatting:** titles joined by `<eos>`, then BPE-encoded into one long sequence per split.

## Data

- **Source:** `julien040/hacker-news-posts` (HuggingFace datasets, `split="train"`).
- **Sample:** `num_titles=100_000`, shuffled with `seed=1337`.
- **Split:** 90k train / 10k val (`val_frac=0.10`).
- **Batching:** flat token stream, contiguous `block_size`-length windows; no shuffling of windows within an epoch.

Batches per epoch: `len(train_ids) // (block_size * batch_size)` ≈ 134.  
Total steps: `epochs * batches = 7 * 134 = 938`.

## Training recipe (baseline)

| Component | Setting |
|-----------|---------|
| Optimizer | `torch.optim.SGD(lr=6e-3, weight_decay=0.0)` — no momentum |
| Scheduler | `CosineAnnealingLR(T_max=938)` — full cosine over all steps, no warmup |
| Gradient clipping | `clip_grad_norm_(..., 1.0)` |
| Loss | `F.cross_entropy(..., reduction='mean')` over flattened logits |
| Eval cadence | `evals_per_epoch=3` → eval every 44–45 training steps, plus step 1 and the final step |
| Device | `cuda` if available, else `cpu` (baseline was CPU — ~30 min/run at 938 steps) |

## Evaluation — **frozen**

`evaluate()` (nested in `main()`):

```python
def evaluate():
    model.eval()
    losses = 0.0
    with torch.no_grad():
        for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
            logits, _ = model(xb, yb)
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
            losses += loss.item()
    model.train()
    return losses / len(val_text)
```

Key detail: denominator is `len(val_text)` (character count of the validation text), not `len(val_ids)` or the number of predicted positions. This is **loss per character of the raw validation string**, not a standard per-token NLL. Any change in the tokenizer shifts tokens-per-character, which changes the number of predictions made over the same `val_text` — the metric implicitly penalizes inefficient tokenization.

## Logging

Structured JSON via `structlog` to `./logs/mainrun.log`, plus a tqdm print side-channel. Events: `hyperparameters_configured`, `device_info`, `dataset_info`, `model_info`, `training_step`, `validation_step`. Each `validation_step` entry carries `step`, `max_steps`, `loss`, `elapsed_time`.

## Pipeline entry points

- `task train` → `scripts/checkpoint.mjs` (auto-commit) → `download_dataset.py` → `train.py`.
- `task submit` → `scripts/checkpoint.mjs` → `submit.mjs` (zip + upload).
