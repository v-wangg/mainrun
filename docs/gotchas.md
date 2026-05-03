# Gotchas

Traps, anti-patterns, and submission-breaking pitfalls. Re-read before every non-trivial change.

## Constraint-breaking

- **Re-binding `Hyperparameters` fields at runtime is still a change.** Overriding `args.epochs = 10` anywhere in the pipeline breaks the rule just as much as editing the dataclass default. Don't do it.
- **`evaluate()` body is frozen.** No refactor, no helper extraction, no "just adding a print". Leave it exactly as shipped.
- **Don't change the dataset string, revision, or split.** `julien040/hacker-news-posts`, `split="train"`, default revision.
- **Pre-trained weights** — any `from_pretrained`, `torch.hub.load`, `torch.load(<external>)`, remote weight download, or manually-pasted state dict breaks the rules.
- **Data augmentation is forbidden.** This includes subtle things: token dropout on inputs *is* augmentation of training data; lowercasing *is* augmentation; title concatenation order changes *are* augmentation when non-deterministic. Augmenting the validation set is obviously also out.
- **`val_frac` is 0.10, not "approximately 10%".** Don't change the split logic in `get_titles()`.

## Validation metric — the denominator

`evaluate()` returns `total_loss_sum / len(val_text)`. `val_text` is the **character count of the raw validation string**. This means:

- The metric is loss-per-character, not loss-per-token.
- More efficient tokenization (fewer tokens for the same text) → fewer summed terms in the numerator → lower reported loss, even if per-token NLL is identical.
- Changing the tokenizer changes the apparent loss independent of model quality. Log tokens-per-character for every tokenizer variant so the comparison is honest.

**Easy misread to avoid.** `val_text` (constructed in `main()` as `eos_token.join(val_titles) + eos_token`) is a Python *string*. `val_ids` (the token tensor) is a *separate* variable. A hurried read can flip the two and report the metric as per-token; it is per-character. The `<eos>` separator expands to 5 literal characters inside `val_text`, so the denominator is (sum of headline char-lengths) + 5 × (n_val_titles + 1), not just the headline chars — constant across runs, but worth knowing if you ever sanity-check the number by hand.

## Auto-checkpoint will commit everything

`task train` runs `scripts/checkpoint.mjs` first, which does `git add . && git commit`. This means **any dirty file** — stray notes, experimental scripts, scratch plots — gets committed into the submission history.

- Keep the working tree clean before `task train`.
- Use `.gitignore` for scratch dirs.
- Ablation branches aren't allowed by our main-only git rule, but WIP edits that you haven't decided to keep should live outside the repo or under an ignored path.

## Dev container enforcement

`mainrun/utils.py` aborts on `_check_devcontainer()` if `/root/.mainrun` doesn't exist. Running `python3 mainrun/train.py` from a host shell will hard-exit. Use `task train`.

## Submission integrity

- `task submit` packages the entire repo. Anything committed is in the zip.
- `report.pdf` must be at `mainrun/report.pdf` (not repo root) per `README.md`.
- Logs from the final run must reflect the final code state — don't submit a stale `baseline.log`-shaped file in place of a fresh `mainrun.log`.

## Tokenizer is trained on train-only

`main()` trains the tokenizer on `train_titles` (not `train_titles + val_titles`). Validation text does not shape the vocabulary, so there's no information-flow channel from val into the BPE merges. If you change this back, note it as an ablation — the metric is sensitive because tokenization efficiency directly affects the per-character denominator (see "Validation metric — the denominator" above).

## Schedule length depends on `max_steps`, which depends on data shape

The custom `get_lr()` in `mainrun/optim.py` is parameterized by `max_steps = args.epochs * batches`, where `batches` depends on `block_size`, `batch_size`, and the post-BPE `len(train_ids)`. Changing any of these changes `max_steps` — recompute it the same way in any scheduler variant, and re-pick `warmup_steps` proportionally rather than as a fixed integer.

## CPU-vs-GPU determinism

Seed is fixed (`1337`), but CPU and CUDA execute non-deterministic ops differently (non-commutative float reductions, different kernel choices). Baseline was CPU. GPU runs may produce slightly different training dynamics even with identical code and seed. Note the device in every devlog entry.
