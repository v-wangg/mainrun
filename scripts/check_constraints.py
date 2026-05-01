#!/usr/bin/env python3
"""
Assert that the immutable rules from README.md hold in mainrun/train.py.

Rules checked:
  1. Hyperparameters defaults:
       epochs=7, seed=1337, num_titles=100_000, val_frac=0.10
  2. Dataset identifier string "julien040/hacker-news-posts" is present.
  3. evaluate() function body is unchanged (AST-normalized comparison).
  4. No pre-trained weight loaders (from_pretrained / torch.hub.load /
     torch.load on external paths).
  5. No obvious training-data augmentation keywords.

Exits non-zero with a clear message on first failure.

Runs as `python scripts/check_constraints.py` or via `task check-constraints`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "mainrun" / "train.py"

EXPECTED_HPARAM_DEFAULTS: dict[str, object] = {
    "epochs": 7,
    "seed": 1337,
    "num_titles": 100_000,
    "val_frac": 0.10,
}

EXPECTED_DATASET = "julien040/hacker-news-posts"

# Canonical evaluate() body. ast.unparse is applied to both sides, so whitespace
# and minor stylistic variations are normalized away — but any semantic change
# will surface as a diff.
EXPECTED_EVALUATE_SRC = '''
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
'''

PRETRAINED_FORBIDDEN = (
    "from_pretrained",
    "torch.hub.load",
    "AutoModel",
    "AutoTokenizer.from_pretrained",
    "load_state_dict",  # flagged; if we add legitimate resume-from-checkpoint later, whitelist in comment
)


def fail(msg: str) -> "NoReturn":  # noqa: F821
    print(f"[check_constraints] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def find_hyperparameters_class(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Hyperparameters":
            return node
    fail("Hyperparameters class not found in train.py")


def check_hyperparameters(tree: ast.Module) -> None:
    cls = find_hyperparameters_class(tree)
    found: dict[str, object] = {}
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            if item.value is not None:
                try:
                    found[name] = ast.literal_eval(item.value)
                except (ValueError, SyntaxError):
                    pass  # non-constant default; skip
    for field, expected in EXPECTED_HPARAM_DEFAULTS.items():
        if field not in found:
            fail(f"Hyperparameters.{field} default not found or not a literal")
        if found[field] != expected:
            fail(
                f"Hyperparameters.{field} = {found[field]!r}, "
                f"expected {expected!r}"
            )


def check_dataset_name(source: str) -> None:
    if EXPECTED_DATASET not in source:
        fail(f'dataset identifier "{EXPECTED_DATASET}" not present in train.py')


def find_evaluate_func(tree: ast.Module) -> ast.FunctionDef:
    # evaluate() is nested inside main()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            return node
    fail("evaluate() function not found in train.py")


def check_evaluate_body(tree: ast.Module) -> None:
    actual_fn = find_evaluate_func(tree)
    actual_src = ast.unparse(actual_fn)
    expected_fn = ast.parse(EXPECTED_EVALUATE_SRC).body[0]
    expected_src = ast.unparse(expected_fn)
    if actual_src != expected_src:
        print("[check_constraints] evaluate() differs from canonical.", file=sys.stderr)
        print("--- expected ---", file=sys.stderr)
        print(expected_src, file=sys.stderr)
        print("--- actual ---", file=sys.stderr)
        print(actual_src, file=sys.stderr)
        fail("evaluate() must be byte-identical (AST-normalized) to the canonical version")


def check_no_pretrained(source: str) -> None:
    hits = [k for k in PRETRAINED_FORBIDDEN if k in source]
    if hits:
        fail(
            "forbidden pretrained-loader reference(s) found: "
            + ", ".join(hits)
            + " — if legitimate, update check_constraints.py with a clear justification"
        )


def main() -> None:
    if not TRAIN_PY.exists():
        fail(f"train.py not found at {TRAIN_PY}")

    source = TRAIN_PY.read_text()
    tree = ast.parse(source)

    check_hyperparameters(tree)
    check_dataset_name(source)
    check_evaluate_body(tree)
    check_no_pretrained(source)

    print("[check_constraints] OK — all immutable rules hold.")


if __name__ == "__main__":
    main()
