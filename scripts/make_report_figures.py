#!/usr/bin/env python3
"""Generate report figures into docs/images/.

Figure 1: Ablation cascade — chronological cumulative-best per-char val loss.
Figure 2 is the user's wandb screenshot, copied into docs/images/ directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "images"

CASCADE = [
    # (label, loss, kind) — kind: "baseline", "step" (cumulative-best), "ablation" (control)
    ("Baseline reprod",         1.7545, "baseline"),
    ("AdamW + warmup",          1.4293, "step"),
    ("Stochastic dataloader",   1.4357, "step"),
    ("Scaled residual init",    1.3769, "step"),
    ("BPE train-only",          1.3722, "step"),
    ("LR → 1e-3",               1.3627, "step"),
    ("Dropout = 0",             1.3470, "step"),
    ("B=256, lr=2e-3",          1.0763, "step"),
    ("GPT-4 pre-tok",           1.0670, "step"),
    ("− Doc mask (ablation)",   1.0848, "ablation"),
]


def _bar_color(delta: float | None, kind: str) -> str:
    if kind == "baseline":
        return "#7d8a99"
    if kind == "ablation":
        return "#d97a7a"
    if delta is not None and delta > 0:
        return "#d97a7a"
    mag = abs(delta) if delta is not None else 0.0
    if mag >= 0.2:
        return "#1f5fbf"
    if mag >= 0.05:
        return "#5fa3e6"
    return "#a8c8e8"


def make_figure_1(out_path: Path) -> None:
    labels = [c[0] for c in CASCADE]
    losses = [c[1] for c in CASCADE]
    kinds = [c[2] for c in CASCADE]

    deltas: list[float | None] = [None]
    last_step_loss = losses[0]
    for i in range(1, len(losses)):
        if kinds[i] == "ablation":
            deltas.append(losses[i] - losses[i - 1])
        else:
            deltas.append(losses[i] - last_step_loss)
            last_step_loss = losses[i]

    colors = [_bar_color(d, k) for d, k in zip(deltas, kinds)]
    hatches = ["///" if k == "ablation" else "" for k in kinds]

    fig, ax = plt.subplots(figsize=(10, 6.0), dpi=150)
    y = np.arange(len(labels))
    bars = ax.barh(y, losses, color=colors, edgecolor="white", height=0.7)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()

    for i, (loss, delta) in enumerate(zip(losses, deltas)):
        if delta is None:
            label = f"{loss:.4f}"
        else:
            sign = "+" if delta > 0 else "−"
            label = f"{loss:.4f}    Δ {sign}{abs(delta):.3f}"
        ax.text(loss + 0.012, i, label, va="center", fontsize=10, color="#222222")

    ax.set_xlabel("Per-character validation loss", fontsize=11)
    ax.set_xlim(1.0, 2.05)
    ax.set_title(
        "Intervention cascade (chronological, cumulative-best)",
        fontsize=12, pad=14, loc="left",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_figure_1(OUT_DIR / "figure-1-cascade.png")


if __name__ == "__main__":
    main()
