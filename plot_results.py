"""
plot_results.py

Generates a comparison plot + fairness table from run_exp.py's saved
<checkpoint_dir>/<variant>/{history.json, fairness_report.json}.

Usage:
    python plot_results.py
    python plot_results.py --checkpoint_dir checkpoints --output attn_res_vs_baseline.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_run(run_dir: Path):
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        history = json.load(f)
    fairness = {}
    fairness_path = run_dir / "fairness_report.json"
    if fairness_path.exists():
        with open(fairness_path) as f:
            fairness = json.load(f)
    return {"history": history, "fairness": fairness}


def main():
    parser = argparse.ArgumentParser(description="Plot baseline vs. attn_res training curves.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="attn_res_vs_baseline.png")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    variants = {"baseline": checkpoint_dir / "baseline", "attn_res": checkpoint_dir / "attn_res"}

    runs = {name: load_run(path) for name, path in variants.items()}
    available = {k: v for k, v in runs.items() if v is not None}
    missing = [k for k, v in runs.items() if v is None]

    if not available:
        print(f"No results found under {checkpoint_dir}/. Run run_exp.py first.")
        return
    if missing:
        print(f"No results yet for: {missing} -- plotting only {list(available.keys())}.")

    COLORS = {"baseline": "#4C72B0", "attn_res": "#DD8452"}
    LABELS = {"baseline": "Baseline (PreNorm)", "attn_res": "Attention Residuals"}

    plt.rcParams.update({
        "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Attention Residuals vs. PreNorm Baseline -- BERT-Medium MLM Pre-training on WikiText-103",
        fontsize=13, y=1.02,
    )

    for name, run in available.items():
        h = run["history"]
        color, label = COLORS[name], LABELS[name]
        axes[0, 0].plot(h["step"], h["train_loss"], color=color, label=label, linewidth=1.8)
        axes[0, 1].plot(h["step"], h["train_ppl"], color=color, label=label, linewidth=1.8)
        if h.get("val_step"):
            axes[1, 0].plot(h["val_step"], h["val_loss"], color=color, marker="o", label=label, linewidth=1.8)
            axes[1, 1].plot(h["val_step"], h["val_ppl"], color=color, marker="o", label=label, linewidth=1.8)

    titles = [("Training Loss", "MLM Loss"), ("Training Perplexity", "Perplexity"),
              ("Validation Loss", "MLM Loss"), ("Validation Perplexity", "Perplexity")]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

    print("\n--- Fairness report ---")
    metrics = [
        ("num_parameters", "Params"),
        ("est_forward_flops_per_step", "Est. FLOPs/step"),
        ("avg_tokens_per_sec", "Tokens/sec (avg)"),
        ("total_wall_clock_sec", "Wall clock (s)"),
    ]
    header = f"{'Metric':<22}" + "".join(f"{LABELS[n]:>22}" for n in available)
    print(header)
    print("-" * len(header))
    for key, label in metrics:
        row = f"{label:<22}"
        for name in available:
            val = available[name]["fairness"].get(key, "—")
            row += f"{val:>22}" if isinstance(val, str) else f"{val:>22,.4g}"
        print(row)


if __name__ == "__main__":
    main()