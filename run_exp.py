#!/usr/bin/env python
"""
run_exp.py

CLI entry point for benchmarking Attention Residuals (AttnRes) against
a standard PreNorm-residual BERT baseline on MLM pre-training over WikiText-103.

Example usage:

    # Baseline (standard PreNorm identity residuals)
    python run_exp.py --model_variant baseline --epochs 3 --batch_size 32 --lr 3e-4

    # Full Attention Residuals (Kimi Team, arXiv:2603.15031)
    python run_exp.py --model_variant attn_res --epochs 3 --batch_size 32 --lr 3e-4

Both runs write their config, metric history, and final weights under
``--output_dir/<run_name>/``, so the two variants can be diffed / plotted
against each other afterward.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.modeling import BertConfig, BertForMaskedLM, estimate_forward_flops
from src.dataset import get_dataloaders
from src.training import TrainerConfig, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Attention Residuals (AttnRes) vs. a standard PreNorm BERT baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment selection -------------------------------------------------
    parser.add_argument(
        "--model_variant", type=str, required=True, choices=["baseline", "attn_res"],
        help="'baseline' = standard PreNorm identity residuals; "
             "'attn_res' = Full Attention Residuals (Kimi Team).",
    )
    parser.add_argument("--run_name", type=str, default=None,
                         help="Optional run tag for output_dir/history logging; defaults to model_variant.")

    # --- Model / architecture (BERT-Medium defaults) --------------------------
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--max_seq_length", type=int, default=256,
                         help="Sequence length used for both packing and position embeddings.")

    # --- Data -------------------------------------------------------------
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                         help="HF tokenizer whose vocabulary/WordPiece rules to reuse (weights are NOT loaded).")
    parser.add_argument("--num_proc", type=int, default=4, help="CPU workers for datasets.map tokenization/packing.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where `datasets` caches the raw download AND tokenized/packed results. On Colab, "
                            "point this at a mounted Google Drive path (e.g. /content/drive/MyDrive/hf_cache) "
                            "so tokenization is only ever paid for once, not re-run every fresh runtime.")

    # --- Optimization -------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "no"],
                         help="'fp16' is the safe default for a Colab T4 (no native bf16 support).")

    # --- Logging / Miscellaneous -------------------------------------------------------
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or args.model_variant
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    num_proc = args.num_proc if args.num_proc is not None else min(4, os.cpu_count() or 1)
 
    if args.cache_dir is not None:
        print(f"=== [{run_name}] Using dataset cache_dir={args.cache_dir} "
              f"(point this at a mounted Drive path on Colab to persist across sessions) ===")
 
    print(f"=== [{run_name}] Building dataloaders (WikiText-103-raw-v1, seq_len={args.max_seq_length}, "
          f"num_proc={num_proc}) ===")

    loaders = get_dataloaders(
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_proc=num_proc,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        mlm_probability=args.mlm_probability,
    )
    tokenizer = loaders["tokenizer"]

    print(f"=== [{run_name}] Building model (variant={args.model_variant}) ===")
    model_config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
        use_attn_res=(args.model_variant == "attn_res"),
    )
    model = BertForMaskedLM(model_config)
    print(f"[{run_name}] Model parameters: {model.num_parameters / 1e6:.2f}M "
          f"(use_attn_res={model_config.use_attn_res})")

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        eval_every=args.eval_every,
        mixed_precision=args.mixed_precision,
        output_dir=str(output_dir),
    )

    print(f"=== [{run_name}] Training ({args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}) ===")
    history = train(model, loaders["train"], loaders["val"], trainer_config, run_name=run_name)

    # Matching epoch count between variants matches tokens seen, but NOT
    # compute or wall-clock: AttnRes does strictly more FLOPs/step than the
    # baseline (its depth-wise softmax mixes). This report makes that
    # asymmetry explicit rather than letting "3 epochs vs 3 epochs" quietly
    # imply "equal compute".
    flops_per_step = estimate_forward_flops(model_config, args.batch_size, args.max_seq_length)
    fairness_report = {
        "model_variant": args.model_variant,
        "num_parameters": model.num_parameters,
        "est_forward_flops_per_step": flops_per_step,
        "epochs": args.epochs,
        "tokens_per_step": args.batch_size * args.max_seq_length,
        "avg_tokens_per_sec": history.get("avg_tokens_per_sec"),
        "total_wall_clock_sec": history.get("total_wall_clock_sec"),
    }
    print(f"=== [{run_name}] Fairness report (compare against the other variant's) ===")
    for k, v in fairness_report.items():
        print(f"    {k}: {v}")
 
    # --- persist run artifacts -------------------------------------------------
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "fairness_report.json", "w") as f:
        json.dump(fairness_report, f, indent=2)
    torch.save(model.state_dict(), output_dir / "model.pt")
 
    print(f"=== [{run_name}] Done. Artifacts saved to {output_dir} ===")
 
 
if __name__ == "__main__":
    main()
 
