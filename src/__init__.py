"""
attn-res-bert
=============

A small, from-scratch PyTorch research library for isolating and benchmarking
"Attention Residuals" (AttnRes, Kimi Team, arXiv:2603.15031) against a
standard PreNorm residual baseline, inside an 8-layer BERT-Medium-sized
encoder trained on masked language modeling (WikiText-103).

    from src.modeling import BertConfig, BertForMaskedLM
    from src.dataset import get_dataloaders
    from src.training import TrainerConfig, train
"""

from .modeling import (
    BertConfig,
    BertForMaskedLM,
    BertEncoder,
    AttentionResidualMix,
    DepthRMSNorm,
)
from .dataset import get_dataloaders, build_tokenizer, MLMCollator
from .training import TrainerConfig, train, evaluate

__all__ = [
    "BertConfig",
    "BertForMaskedLM",
    "BertEncoder",
    "AttentionResidualMix",
    "DepthRMSNorm",
    "get_dataloaders",
    "build_tokenizer",
    "MLMCollator",
    "TrainerConfig",
    "train",
    "evaluate",
]
