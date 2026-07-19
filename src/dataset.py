"""
src/dataset.py
===============

A clean, memory-mapped (Arrow-backed) WikiText-103 pipeline for MLM
pre-training:

  1. Load ``wikitext-103-raw-v1`` via Hugging Face ``datasets``.
  2. Tokenize with a Hugging Face ``tokenizers``-backed fast tokenizer
     (we reuse the ``bert-base-uncased`` WordPiece vocabulary only --
     no pretrained weights are ever loaded anywhere in this library;
     the model in ``modeling.py`` is trained entirely from scratch).
  3. Concatenate + re-chunk ("pack") the token stream into dense,
     padding-free ``max_seq_length`` blocks -- the standard trick used in
     HF's own MLM pre-training examples, so no compute is wasted on [PAD].
  4. Apply dynamic masking (a fresh 15% BERT-style mask every time a
     batch is collated) rather than masking once and caching -- this gives
     the model a different masking pattern on every pass over the data.

Everything here is built on ``datasets.map(..., num_proc=...)``, which is
Arrow/memory-mapped under the hood, so this scales past what fits in RAM
and works fine in a Colab session.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

def build_tokenizer(tokenizer_name: str = "bert-base-uncased") -> PreTrainedTokenizerFast:
    """Loads a Rust (``tokenizers``-backed) fast WordPiece tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer '{tokenizer_name}' has no [MASK] token; MLM requires one.") # Check potential improvement here. In principle, we could add a [MASK] token to the tokenizer if it doesn't have one, but that would require adjusting the model's embedding layer as well. For now, we just raise an error.
    return tokenizer


# --------------------------------------------------------------------------- #
# Tokenization + packing ("group_texts")
# --------------------------------------------------------------------------- #

def _tokenize_batch(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerFast) -> Dict[str, List]:
    # No truncation/padding at this stage -- we tokenize the raw text as-is
    # and defer fixed-length chunking to `_group_texts` below.
    return tokenizer(examples["text"], add_special_tokens=True)


def _group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    """Concatenate every sequence in the batch end-to-end, then re-slice
    into contiguous chunks of exactly ``block_size`` tokens. This "packing"
    approach (used in HF's ``run_mlm.py``) avoids padding waste: every
    training example is a fully-dense sequence of real tokens.

    Note: this does mean a small fraction of examples straddle document
    boundaries (with [CLS]/[SEP] tokens from adjacent WikiText articles
    appearing mid-sequence) -- an accepted, standard trade-off for
    pre-training throughput.
    """
    # batched = True -> examples is a list of lists; we want to concatenate all of them into one long list (per key).
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[next(iter(examples.keys()))]) # len(concatenated["input_ids"]) would also work, but this is more general in case the keys change.
    
    total_length = (total_length // block_size) * block_size  # drop the remainder
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    # result is a dict of lists of lists (per key), where each inner list is a block of size block_size. The outer list has length total_length // block_size.
    return result


def load_wikitext_mlm_dataset(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    max_seq_length: int = 256,
    num_proc: int = 4,
    dataset_config: str = "wikitext-103-raw-v1",
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Loads, tokenizes, and packs WikiText-103 for a given split.

    Returns an Arrow-backed ``datasets.Dataset`` (memory-mapped, not loaded
    fully into RAM) with columns ``input_ids`` and ``attention_mask``, each
    a fixed-length list of exactly ``max_seq_length`` tokens.
    """
    raw = load_dataset("wikitext", dataset_config, split=split, cache_dir=cache_dir)

    # wikitext-raw contains many blank lines and bare markdown headings
    # ("= = Title = ="); filtering these out avoids wasting packed blocks
    # on near-empty content.
    raw = raw.filter(lambda ex: len(ex["text"].strip()) > 0, num_proc=num_proc)

    tokenized = raw.map(
        lambda ex: _tokenize_batch(ex, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing wikitext[{split}]",
    )

    packed = tokenized.map(
        lambda ex: _group_texts(ex, block_size=max_seq_length),
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc=f"Packing wikitext[{split}] into {max_seq_length}-token blocks",
    )

    packed.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return packed


# --------------------------------------------------------------------------- #
# Dynamic MLM masking collator
# --------------------------------------------------------------------------- #

@dataclass
class MLMCollator:
    """Dynamic masking collator implementing the classic BERT recipe.

    For each token position (excluding special tokens), independently with
    probability ``mlm_probability`` (default 15%) the position is selected
    for masking. Among selected positions:
      - 80% are replaced with ``[MASK]``
      - 10% are replaced with a uniformly random vocabulary token
      - 10% are left unchanged

    ``labels`` is set to -100 (the ``ignore_index`` used in
    ``training.py`` / ``modeling.py``) everywhere except the selected
    positions, where it holds the original (pre-masking) token id -- this
    is what makes the loss "masked" in the first place.

    Because masking is recomputed inside ``__call__`` (i.e. every time a
    batch is collated, so every epoch), this is *dynamic* masking, as
    opposed to computing one fixed mask per example ahead of time.
    """

    tokenizer: PreTrainedTokenizerFast
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # examples is a list of dicts of length batch_size, each dict has keys "input_ids" and "attention_mask", each a tensor of shape (max_seq_length,). We want to collate these into a single dict of tensors of shape (batch_size, max_seq_length).
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        attention_mask = torch.stack([ex["attention_mask"] for ex in examples])
        input_ids, labels = self._mask_tokens(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _mask_tokens(self, input_ids: torch.Tensor):
        labels = input_ids.clone()

        # Never mask special tokens ([CLS], [SEP], [PAD], etc.).
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
                for seq in input_ids # Loop over each sequence in the batch to get a mask of special tokens. This will be a list of lists of 0s and 1s, where 1 indicates a special token. We convert this to a tensor of shape (batch_size, max_seq_length).
            ],
            dtype=torch.bool,
        )

        probability_matrix = torch.full(labels.shape, self.mlm_probability) # Create a matrix of shape (batch_size, max_seq_length) filled with the masking probability (0.15 by default).
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # prob = 0.0 for special tokens, so they are never masked. The rest of the tokens have a 15% chance of being masked.
        masked_indices = torch.bernoulli(probability_matrix).bool() # Uses the probability matrix to sample a mask of the same shape, where each position is True with probability 0.15 (except for special tokens, which are always False).
        labels[~masked_indices] = -100 # loss is only computed at masked positions

        # 80% of masked positions -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # Of the remaining 20%, half (i.e. 10% overall) -> random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        )
        random_tokens = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        # The final 10% are left as-is (label still points to the true
        # token, but the model sees the correct token as input -- this
        # forces it to build representations that don't purely rely on the
        # presence of [MASK] as a signal).
        return input_ids, labels


# DataLoader builder

def get_dataloaders(
    tokenizer_name: str = "bert-base-uncased",
    max_seq_length: int = 256,
    batch_size: int = 32,
    num_proc: int = 4,
    num_workers: int = 2,
    mlm_probability: float = 0.15,
) -> Dict[str, object]:
    """One-shot builder: returns train/validation DataLoaders plus the
    tokenizer (needed downstream to size the model's vocabulary and to
    resolve special-token ids)."""

    tokenizer = build_tokenizer(tokenizer_name)

    train_ds = load_wikitext_mlm_dataset(
        tokenizer, split="train", max_seq_length=max_seq_length, num_proc=num_proc
    )
    val_ds = load_wikitext_mlm_dataset(
        tokenizer, split="validation", max_seq_length=max_seq_length, num_proc=num_proc
    )

    collator = MLMCollator(tokenizer=tokenizer, mlm_probability=mlm_probability)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "tokenizer": tokenizer}
