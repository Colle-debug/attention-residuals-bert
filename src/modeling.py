"""
src/modeling.py
================

From-scratch PyTorch implementation of a PreNorm, BERT-Medium-sized encoder
(hidden=512, layers=8, heads=8, ffn=2048), with a single architectural toggle:

    BertConfig(use_attn_res: bool)

If ``use_attn_res=False`` (the baseline), every sublayer uses a standard
PreNorm identity residual:

    h_l = h_{l-1} + f_{l-1}(LN(h_{l-1}))

If ``use_attn_res=True``, every sublayer's residual is replaced by **Full
Attention Residuals** exactly as described in "Attention Residuals"
(Kimi Team, technical report, arXiv:2603.15031):

    phi(q, k)    = exp( q^T RMSNorm(k) )
    alpha_{i->l} = phi(w_l, v_i) / sum_j phi(w_l, v_j)     (softmax over depth)
    h_l          = sum_i alpha_{i->l} * v_i

where the "value history" v_0, v_1, ..., v_{l-1} is the token embedding
followed by the raw output of every preceding sublayer (attention or MLP),
w_l is a single learned, zero-initialized pseudo-query vector *per sublayer*
(decoupled from the forward activations), and the RMSNorm applied to the
*keys* (not the values) is what we call ``DepthRMSNorm`` below: it prevents
sublayers with naturally larger output magnitudes from dominating the
depth-wise softmax purely due to scale.

A standard tied-weight Masked Language Modeling (MLM) head sits on top.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class BertConfig:
    """Architecture + toggle configuration. Defaults match BERT-Medium
    (Turc et al., 2019): hidden_size=512, num_hidden_layers=8, heads=8,
    intermediate_size=2048 -- small enough to comfortably pre-train from
    scratch on a single Colab T4 within a demo-scale budget."""

    vocab_size: int = 30522
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 256
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

    # The single architectural toggle this library exists to benchmark:
    use_attn_res: bool = False


# --------------------------------------------------------------------------- #
# Attention Residuals building blocks
# --------------------------------------------------------------------------- #

class DepthRMSNorm(nn.Module):
    """RMSNorm applied independently, along the last (feature) dimension, to
    each vector in a stack of "depth" candidates.

    Given a tensor of shape ``[N, B, T, D]`` -- N candidate value vectors
    (one per depth/history entry) for every (batch, token) position -- this
    module RMS-normalizes each of the N vectors *independently* over D. This
    is exactly the "RMSNorm on keys" ablated in the Attention Residuals
    technical report: without it, sublayers whose outputs happen to have
    larger norm would dominate the depth-wise softmax purely due to scale,
    rather than the content actually being more relevant to the current
    query. The report finds removing it measurably hurts loss for both Full
    and Block AttnRes (Table 4).

    Deliberately has **no learnable gain**: a learned per-channel scale would
    reintroduce a magnitude-dominance channel this normalization exists to
    remove, and the technical report's formulation ``exp(q^T RMSNorm(k))``
    uses a bare (gain-free) RMSNorm on the keys.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over the last dim only; broadcasts cleanly over any
        # leading [depth, batch, seq] shape.
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)


class AttentionResidualMix(nn.Module):
    """Full Attention Residuals (Kimi Team).

    Replaces the fixed, unit-weight additive accumulation of a standard
    residual stream with depth-wise **softmax attention** over the entire
    history of previous "value" vectors:

        v_0 = token embedding
        v_i = output of sublayer i,  for i = 1, ..., l-1

        alpha_{i->l} = softmax_i( w_l . RMSNorm(v_i) )
        h_l          = sum_i alpha_{i->l} * v_i

    ``w_l`` (this module's ``self.w``) is a single learned pseudo-query
    vector per sublayer. Crucially it does **not** depend on the current
    hidden state or token content -- it is a free parameter, decoupled from
    the forward computation, which is what allows (in the original paper) a
    whole block's queries to be batched together for efficient inference.

    Zero-initialization of ``w`` is not a minor detail but a *requirement*
    from the paper: at ``w = 0``, every ``phi(w, k) = exp(0) = 1``, so
    ``alpha_{i->l}`` is uniform over all history entries and Full AttnRes
    starts out as a plain running average -- close to (though not identical
    to, due to the 1/N vs. 1 scaling) the additive baseline. This keeps
    training stable at initialization and lets the model *learn* to deviate
    from uniform averaging as training progresses, rather than starting from
    an arbitrary, possibly destabilizing, mixture.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # REQUIRED zero-init -- see class docstring. Do not change this to a
        # random init; doing so breaks the "reduces to a residual at step 0"
        # guarantee the technical report relies on for training stability.
        self.w = nn.Parameter(torch.zeros(hidden_size))
        self.key_norm = DepthRMSNorm(eps=eps)

    def forward(self, history: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            history: list of N tensors, each of shape [B, T, D]. history[0]
                is the token embedding (v_0); history[1:] are the raw
                (un-normalized) outputs of every preceding sublayer.

        Returns:
            Mixed hidden state h_l of shape [B, T, D] -- the depth-attended
            input to the *current* sublayer.
        """
        # [N, B, T, D]. This stacking is the O(L*d) memory the paper
        # discusses; in vanilla (non-pipelined, non-recomputed) training it
        # overlaps entirely with activations already retained for backprop,
        # so it adds no *additional* memory over a normal residual stream.
        values = torch.stack(history, dim=0)
        keys = self.key_norm(values)  # RMSNorm(v_i), applied per-i

        # phi(w_l, v_i) = exp(w_l . RMSNorm(v_i))  -- computed here in log
        # space as raw dot-product logits; the exp() + normalization is
        # folded into softmax() below for numerical stability.
        logits = torch.einsum("d,nbtd->nbt", self.w, keys)          # [N, B, T]
        alpha = torch.softmax(logits, dim=0)                        # softmax over depth (N)

        # h_l = sum_i alpha_{i->l} * v_i  -- weighted combination of the RAW
        # (un-normalized) values, not the normalized keys.
        mixed = torch.einsum("nbt,nbtd->btd", alpha, values)
        return mixed


# --------------------------------------------------------------------------- #
# Standard Transformer sublayers
# --------------------------------------------------------------------------- #

class MultiHeadSelfAttention(nn.Module):
    """Standard bidirectional multi-head self-attention (no causal mask --
    this is an encoder). Uses ``F.scaled_dot_product_attention`` for a
    fused, memory-efficient kernel (Flash-Attention-style) so the library
    stays fast enough for a Colab T4."""

    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.dropout_p = config.attention_dropout_prob

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention_mask, if given, is a boolean tensor broadcastable to
        # [B, num_heads, T_query, T_key] with True == "attend to this key".
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """Standard two-layer GELU MLP (position-wise feed-forward sublayer)."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #

class TransformerLayer(nn.Module):
    """One PreNorm Transformer encoder block, comprising two sublayers
    (self-attention, then MLP), each with its own residual "entry point".

    Baseline (``use_attn_res=False``):
        h = h + Dropout( Attn( LN(h) ) )
        h = h + Dropout( MLP ( LN(h) ) )

    AttnRes (``use_attn_res=True``): each entry point is replaced by a
    depth-wise softmax mix over the *entire* running history of previous
    sublayer outputs (see ``AttentionResidualMix``), and every sublayer's
    raw output is appended to that history for subsequent layers to
    (selectively) attend back to:

        h_mix = AttnResMix_attn(history)
        a_out = Attn( LN(h_mix) );  history += [a_out]
        h_mix = AttnResMix_mlp(history)
        m_out = MLP ( LN(h_mix) );  history += [m_out]
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.use_attn_res = config.use_attn_res

        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.use_attn_res:
            self.attn_mix = AttentionResidualMix(config.hidden_size, eps=config.layer_norm_eps)
            self.mlp_mix = AttentionResidualMix(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        h: torch.Tensor,
        history: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            h: the "carried" hidden state [B, T, D]. For the baseline this
                is h_{l-1} and is used directly. For AttnRes it is ignored
                as an *input* (the depth mix recomputes the effective input
                from ``history``) but is still threaded through the call
                signature so both code paths share one interface.
            history: running list of all previous value vectors (embedding +
                every prior sublayer's raw output). Only read/appended to
                when ``use_attn_res=True``; passed through unchanged
                (and unused) otherwise.

        Returns:
            (new_h, new_history)
        """
        if self.use_attn_res:
            # ---- Self-attention sublayer ----
            h_mix = self.attn_mix(history)
            attn_out = self.dropout(self.attn(self.attn_norm(h_mix), attention_mask))
            history = history + [attn_out]

            # ---- MLP sublayer ----
            h_mix = self.mlp_mix(history)
            mlp_out = self.dropout(self.mlp(self.mlp_norm(h_mix)))
            history = history + [mlp_out]

            h = h_mix  # carried forward mainly for API symmetry / debugging
        else:
            # ---- Standard PreNorm identity residual ----
            attn_out = self.dropout(self.attn(self.attn_norm(h), attention_mask))
            h = h + attn_out

            mlp_out = self.dropout(self.mlp(self.mlp_norm(h)))
            h = h + mlp_out

        return h, history


# --------------------------------------------------------------------------- #
# Embeddings + full encoder stack
# --------------------------------------------------------------------------- #

class BertEmbeddings(nn.Module):
    """Word + learned absolute position embeddings. No token-type
    embeddings -- this library targets single-segment MLM pre-training on
    WikiText, so the sentence-pair NSP-style segment embedding is omitted
    to keep the architecture minimal and focused on the residual ablation."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, T = input_ids.shape
        position_ids = self.position_ids[:, :T]
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(embeddings)


class BertEncoder(nn.Module):
    """The full PreNorm encoder stack: embeddings -> N Transformer layers ->
    final LayerNorm, with an optional final AttnRes aggregation step.

    Note on the final aggregation: Figure 1(b) of the technical report shows
    a final depth-wise mix (with its own ``w``) feeding the "Output" node
    after the last sublayer, rather than just reading off the last
    sublayer's raw output. We replicate that here via ``self.output_mix``
    when ``use_attn_res=True``, so the model's final representation is
    itself a *learned, selective* summary of the whole depth history, not
    just whatever the very last MLP happened to produce.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList(TransformerLayer(config) for _ in range(config.num_hidden_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_attn_res:
            self.output_mix = AttentionResidualMix(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.embeddings(input_ids)

        sdpa_mask = None
        if attention_mask is not None:
            # attention_mask: [B, T] with 1 = real token, 0 = padding.
            # Broadcast to [B, 1, 1, T]; SDPA treats True == "may attend".
            sdpa_mask = attention_mask[:, None, None, :].bool()

        # v_0 = token embedding, the seed of the depth history.
        history: List[torch.Tensor] = [h] if self.config.use_attn_res else []

        for layer in self.layers:
            h, history = layer(h, history, attention_mask=sdpa_mask)

        if self.config.use_attn_res:
            h = self.output_mix(history)

        return self.final_norm(h)


# --------------------------------------------------------------------------- #
# MLM head + top-level model
# --------------------------------------------------------------------------- #

class BertMLMHead(nn.Module):
    """Standard BERT MLM head: dense -> GELU -> LayerNorm -> tied decoder."""

    def __init__(self, config: BertConfig, word_embedding_weight: nn.Parameter):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Weight tying: reuse the input word embedding matrix as the output
        # projection, standard practice for MLM heads (halves embedding
        # parameter count and tends to improve sample efficiency).
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = word_embedding_weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(self.act(self.dense(h)))
        return self.decoder(h) + self.bias


class BertForMaskedLM(nn.Module):
    """Top-level model: encoder + MLM head, with built-in loss computation."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.encoder = BertEncoder(config)
        self.mlm_head = BertMLMHead(config, self.encoder.embeddings.word_embeddings.weight)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """BERT-style initialization (N(0, 0.02^2) for weights, zeros for
        biases, ones/zeros for LayerNorm). Note this intentionally does
        *not* touch ``AttentionResidualMix.w``: that parameter's required
        zero-init is set explicitly in its own ``__init__`` and must remain
        untouched by any generic re-initialization pass."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        h = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.mlm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,  # -100 marks non-masked positions (no loss)
            )
        return {"loss": loss, "logits": logits}

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Convenience config factory
# --------------------------------------------------------------------------- #

def get_bert_medium_config(vocab_size: int, use_attn_res: bool, max_seq_length: int = 256) -> BertConfig:
    """BERT-Medium (Turc et al., 2019) dimensions, the default architecture
    this whole library is built around."""
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=max_seq_length,
        use_attn_res=use_attn_res,
    )
