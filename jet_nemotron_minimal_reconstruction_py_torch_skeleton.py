"""
Jet‑Nemotron (Minimal Reconstruction) — PyTorch Skeleton
--------------------------------------------------------------------------------
This is a practical, minimal PyTorch scaffold that mirrors the *ideas* described
in the Jet‑Nemotron paper: (1) compact, efficient transformer *JetBlocks*
composed of switchable sublayers, and (2) a *PostNAS* selection that swaps block
choices *after* an initial pretrain/fine‑tune using a small validation proxy.

⚠️ Disclaimer
- This is NOT the official NVLabs implementation.
- It’s designed to be simple, readable, and hackable for research.
- Replace/extend sublayers and scoring with your own implementations.

Tested: Python 3.10+, PyTorch 2.2+
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utility: Rotary Embeddings
# ---------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)


# ---------------------------
# Attention Variants
# ---------------------------
class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head * n_heads == d_model, "d_model must be divisible by n_heads"
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.empty(0), persistent=False)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self.mask.shape[:2] != (T, T):
            m = torch.full((T, T), float("-inf"), device=device)
            m = torch.triu(m, diagonal=1)
            self.mask = m
        return self.mask

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape and apply rotary
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        cos, sin = cos_sin
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]
        attn = attn + self._causal_mask(T, x.device)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.o(y))
        return y


class LinearAttention(nn.Module):
    """A simple linear attention (performer-style kernelized) placeholder.
    Replace with your preferred efficient attention.
    """
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_heads
        Dh = C // H
        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        q = self.feature_map(q)
        k = self.feature_map(k)
        kv = torch.einsum("bhtd,bhte->bhde", k, v)  # [B,H,Dh,Dh]
        z = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhtd,bhde,bht->bhte", q, kv, z)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(self.drop(out))


# ---------------------------
# MLP / FFN Variants
# ---------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        inner = expansion * d_model
        self.w1 = nn.Linear(d_model, inner, bias=bias)
        self.w2 = nn.Linear(d_model, inner, bias=bias)
        self.w3 = nn.Linear(inner, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(self.drop(x))
        return x


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        inner = expansion * d_model
        self.fc = nn.Linear(d_model, inner * 2, bias=bias)
        self.proj = nn.Linear(inner, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.fc(x).chunk(2, dim=-1)
        x = F.gelu(a) * torch.sigmoid(g)
        return self.proj(self.drop(x))


# ---------------------------
# Optional Local Mixing (Conv)
# ---------------------------
class DepthwiseConvMix(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.pad = kernel_size // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=self.pad)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> conv over T
        x = x.transpose(1, 2)
        x = self.pw(F.gelu(self.dw(x)))
        x = x.transpose(1, 2)
        return self.drop(x)


# ---------------------------
# JetBlock: switchable sublayers
# ---------------------------
@dataclass
class JetBlockConfig:
    d_model: int
    n_heads: int
    attn_type: str = "mha"          # {"mha", "linear"}
    mlp_type: str = "swiglu"         # {"swiglu", "gated"}
    conv_kernel: int = 0             # 0 disables conv mix
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class JetBlock(nn.Module):
    def __init__(self, cfg: JetBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        if cfg.attn_type == "mha":
            self.attn = MHA(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        elif cfg.attn_type == "linear":
            self.attn = LinearAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        else:
            raise ValueError(f"Unknown attn_type: {cfg.attn_type}")

        if cfg.mlp_type == "swiglu":
            self.mlp = SwiGLU(cfg.d_model, expansion=4, dropout=cfg.dropout)
        elif cfg.mlp_type == "gated":
            self.mlp = GatedMLP(cfg.d_model, expansion=4, dropout=cfg.dropout)
        else:
            raise ValueError(f"Unknown mlp_type: {cfg.mlp_type}")

        self.conv = None
        if cfg.conv_kernel and cfg.conv_kernel > 0:
            self.conv = DepthwiseConvMix(cfg.d_model, kernel_size=cfg.conv_kernel, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Attn
        a = self.ln1(x)
        a = self.attn(a, cos_sin)
        x = x + a

        # Optional local mixing before FFN
        if self.conv is not None:
            x = x + self.conv(self.ln2(x))

        # FFN
        m = self.mlp(self.ln2(x))
        x = x + m
        return x


# ---------------------------
# Jet‑Nemotron Minimal Model
# ---------------------------
@dataclass
class JetNemotronConfig:
    vocab_size: int
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    attn_type: str = "mha"
    mlp_type: str = "swiglu"
    conv_kernel: int = 0
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True


class JetNemotron(nn.Module):
    def __init__(self, cfg: JetNemotronConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rotary = RotaryEmbedding(cfg.d_model // cfg.n_heads)
        block_cfg = JetBlockConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            attn_type=cfg.attn_type,
            mlp_type=cfg.mlp_type,
            conv_kernel=cfg.conv_kernel,
            dropout=cfg.dropout,
        )
        self.blocks = nn.ModuleList([JetBlock(block_cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        B, T = idx.shape
        x = self.tok(idx)
        cos_sin = self.rotary(T, device=idx.device)
        for blk in self.blocks:
            x = blk(x, cos_sin)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            T = idx.shape[1]
            logits = self.forward(idx)[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------
# Training & Proxy PPL Evaluation
# ---------------------------
@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def train_step(model: JetNemotron, opt: torch.optim.Optimizer, 
               batch: Tuple[torch.Tensor, torch.Tensor], cfg: TrainConfig) -> float:
    model.train()
    x, y = batch
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.item())


@torch.no_grad()
def ppl_on_batch(model: JetNemotron, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
    model.eval()
    x, y = batch
    logits = model(x)
    loss = cross_entropy(logits, y)
    return math.exp(float(loss.item()))


# ---------------------------
# PostNAS (late selection)
# ---------------------------
"""
Idea: After a warmup/pretrain of the base model, evaluate a *candidate set* of
block configurations on a small proxy validation set. Select the best per-layer
(or per‑group) and then *swap* blocks accordingly, followed by a brief
post‑tuning phase.

Below is a simple per‑layer greedy selection using proxy perplexity.
Replace the candidate pool with your own search space (attn/ffn/conv variants).
"""

JetCandidate = Dict[str, Any]  # e.g., {"attn_type": "mha", "mlp_type": "swiglu", "conv_kernel": 7}


def clone_block_with(model: JetNemotron, layer_idx: int, cand: JetCandidate) -> JetBlock:
    base_cfg = model.blocks[layer_idx].cfg
    new_cfg = dataclasses.replace(
        base_cfg,
        attn_type=cand.get("attn_type", base_cfg.attn_type),
        mlp_type=cand.get("mlp_type", base_cfg.mlp_type),
        conv_kernel=cand.get("conv_kernel", base_cfg.conv_kernel),
    )
    return JetBlock(new_cfg).to(next(model.parameters()).device)


@torch.no_grad()
def evaluate_layer_swap(model: JetNemotron, layer_idx: int, new_block: JetBlock,
                        val_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
    # Swap in-memory (non‑destructive): forward with a temporary replacement
    orig = model.blocks[layer_idx]
    model.blocks[layer_idx] = new_block
    ppl = ppl_on_batch(model, val_batch)
    model.blocks[layer_idx] = orig
    return ppl


def postnas_select(
    model: JetNemotron,
    candidate_space: List[JetCandidate],
    val_batch: Tuple[torch.Tensor, torch.Tensor],
) -> List[JetCandidate]:
    """Greedy per‑layer selection minimizing perplexity on a proxy batch."""
    chosen: List[JetCandidate] = []
    for i, blk in enumerate(model.blocks):
        best_ppl = float("inf")
        best_cand = {}
        for cand in candidate_space:
            new_block = clone_block_with(model, i, cand)
            ppl = evaluate_layer_swap(model, i, new_block, val_batch)
            if ppl < best_ppl:
                best_ppl, best_cand = ppl, cand
        chosen.append(best_cand)
        # Commit the best candidate to the model
        model.blocks[i] = clone_block_with(model, i, best_cand)
        print(f"[PostNAS] Layer {i}: chose {best_cand} -> proxy PPL={best_ppl:.3f}")
    return chosen


# ---------------------------
# Tiny Demo
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Toy vocab & data (random integers) — replace with real tokenizer/data.
    V = 32000
    T = 128
    B = 4
    torch.manual_seed(0)
    x = torch.randint(0, V, (B, T), device=device)
    y = torch.roll(x, shifts=-1, dims=1)

    base_cfg = JetNemotronConfig(
        vocab_size=V,
        d_model=512,
        n_heads=8,
        n_layers=6,
        attn_type="mha",
        mlp_type="swiglu",
        conv_kernel=0,
        max_seq_len=T,
        dropout=0.1,
        tie_embeddings=True,
    )
    model = JetNemotron(base_cfg).to(device)

    # Warmup steps (very small, just to make PostNAS differentiating)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(20):
        loss = train_step(model, opt, (x, y), TrainConfig())
        if (step + 1) % 5 == 0:
            print(f"warmup step {step+1}: loss={loss:.3f}")

    # Candidate space for PostNAS
    candidate_space: List[JetCandidate] = [
        {"attn_type": "mha", "mlp_type": "swiglu", "conv_kernel": 0},
        {"attn_type": "mha", "mlp_type": "gated",  "conv_kernel": 0},
        {"attn_type": "linear", "mlp_type": "swiglu", "conv_kernel": 0},
        {"attn_type": "mha", "mlp_type": "swiglu", "conv_kernel": 7},
    ]

    # Run PostNAS on a small proxy batch (same as our toy batch here)
    chosen = postnas_select(model, candidate_space, (x, y))
    print("Chosen candidates per layer:", chosen)

    # Quick post‑tune
    for step in range(10):
        loss = train_step(model, opt, (x, y), TrainConfig(lr=2e-4))
    print("post‑tune done; proxy ppl:", ppl_on_batch(model, (x, y)))

    # Sample generation (greedy, toy)
    start = torch.randint(0, V, (1, 1), device=device)
    out = model.generate(start, max_new_tokens=16)
    print("sample ids:", out.tolist()[0])
