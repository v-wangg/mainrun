import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    eos_id: int | None = None
    use_doc_mask: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        # Marks the residual-stream output projection so GPT._init_weights applies
        # the GPT-2 scaled init (std *= (2 * n_layer) ** -0.5). Read by GPT, not used here.
        self.proj.RESIDUAL_SCALE_INIT = True
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :] # (B, n_head, T, head_dim) each
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
        if attn_mask is not None:
            # attn_mask: (B, T, T) bool, True = attend, False = mask. Broadcasts over heads.
            att = att.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        else:
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_head, head_dim) -> (B, T, n_head * head_dim = d_model)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        # net[2] is the d_model-out projection that lands back in the residual stream.
        # Marked for GPT-2 scaled init in GPT._init_weights.
        self.net[2].RESIDUAL_SCALE_INIT = True
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x, attn_mask: torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.eos_id = cfg.eos_id
        # Belt-and-suspenders: if eos_id is missing, silently fall back to standard
        # causal attention rather than crash. train.py logs the resolved state.
        self.use_doc_mask = cfg.use_doc_mask and (cfg.eos_id is not None)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if isinstance(module, nn.Linear) and getattr(module, "RESIDUAL_SCALE_INIT", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def n_residual_proj(self) -> int:
        return sum(
            1 for m in self.modules()
            if isinstance(m, nn.Linear) and getattr(m, "RESIDUAL_SCALE_INIT", False)
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx) # (B, T, d_model)
        pos = self.pos_emb[:, :T, :] # (B, T, d_model)
        x = self.drop(tok + pos)

        # Document attention mask: each token attends only to tokens in the same
        # document, where documents are separated by <eos>. Convention: <eos>
        # belongs to its preceding document (predicted from in-headline content),
        # and the next token starts a fresh attention island.
        # cumsum(is_eos) - is_eos => document index that <eos> shares with its prior tokens.
        attn_mask = None
        if self.use_doc_mask:
            is_eos = (idx == self.eos_id).to(torch.int32)               # (B, T)
            doc_id = torch.cumsum(is_eos, dim=1) - is_eos               # (B, T)
            same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)       # (B, T, T)
            causal = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
            attn_mask = same_doc & causal                               # (B, T, T)

        for block in self.blocks: x = block(x, attn_mask=attn_mask) # (B, T, d_model)
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
