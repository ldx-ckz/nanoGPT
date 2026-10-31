import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    embed_size: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.n_heads == 0
        self.c_attn = nn.Linear(config.embed_size, 3*config.embed_size)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        self.c_proj.GPT2_SCALE_INIT = 1
        self.n_heads = config.n_heads
        self.embed_size = config.embed_size

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_size, dim=2)
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_size, 4*config.embed_size)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.embed_size, config.embed_size)
        self.c_proj.GPT2_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_proj(self.gelu(x))
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_size),
            wpe=nn.Embedding(config.block_size, config.embed_size),
            h=nn.Modulelist([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, y=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb+tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
