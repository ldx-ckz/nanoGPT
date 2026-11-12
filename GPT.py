from torch.distributed import init_process_group, destroy_process_group
import argparse
from bisect import bisect_right
import json
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import inspect
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIRS = {
    "wikitext": "wikitext103",
    "tinystory": "tinystories",
}


class RoPE(nn.Module):
    def __init__(self, config, base=10000.0, head_dim=None):
        super().__init__()
        if head_dim is None:
            head_dim = config.n_embed//config.n_head
        assert head_dim % 2 == 0
        inv_freq = base**(
            -torch.arange(0, head_dim, 2, dtype=torch.float32)
            / head_dim
        )
        pos = torch.arange(config.block_size, dtype=torch.float32)
        angels = pos[:, None]*inv_freq[None, :]
        self.register_buffer(
            "cos_cached",
            angels.cos()[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            angels.sin()[None, None, :, :],
            persistent=False,
        )

    def forward(self, x, offset=0):
        T = x.size(-2)
        end = T+offset
        assert end <= self.cos_cached.size(-2)
        cos = self.cos_cached[:, :, offset:end, :]
        sin = self.sin_cached[:, :, offset:end, :]
        x_float = x.float()
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]
        rot_even = x_even*cos-x_odd*sin
        rot_odd = x_even*sin+x_odd*cos

        rot = torch.stack(
            (rot_even, rot_odd), dim=-1
        ).flatten(-2)
        return rot.to(dtype=x.dtype)

class DSAIndexer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_head = config.dsa_index_heads
        self.head_dim = config.dsa_index_dim
        self.rope_dim = config.mla_rope_dim
        assert self.rope_dim % 2 == 0
        self.q_proj = nn.Linear(config.n_embed,self.n_head*self.head_dim,bias=False)
        self.k_proj = nn.Linear(config.n_embed,self.head_dim,bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weight_proj = nn.Linear(config.n_embed,self.n_head,bias=False)
        self.rope = RoPE(config,head_dim=self.rope_dim)
    def forward(self,x,past_key = None,offset=0):
        x = x.detach()
        B,T,_ = x.size()
        q = self.q_proj(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = self.k_norm(self.k_proj(x)).unsqueeze(1)
        q = torch.cat((
            self.rope(q[...,:self.rope_dim],offset=offset),
            q[...,self.rope_dim:]
        ),dim=-1)
        k = torch.cat((
            self.rope(k[...,:self.rope_dim],offset=offset),
            k[...,self.rope_dim:]
        ),dim=-1)
        if past_key is not None:
            k = torch.cat((past_key,k),dim=-2)
        weights = self.weight_proj(x).transpose(1,2).unsqueeze(-1)
        with torch.autocast(device_type=x.device.type,enabled=False):
            similarity = q.float()@k.float().transpose(-2,-1)
            similarity = F.relu(similarity*self.head_dim**-0.5)
            scores = (similarity*weights.float()*self.n_head**-0.5).sum(dim=1)
        return scores,k



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed//config.n_head
        self.use_mla = config.attention in ("mla","dsa")
        self.use_dsa = config.attention == "dsa"
        if self.use_dsa:
            assert config.dsa_topk>0
            self.topk = config.dsa_topk
            self.indexer = DSAIndexer(config)
        if self.use_mla:
            self.kv_rank = config.mla_kv_rank
            self.rope_dim = config.mla_rope_dim
            assert self.kv_rank > 0
            assert 0 < self.rope_dim < self.head_dim
            assert self.rope_dim % 2 == 0
            self.cont_dim = self.head_dim-self.rope_dim
            self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)
            self.kv_down = nn.Linear(
                self.n_embed, self.kv_rank+self.rope_dim, bias=False)
            self.kv_norm = RMSNorm(self.kv_rank)
            self.kv_up = nn.Linear(self.kv_rank, self.n_head*(
                self.cont_dim+self.head_dim), bias=False)
        else:
            self.c_attn = nn.Linear(
                config.n_embed, config.n_embed*3, bias=False)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.c_proj.GPT2_SCALE_INIT = 1
        self.rope = RoPE(
            config, head_dim=self.rope_dim if self.use_mla else None)

    def _mla_qkv(self, x, past_kv, past_len, use_cache):
        B, T, _ = x.size()
        q = self.q_proj(x).view(B, T, self.n_head,
                                self.head_dim).transpose(1, 2)
        q_cont, q_rope = q.split((self.cont_dim, self.rope_dim), dim=-1)
        q_rope = self.rope(q_rope, offset=past_len)
        latent, k_rope = self.kv_down(x).split(
            (self.kv_rank, self.rope_dim), dim=-1)
        latent = self.kv_norm(latent)
        k_rope = self.rope(k_rope.unsqueeze(1), offset=past_len)
        if past_kv is not None:
            past_latent, past_k_rope = past_kv
            latent = torch.cat((past_latent, latent), dim=-2)
            k_rope = torch.cat((past_k_rope, k_rope), dim=-2)
        new_cache = (latent, k_rope) if use_cache else None
        S = latent.size(-2)
        kv = self.kv_up(latent).reshape(B, S, self.n_head, -1).transpose(1, 2)
        k_cont, v = kv.split((self.cont_dim, self.head_dim), dim=-1)
        q = torch.cat((q_cont, q_rope), dim=-1)
        k = torch.cat((k_cont, k_rope.expand(-1, self.n_head, -1, -1)), dim=-1)
        return q, k, v, new_cache

    def _dsa_attention(self,x,past_kv,use_cache,dsa_sparse,return_aux_loss):
        B,T,C = x.size()
        past_len = 0 if past_kv is None else past_kv[0].size(-2)
        mla_past = None if past_kv is None else past_kv[:2]
        index_past = None if past_kv is None else past_kv[2]
        q,k,v,mla_cache = self._mla_qkv(x,mla_past,past_len,use_cache)
        scores,index_k = self.indexer(x,past_key=index_past,offset=past_len)
        S = k.size(-2)
        q_pos = past_len+torch.arange(T,device=x.device)
        k_pos = torch.arange(S,device=x.device)
        causal = k_pos[None,:]<=q_pos[:,None]
        allowed = causal.unsqueeze(0).expand(B,-1,-1)
        if dsa_sparse:
            r_scores = scores.detach().masked_fill(~allowed,-torch.inf)
            indices = r_scores.argsort(
                dim=-1,descending=True,stable=True
            )[...,:min(self.topk,S)]
            selected = torch.zeros_like(allowed)
            selected.scatter_(-1,indices,True)
            allowed = selected&allowed
        aux_loss = x.new_zeros((),dtype=torch.float32)
        if return_aux_loss:
            with torch.autocast(device_type=x.device.type,enabled=False):
                with torch.no_grad():
                    teacher = torch.zeros_like(scores,dtype=torch.float32)
                    for head in range(self.n_head):
                        attention_scores = (q[:,head].float()@\
                                            k[:,head].float().transpose(-2,-1))*\
                                            self.head_dim**-0.5
                        attention_scores = attention_scores.masked_fill(
                            ~allowed,-torch.inf
                        )
                        teacher += attention_scores.softmax(dim=-1)
                    teacher /= self.n_head
                index_log_probs = scores.float().masked_fill(
                        ~allowed,-torch.inf
                    ).log_softmax(dim=-1)
                index_log_probs = index_log_probs.masked_fill(~allowed, 0.0)
                aux_loss = F.kl_div(
                    index_log_probs,teacher,reduction="sum"
                )/(B*T)
        y = F.scaled_dot_product_attention(
            q,k,v,attn_mask=allowed.unsqueeze(1),
            is_causal=False
        )
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        new_cache = None
        if use_cache:
            new_cache = (*mla_cache,index_k)
        return y,new_cache,aux_loss

    def forward(self, x, past_kv=None, use_cache=False,dsa_sparse=True,return_aux_loss=False):
        if self.use_dsa:
            return self._dsa_attention(
            x, past_kv, use_cache, dsa_sparse, return_aux_loss)   
        B, T, C = x.size()
        past_len = 0 if past_kv is None else past_kv[0].size(-2)
        if self.use_mla:
            q, k, v, new_cache = self._mla_qkv(x, past_kv, past_len, use_cache)
        else:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embed, dim=-1)
            q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
            q = self.rope(q, offset=past_len)
            k = self.rope(k, offset=past_len)
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)
            new_cache = (k, v) if use_cache else None
        if past_len == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            q_pos = past_len+torch.arange(T, device=x.device)
            k_pos = torch.arange(k.size(-2), device=x.device)
            mask = k_pos[None, :] <= q_pos[:, None]
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_cache,x.new_zeros((),dtype=torch.float32)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.n_embed
        hidden_dim = 8 * dim // 3

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        self.down_proj.GPT2_SCALE_INIT = 1

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swg = SwiGLU(config)

    def forward(self, x):
        return self.swg(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True)
                                             + self.eps)
        return (normalized * self.weight).to(x.dtype)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False,dsa_sparse=True,return_aux_loss=False):
        attn_out, new_cache,aux_loss= self.attn(
            self.ln_1(x), past_kv=past_kv, use_cache=use_cache,
            dsa_sparse=dsa_sparse,return_aux_loss=return_aux_loss)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache,aux_loss


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_head: int = 12
    n_embed: int = 768
    attention: str = "mha"
    mla_kv_rank: int = 192
    mla_rope_dim: int = 16
    dsa_index_heads: int = 2
    dsa_index_dim: int = 32
    dsa_topk: int = 128


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=RMSNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT2_SCALE_INIT"):
                std *= (2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False,dsa_sparse=True, return_aux_loss=False):
        B, T = idx.size()
        n_layers = len(self.transformer.h)
        if past_kv is None:
            past_kv = [None]*n_layers
        else:
            assert len(past_kv) == n_layers

        past_lengths = [
            0 if cache is None else cache[0].size(-2)
            for cache in past_kv
        ]
        assert len(set(past_lengths)) == 1
        past_len = past_lengths[0]

        assert past_len+T <= self.config.block_size, (f"Cannot forward sequence of length {past_len+T}, "
                                                      f"block size is only {self.config.block_size}")
        x = self.transformer.wte(idx)
        new_past_kv = [] if use_cache else None
        aux_loss = x.new_zeros((), dtype=torch.float32)
        for layer_idx, block in enumerate(self.transformer.h):
            x, layer_cache,layer_aux = block(
                x, past_kv=past_kv[layer_idx], use_cache=use_cache,dsa_sparse=dsa_sparse,return_aux_loss=return_aux_loss,)
            aux_loss += layer_aux
            if use_cache:
                new_past_kv.append(layer_cache)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_aux_loss:
            return logits, loss, new_past_kv, aux_loss / n_layers
        return logits, loss, new_past_kv

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and str(device).startswith("cuda")
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(
            0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val', 'test'}
        data_root = Path(
            data_root) if data_root is not None else PROJECT_DIR / 'wikitext103'
        shards = sorted(data_root.glob(f'*_{split}_*.npy'))
        self.shards = shards
        assert len(shards) > 0, f"no shards found for {split}"
        if master_process:
            print(f"found {len(shards)} shards for {split}")
        self.ends = []
        total = 0
        for path in shards:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.ndim != 1 or len(array) == 0:
                raise ValueError(f'Invalid token shard: {path}')
            total += len(array)
            self.ends.append(total)
            del array
        self.total_tokens = total
        if total < B*T*num_processes + 1:
            raise ValueError(
                f'{split} has {total} tokens; batch needs at least {B*T*num_processes + 1}')
        self.reset()

    def reset(self):
        self.position = 0
        self.cur_shard = 0
        self.tokens = load_tokens(self.shards[self.cur_shard])

    def _read_tokens(self, start, count):
        parts = []
        while count:
            start %= self.total_tokens
            shard = bisect_right(self.ends, start)
            if shard != self.cur_shard:
                self.cur_shard = shard
                self.tokens = load_tokens(self.shards[shard])
            shard_start = self.ends[shard - 1] if shard else 0
            offset = start - shard_start
            take = min(count, len(self.tokens) - offset)
            parts.append(self.tokens[offset:offset + take])
            start += take
            count -= take
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def next_batch(self):
        B, T = self.B, self.T
        buf = self._read_tokens(self.position + B*T*self.process_rank, B*T + 1)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.position = (self.position + B*T *
                         self.num_processes) % self.total_tokens
        return x, y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the GPT baseline on WikiText-103")
    parser.add_argument("--data-dir", default=DATA_DIRS["wikitext"],
                        help="Shard directory, relative to the project or absolute")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="Sequences per GPU")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--total-batch-size", type=int,
                        default=65536, help="Global tokens per optimizer update")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-steps", type=int)
    budget.add_argument("--train-tokens", type=int,
                        help="Rounded up to whole optimizer updates")
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--output-dir", default="runs/wikitext103",
                        help="Run directory; logs and checkpoints are stored inside")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--attention", choices=("mha", "mla","dsa"), default="mha")
    parser.add_argument("--dsa-warmup-steps", type=int, default=200)
    parser.add_argument("--dsa-loss-weight", type=float, default=1.0)
    parser.add_argument("--dsa-topk", type=int, default=128)
    args = parser.parse_args()
    if args.total_batch_size <= 0 or (args.train_tokens is not None and args.train_tokens <= 0):
        parser.error("Global batch size and token budget must be positive")
    if args.max_steps is None:
        args.train_tokens = args.train_tokens if args.train_tokens is not None else 300_000_000
        args.max_steps = math.ceil(args.train_tokens / args.total_batch_size)
    if args.warmup_steps is None:
        args.warmup_steps = min(
            args.max_steps - 1, math.ceil(args.max_steps * 0.03))
    if any(value <= 0 for value in (args.batch_size, args.total_batch_size, args.max_steps, args.eval_interval, args.val_batches, args.checkpoint_interval)):
        parser.error("Batch sizes, steps and intervals must be positive")
    if not 1 <= args.seq_len <= 1024 or not 0 <= args.warmup_steps < args.max_steps:
        parser.error(
            "Require 1 <= seq-len <= 1024 and 0 <= warmup-steps < max-steps")
    if args.max_lr <= 0 or (args.train_tokens is not None and args.train_tokens <= 0):
        parser.error("Learning rate and token budget must be positive")
    if args.dsa_warmup_steps < 0 or args.dsa_loss_weight <= 0 or args.dsa_topk <= 0:
        parser.error("Require nonnegative DSA warmup and positive loss weight/topk")
    return args


master_process = True


def main():
    global master_process
    args = parse_args()
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    total_batch_size = args.total_batch_size
    B = args.batch_size
    T = args.seq_len
    data_root = PROJECT_DIR / args.data_dir
    if not (data_root / "manifest.json").is_file():
        raise FileNotFoundError(
            "Run python wikitext.py first; complete manifest.json is required")
    assert total_batch_size % (B*T*ddp_world_size) == 0
    grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', data_root=data_root)
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', data_root=data_root)

    torch.set_float32_matmul_precision('high')

    raw_model = GPT(GPTConfig(
        vocab_size=50304, attention=args.attention, dsa_topk=args.dsa_topk,
    )).to(device)
    model = raw_model
    if os.environ.get("TORCH_COMPILE_DISABLE") != "1":
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    max_steps = args.max_steps
    max_lr = args.max_lr
    min_lr = max_lr*0.1
    warmup_steps = args.warmup_steps

    def get_lr(it):
        if it < warmup_steps:
            return max_lr*(it+1)/warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it-warmup_steps)/(max_steps-warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio))
        return min_lr + coeff*(max_lr-min_lr)

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=max_lr, device=device)

    run_dir = PROJECT_DIR / args.output_dir
    log_dir = run_dir / "log"
    checkpoints_dir = run_dir / "checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    if master_process:
        (run_dir / "training_config.json").write_text(json.dumps(vars(args),
                                                                 indent=2), encoding="utf-8")
        print(
            f"dataset: {data_root.name}; train tokens: {train_loader.total_tokens:,}; val tokens: {val_loader.total_tokens:,}")
        print(
            f"planned updates: {max_steps}; planned training tokens: {max_steps * total_batch_size:,}")
        with open(log_file, "w") as f:
            pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps-1)
        dsa_sparse = step >= args.dsa_warmup_steps
        if step % args.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = min(
                    args.val_batches, val_loader.total_tokens // (B*T*ddp_world_size))
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss, _ = model(x, y,dsa_sparse=dsa_sparse)
                    loss = loss/val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        index_loss_accum = torch.zeros((),device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps-1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, ce_loss, _ ,index_loss= model(x, y,dsa_sparse=dsa_sparse,return_aux_loss=True)
                total_loss = ce_loss+args.dsa_loss_weight*index_loss
            loss_accum += ce_loss.detach()/grad_accum_steps
            index_loss_accum += index_loss.detach()/grad_accum_steps
            (total_loss / grad_accum_steps).backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(index_loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * \
            train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | index_kl: {index_loss_accum.item():.6f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
                f.write(f"{step} index_kl {index_loss_accum.item():.6f}\n")
            if (step + 1) % args.checkpoint_interval == 0 or last_step:
                checkpoints_path = checkpoints_dir / f"model_{step + 1:05d}.pt"
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': asdict(raw_model.config),
                    'step': step,
                    'completed_steps': step + 1,
                    'training_config': vars(args),
                    'trained_tokens': (step + 1) * total_batch_size,
                }
                torch.save(checkpoint, checkpoints_path)
                print(f"saved checkpoint: {checkpoints_path}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
