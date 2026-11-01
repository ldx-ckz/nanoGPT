"""Check dependencies, shard headers, rank boundaries and one small optimizer step.

Run from any directory: python scripts/check_training.py
This does not start the full training loop or download datasets.
"""
import argparse
import importlib.util
from dataclasses import asdict, dataclass
import inspect
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def load_training_definitions():
    source = PROJECT_DIR / "GPT-2.py"
    spec = importlib.util.spec_from_file_location("gpt_training", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.master_process = False
    return vars(module)


def check_shards(data_root, splits=("train", "val", "test")):
    counts = {}
    total_tokens = 0
    for split in splits:
        split_tokens = 0
        paths = sorted(data_root.glob(f"*_{split}_*.npy"))
        assert paths, f"No {split} shards found"
        for path in paths:
            tokens = np.load(path, mmap_mode="r", allow_pickle=False)
            assert tokens.ndim == 1 and tokens.dtype == np.uint16, str(path)
            assert len(tokens) > 0, f"Empty shard: {path}"
            sample = tokens[:4096]
            assert int(sample.max()) < 50257, f"Invalid GPT-2 token in {path}"
            total_tokens += len(tokens)
            split_tokens += len(tokens)
            del tokens
        assert split_tokens >= 16 * 1024 * 2 + 1, f"Split too short for default two-GPU batch: {split}"
        counts[split] = len(paths)
    print(f"Shard headers and token samples OK: {counts}, {total_tokens:,} total tokens")


def check_rank_boundaries(namespace):
    loader_type = namespace["DataLoaderLite"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "wikitext103").mkdir()
        for shard in range(2):
            np.save(root / "wikitext103" / f"test_train_{shard:06d}.npy",
                    np.arange(51, dtype=np.uint16) + shard * 100)
        original_root = namespace["PROJECT_DIR"]
        namespace["PROJECT_DIR"] = root
        try:
            loaders = [loader_type(2, 4, rank, 2, "train") for rank in range(2)]
            stream = torch.cat([torch.arange(51), torch.arange(100, 151)])
            for batch in range(12):
                for rank, loader in enumerate(loaders):
                    x, y = loader.next_batch()
                    indices = (torch.arange(9) + batch * 16 + rank * 8) % len(stream)
                    expected = stream[indices]
                    assert torch.equal(x.flatten(), expected[:-1])
                    assert torch.equal(y.flatten(), expected[1:])
                assert loaders[0].position == loaders[1].position
        finally:
            namespace["PROJECT_DIR"] = original_root
    print("Two-rank shard boundaries and shifted targets OK (simulated on CPU)")


def check_optimizer_and_checkpoint(namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = namespace["GPTConfig"](block_size=8, vocab_size=50304,
                                    n_layers=2, n_head=2, n_embed=32)
    model = namespace["GPT"](config).to(device)
    optimizer = model.configure_optimizers(0.1, 6e-4, device)
    x = torch.randint(0, 50257, (2, 8), device=device)
    y = torch.randint(0, 50257, (2, 8), device=device)
    parameter = model.transformer.h[0].attn.c_attn.weight
    before = parameter.detach().clone()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        _, loss = model(x, y)
    assert torch.isfinite(loss), "Nonfinite loss"
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm), "Nonfinite gradients"
    optimizer.step()
    assert not torch.equal(before, parameter), "Optimizer did not update weights"
    model.eval()
    with torch.no_grad():
        expected, _ = model(x)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "config": asdict(config)}, path)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        restored = namespace["GPT"](namespace["GPTConfig"](**checkpoint["config"])).to(device)
        restored.load_state_dict(checkpoint["model"])
        restored.eval()
        with torch.no_grad():
            actual, _ = restored(x)
        torch.testing.assert_close(actual, expected)
    print(f"Small-model forward/backward, AdamW update and weight reload OK on {device}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}; visible CUDA GPUs: {torch.cuda.device_count()}")
    namespace = load_training_definitions()
    data_root = args.data_dir or PROJECT_DIR / "wikitext103"
    check_shards(data_root)
    check_rank_boundaries(namespace)
    check_optimizer_and_checkpoint(namespace)
    print("Training preflight passed. Real multi-GPU NCCL training must run on Linux.")


if __name__ == "__main__":
    main()
