"""Measure token-level cross-entropy and perplexity on a WikiText split."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

import GPT as training
from predict import load_checkpoint_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path,
        default=training.PROJECT_DIR / training.DATA_DIRS["wikitext"],
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        parser.error("Use fp32 when evaluating without CUDA.")

    manifest_bytes = (args.data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    split = manifest["splits"][args.split]
    if not split["files"]:
        parser.error(f"No shards listed for split {args.split}.")
    tokens = torch.cat([
        training.load_tokens(args.data_dir / filename)
        for filename in split["files"]
    ])
    if tokens.numel() != split["tokens"] or tokens.numel() < 2:
        parser.error("Split token count differs from manifest or is below 2.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_checkpoint_model(checkpoint, device)
    if not 1 <= args.seq_len <= model.config.block_size:
        parser.error("Require 1 <= seq-len <= checkpoint block_size.")
    completed_steps = checkpoint.get("completed_steps")
    trained_tokens = checkpoint.get("trained_tokens")
    del checkpoint

    nll_sum = 0.0
    evaluated_tokens = 0
    with torch.inference_mode():
        for start in range(0, tokens.numel() - 1, args.seq_len):
            end = min(start + args.seq_len, tokens.numel() - 1)
            x = tokens[start:end].unsqueeze(0).to(device)
            targets = tokens[start + 1:end + 1].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=args.precision == "bf16",
            ):
                logits, _, _ = model(x, dsa_sparse=True)
            nll_sum += F.cross_entropy(logits[0].float(), targets, reduction="sum").item()
            evaluated_tokens += targets.numel()

    mean_nll = nll_sum / evaluated_tokens
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "attention": model.config.attention,
        "completed_steps": completed_steps,
        "trained_tokens": trained_tokens,
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tokenizer": manifest["tokenizer"],
        "split": args.split,
        "seq_len": args.seq_len,
        "precision": args.precision,
        "device": str(device),
        "evaluated_tokens": evaluated_tokens,
        "cross_entropy": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
    report = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
