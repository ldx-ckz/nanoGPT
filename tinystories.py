"""Download TinyStories and write GPT-2 uint16 train/val/test shards."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from wikitext import tokenize, write_split, eot

ROOT = Path(__file__).resolve().parent
DATASET = "roneneldan/TinyStories"


def text_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def split_validation(rows):
    """Group exact duplicates together, then hash-split official validation."""
    groups = {"val": [], "test": []}
    seen = set()
    for row in rows:
        text = row["text"]
        digest = text_hash(text)
        if not text.strip() or digest in seen:
            continue
        seen.add(digest)
        split = "val" if int(digest, 16) % 2 == 0 else "test"
        groups[split].append(text)
    return groups, seen


def training_texts(rows, held_out):
    for row in rows:
        text = row["text"]
        if text.strip() and text_hash(text) not in held_out:
            yield text


def limit_tokens(arrays, budget):
    """Keep whole stories. The last story that would exceed the budget is omitted."""
    total = 0
    for array in arrays:
        if budget and total + len(array) > budget:
            break
        total += len(array)
        yield array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tinystories")
    parser.add_argument("--train-tokens", type=int, default=50_000_000, help="Unique training subset budget; 0 = all")
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.train_tokens < 0 or args.shard_size < 1 or args.workers < 1:
        parser.error("Require nonnegative train-tokens and positive shard-size/workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("tinystories_*.npy")):
        parser.error("Output already contains shards; choose a new --output-dir")
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    print(f"Downloading {DATASET} at {revision}", flush=True)
    dataset = load_dataset(DATASET, revision=revision)
    heldout, hashes = split_validation(dataset["validation"])
    texts = {"train": training_texts(dataset["train"], hashes), **heldout}
    manifest = {"dataset": DATASET, "revision": revision, "tokenizer": "gpt2",
                "eot_token": eot, "dtype": "uint16", "shard_size": args.shard_size,
                "train_token_limit": args.train_tokens,
                "heldout_rule": "official validation: stripped-text SHA256 parity, even=val, odd=test; exact duplicates grouped",
                "splits": {}}
    for split in ("train", "val", "test"):
        pool = mp.Pool(args.workers) if args.workers > 1 else None
        try:
            arrays = pool.imap(tokenize, texts[split], chunksize=16) if pool else map(tokenize, texts[split])
            if split == "train":
                arrays = limit_tokens(arrays, args.train_tokens)
            info = write_split(arrays, args.output_dir, split, args.shard_size, prefix="tinystories")
            manifest["splits"][split] = {"source_split": "train" if split == "train" else "validation", **info}
            print(f"{split}: {info['documents']:,} stories, {info['tokens']:,} tokens", flush=True)
        finally:
            if pool:
                pool.terminate()
                pool.join()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finished: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
