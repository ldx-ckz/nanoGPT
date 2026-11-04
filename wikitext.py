"""Download WikiText-103 raw and write GPT-2 uint16 .npy shards.

Usage: python wikitext.py
Official train/validation/test splits become train/val/test files separately.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import re

from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np
import tiktoken
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token


def iter_articles(rows):
    """Reassemble line-based rows, keeping sections and blank lines in order."""
    lines = []
    for row in rows:
        text = row["text"]
        # Top-level headings are '= Title ='; '= = Section = =' stays inside.
        if re.fullmatch(r"= [^= ].* =", text.strip()) and lines:
            article = "".join(lines)
            if article.strip():
                yield article
            lines = []
        lines.append(text if text.endswith("\n") else text + "\n")
    article = "".join(lines)
    if article.strip():
        yield article


def tokenize(text):
    # Match fineweb.py: one EOT before each document, GPT-2 BPE, uint16 storage.
    tokens = np.asarray([eot] + enc.encode_ordinary(text), dtype=np.int32)
    assert tokens.min() >= 0 and tokens.max() < 2**16
    return tokens.astype(np.uint16)


def write_split(tokenized_articles, output_dir, split, shard_size, prefix="wikitext"):
    """Split even very long articles across any number of shards, without loss."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(shard_size, dtype=np.uint16)
    used = 0
    total = 0
    documents = 0
    files = []

    def flush(length):
        filename = f"{prefix}_{split}_{len(files):06d}.npy"
        path = output_dir / filename
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as file:
            np.save(file, buffer[:length], allow_pickle=False)
        os.replace(temporary, path)
        files.append(filename)

    with tqdm(desc=split, unit="tokens", unit_scale=True) as progress:
        for tokens in tokenized_articles:
            documents += 1
            offset = 0
            while offset < len(tokens):
                count = min(shard_size - used, len(tokens) - offset)
                buffer[used:used + count] = tokens[offset:offset + count]
                used += count
                offset += count
                total += count
                progress.update(count)
                if used == shard_size:
                    flush(used)
                    used = 0
        if used:
            flush(used)
    if not total:
        raise ValueError(f"No tokens produced for {split}")
    return {"documents": documents, "tokens": total, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "wikitext103")
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--revision", default="main", help="HF dataset commit or branch")
    args = parser.parse_args()
    if args.shard_size < 1 or args.workers < 1:
        parser.error("--shard-size and --workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("wikitext_*.npy")):
        parser.error("Output already contains shards; choose a new --output-dir")

    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    print(f"Downloading {DATASET}/{CONFIG} at {revision}", flush=True)
    dataset = load_dataset(DATASET, CONFIG, revision=revision)
    manifest = {"dataset": DATASET, "config": CONFIG, "revision": revision,
                "tokenizer": "gpt2", "eot_token": eot, "dtype": "uint16",
                "shard_size": args.shard_size, "splits": {}}
    for source_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        articles = iter_articles(dataset[source_split])
        if args.workers == 1:
            info = write_split(map(tokenize, articles), args.output_dir, split, args.shard_size)
        else:
            with mp.Pool(args.workers) as pool:
                info = write_split(pool.imap(tokenize, articles, chunksize=16),
                                   args.output_dir, split, args.shard_size)
        manifest["splits"][split] = {"source_split": source_split, **info}
        print(f"{split}: {info['documents']:,} articles, {info['tokens']:,} tokens, "
              f"{len(info['files'])} shards", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finished: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
