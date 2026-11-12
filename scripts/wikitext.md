# WikiText-103 training

Source: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext),
configuration `wikitext-103-raw-v1`. Run commands from the repository root using
the Python environment containing `requirements.txt` dependencies.

## Download and preprocess

```bash
python wikitext.py
```

The script downloads the official train, validation and test sets. It joins
line-based rows into articles, preserves their order, sections and blank lines,
and prefixes each article with GPT-2's EOT token.
GPT-2 BPE tokens are stored as uint16 NumPy arrays. A shard contains at most
1,000,000 tokens; the final shard may be shorter. Documents may cross shard
boundaries; splits never cross those boundaries.

```text
wikitext103/
  manifest.json
  wikitext_train_000000.npy
  wikitext_train_000001.npy
  ...
  wikitext_val_000000.npy
  wikitext_test_000000.npy
```

`manifest.json` records the resolved dataset commit, tokenizer, EOT token,
article counts, token counts and filenames per split. It is written only after
all three splits finish. The HF download cache is separate from these arrays.
Do not remove the cache while preprocessing is running.

Optional controls:

```bash
python wikitext.py --workers 4 --shard-size 1000000 --output-dir wikitext103
```

Use `--workers 1` for sequential preprocessing. Existing WikiText shards cause
an error to avoid mixing outputs; use a new output directory for another run.

## Check and train

```bash
python scripts/check_training.py
```

Linux, two NVIDIA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 GPT.py --output-dir runs/wiki_baseline_seed1337
```

Single GPU:

```bash
python GPT.py --output-dir runs/wiki_baseline_seed1337
```

WikiText defaults:

| Option | Value |
| --- | --- |
| Model | GPT-2 variant: RMSNorm, SwiGLU, RoPE; 12 layers, 12 heads, width 768, vocabulary 50304 |
| Per-GPU batch / sequence length | 16 / 1024 |
| Global batch | 65536 tokens (two accumulation steps on two GPUs) |
| Training token budget | 300,000,000, rounded up to 4578 updates / 300,023,808 tokens |
| Peak learning rate | 6e-4 |
| Warmup | 3% of updates, rounded up |
| Validation | Initially, every 100 updates, and before the last update |
| Validation batches | Up to 20, capped at one pass of complete global batches |
| Checkpoints | Every 500 completed updates and after the final update |

Override the budget or learning rate, for example:

```bash
python GPT.py --train-tokens 50000000 --max-lr 3e-4 --output-dir runs/wiki_pilot_seed1337
```

`--max-steps` can replace `--train-tokens`; they cannot be supplied together.
Choose a different `--output-dir` for each experiment: reusing a run directory
overwrites its config, log and matching checkpoint filenames.

Each run stores `training_config.json`, `log/log.txt` and `checkpoints/*.pt`.
Checkpoints contain model weights, model config, completed updates and the
training token count. They do not implement complete training-state resume.

Only train and val are read during training. The held-out test shards are for
final evaluation. The logged validation loss is a fixed-batch training monitor,
not full-test sliding-window perplexity or a published word-level WikiText score.
Article EOT/newline handling is part of this repository's preprocessing protocol.

For TinyStories preparation, dataset selection and autoregressive prediction
with KV-cache timing/memory logs, see [prediction.md](prediction.md).
