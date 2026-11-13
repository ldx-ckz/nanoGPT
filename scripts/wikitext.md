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

## Four-group ablation on WikiText-103

| Group | `--attention` | Architecture |
| --- | --- | --- |
| A | `gpt2` | GPT-2 with learned absolute positions, LayerNorm and GELU |
| B | `mha` | Current RMSNorm, SwiGLU and RoPE backbone with MHA |
| C | `mla` | Current backbone with MLA |
| D | `dsa` | Current backbone with MLA and DSA |

A follows this repository's `GPT-2.py` at
[`9de41b7`](https://github.com/ldx-ckz/nanoGPT-2/commit/9de41b75b4c2562d0b1e06aef8d11bc3312e5dfa),
the WikiText-103 version before the normalization, FFN and RoPE changes.
The architecture follows [Karpathy's `6104ab1`](https://github.com/karpathy/build-nanogpt/blob/6104ab1b53920f6e2159749676073ff7d815c1fa/train_gpt2.py).
Run all four groups through the current `GPT.py` from random initialization.
D's 153 dense warmup updates count toward the same training budget.

Use the same prepared WikiText-103 shards and GPT-2 BPE across all groups.
The launcher records the dataset revision and manifest hash in each run's
`experiment.json`. Training uses the train split and monitors the validation
split. Reserve the test split for final checkpoint evaluation with the shared
evaluator once it is available.

The starting configuration is two GPUs, four sequences per GPU, sequence length
1024 and two accumulation steps per optimizer update. This gives 16,384 tokens
per update and 3,052 updates (50,003,968 tokens) per group. All groups use seed
1337, a peak learning rate of `6e-4`, and eager execution. D uses top-k 128 and
KL weight 1.0. Run a short check on the target GPUs before starting full runs.

From the repo root, preview the commands with PowerShell:

```powershell
.\scripts\ablation.ps1
.\scripts\ablation.ps1 -Group D
```

On the Linux server, activate the project's Python environment and use PowerShell
7 (`pwsh`). Once the experiment preparation is complete, launch one group at a
time from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 pwsh -NoProfile -File scripts/ablation.ps1 -Group A -Run
```

Repeat for B, C and D. Each group writes to its own directory under
`runs/ablation/wikitext-seed1337/`.
