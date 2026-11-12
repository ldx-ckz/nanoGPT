# Datasets, training and KV-cache prediction

Run commands from the repo root in the project Python environment.

## Prepare data

```bash
python wikitext.py
python tinystories.py --workers 4
```

TinyStories source: https://huggingface.co/datasets/roneneldan/TinyStories
The default training subset contains up to 50 million GPT-2 tokens, keeping
whole stories in source order. The last story exceeding the budget is omitted.
`--train-tokens 0` prepares the complete training split. The HF downloader may
cache the full source even when preparing only a subset.

There is no official test split. Official validation stories are deterministically
divided by stripped-text SHA256 parity: even -> val, odd -> test. Exact duplicates
are grouped/deduplicated; exact matches with held-out text are excluded from train.
This is a custom split, not an official TinyStories test benchmark. The resolved
dataset revision, split rule and actual token counts are recorded in manifest.json.

Both datasets use the same GPT-2 BPE, EOT prefix and uint16 .npy shard format.
WikiText files live in `wikitext103/`; TinyStories files in `tinystories/`.

## Train (Linux, two GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 GPT.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 GPT.py --data-dir tinystories --output-dir runs/tinystories --train-tokens 150000000
```

Training defaults to `--data-dir wikitext103` and `--output-dir runs/wikitext103`.
Set both options as shown above for TinyStories. TinyStories preprocessing's
50M unique-token subset is distinct from the cumulative training token budget:
150M training tokens is roughly three passes; the default 300M is roughly six.
Only train/val are read in training; test is held out.

Training and validation explicitly call:

```python
logits, loss, _ = model(x, y, past_kv=None, use_cache=False)
```

KV-cache is used in autoregressive inference, not across independent training batches.
Optional preflight: `python scripts/check_training.py --dataset tinystory`.

## Predict and measure cache effects

```bash
python predict.py --dataset wikitext --prompt "The history of science" --max-new-tokens 128
python predict.py --dataset tinystory --prompt "Once upon a time," --max-new-tokens 128
```

The script finds the latest checkpoint in the selected dataset's default run
directory. For a custom run, use an explicit checkpoint:

```bash
python predict.py --dataset tinystory --checkpoint runs/tinystories/checkpoints/model_02289.pt --max-new-tokens 128 --repeats 3 --warmup 1
```

Dataset selection is checked against checkpoint metadata where available. The
network weights are determined by the checkpoint; changing --dataset does not
turn one trained model into another. No dataset download is needed for inference.

Both modes use eval(), inference_mode(), the same prompt/checkpoint, greedy
decoding and a fixed generation length (including EOT if generated). Padded
vocabulary rows above GPT-2 token ID 50256 are excluded. Prompt+generation must
fit within the trained block_size (1024 by default); no sliding-cache extension
is implemented. Generation is batch size one.

Each mode is warmed up, repeated three times, and order alternates. CUDA is
synchronized around each timed forward/generation step. Tokenization, text
decoding and logging occur outside measurement. FP32 is the CPU default and
BF16 autocast the CUDA default; use --precision fp32 for an FP32 comparison.

The default output is `<run>/prediction/`:

- `prediction_log.txt`: step timings, input/cache lengths, summaries and text.
- `steps.jsonl`: raw per-step metrics for each repeat and cache mode.
- `summary.json`: complete records, generated token IDs, text and median results.

Key measurements: total tokens/s, time to first token (TTFT), average subsequent
decode ms/token, total/decode speedup ratios, number of token positions processed,
logical K/V tensor bytes and peak allocated CUDA memory (including model and
temporaries). Cache length excludes the final generated token, which has not yet
been fed back to the model. Logical cache bytes do not include backing-storage
or allocator overhead. CPU peak CUDA memory is null.

The no-cache mode recomputes the complete prefix on each step; cache mode processes
the prompt once then feeds only the newest token. Both use the existing model
forward, including its LM head. This benchmark includes dynamic torch.cat cache
growth and GPU synchronization overhead; it is not an optimized serving benchmark.
Short sequences may show little benefit or even slow down. Speedup >1 indicates
an actual measured benefit, not a guaranteed improvement.

`output_tokens_equal` reports whether all repeated greedy outputs match. Floating
point differences, especially BF16 near tied logits, may change the chosen token;
inspect this flag instead of assuming exact equality. KV-cache should change
computation reuse, not improve model language quality.

Each invocation overwrites the three prediction log files in its output directory;
use --output-dir to retain different prompts or benchmark settings.
