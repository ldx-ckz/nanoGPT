"""Autoregressive generation and KV-cache benchmark for a trained checkpoint."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import tiktoken
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prediction_training", ROOT / "GPT-2.py")
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, prompt, new_tokens, use_cache, precision="fp32"):
    """Fixed-length greedy decoding; timings exclude tokenization and file I/O."""
    model.eval()
    device = prompt.device
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("Use one nonempty prompt, shape [1, prompt_tokens]")
    if new_tokens < 1 or prompt.shape[1] + new_tokens > model.config.block_size:
        raise ValueError("Prompt plus new tokens must fit model block_size")
    sequence = prompt.clone()
    cache = None
    steps = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    for step in range(new_tokens):
        inputs = sequence[:, -1:] if use_cache and cache is not None else sequence
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == "bf16"):
            logits, _, cache = model(inputs, past_kv=cache if use_cache else None,
                                     use_cache=use_cache)
        # The padded training vocabulary has extra rows without GPT-2 token IDs.
        next_token = logits[:, -1, :min(50257, model.config.vocab_size)].argmax(-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
        synchronize(device)
        elapsed = time.perf_counter() - t0
        steps.append({"step": step + 1, "phase": "prefill" if step == 0 else "decode",
                      "context_tokens": sequence.shape[1] - 1,
                      "forward_tokens": inputs.shape[1], "seconds": elapsed,
                      "cached_tokens": cache[0][0].shape[-2] if cache else 0})
    total = time.perf_counter() - started
    decode_seconds = sum(item["seconds"] for item in steps[1:])
    cache_bytes = sum(t.numel() * t.element_size() for pair in cache for t in pair) if cache else 0
    return {"use_cache": use_cache, "prompt_tokens": prompt.shape[1], "new_tokens": new_tokens,
            "seconds": total, "tokens_per_second": new_tokens / total,
            "ttft_ms": steps[0]["seconds"] * 1000,
            "decode_ms_per_token": decode_seconds * 1000 / (new_tokens - 1) if new_tokens > 1 else None,
            "decode_tokens_per_second": (new_tokens - 1) / decode_seconds if new_tokens > 1 else None,
            "forward_tokens": sum(item["forward_tokens"] for item in steps),
            "logical_cache_bytes": cache_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "token_ids": sequence[0].tolist(), "steps": steps}


def benchmark(model, prompt, new_tokens, repeats=3, warmup=1, precision="fp32"):
    if repeats < 1 or warmup < 0:
        raise ValueError("Require repeats >= 1 and warmup >= 0")
    for _ in range(warmup):
        for cache in (False, True):
            generate(model, prompt, new_tokens, cache, precision)
    runs = []
    for repeat in range(repeats):
        # Alternate order to reduce warmup/order effects.
        for cache in ((False, True) if repeat % 2 == 0 else (True, False)):
            result = generate(model, prompt, new_tokens, cache, precision)
            result["repeat"] = repeat + 1
            runs.append(result)
    modes = {}
    for cache, name in ((False, "no_cache"), (True, "kv_cache")):
        selected = [run for run in runs if run["use_cache"] == cache]
        modes[name] = {key: statistics.median(run[key] for run in selected)
                       if selected[0][key] is not None else None
                       for key in ("seconds", "tokens_per_second", "ttft_ms", "decode_ms_per_token",
                                   "decode_tokens_per_second", "forward_tokens", "logical_cache_bytes",
                                   "peak_allocated_bytes")}
    reference = runs[0]["token_ids"]
    return {"precision": precision, "repeats": repeats, "warmup_per_mode": warmup,
            "prompt_tokens": prompt.shape[1], "new_tokens": new_tokens,
            "output_tokens_equal": all(run["token_ids"] == reference for run in runs),
            "speedup_total": modes["no_cache"]["seconds"] / modes["kv_cache"]["seconds"],
            "speedup_decode": modes["no_cache"]["decode_ms_per_token"] / modes["kv_cache"]["decode_ms_per_token"] if new_tokens > 1 else None,
            "modes": modes, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(training.DATA_DIRS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    dataset = args.dataset or "wikitext"
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        paths = sorted((ROOT / "runs" / training.DATA_DIRS[dataset] / "checkpoints").glob("model_*.pt"))
        if not paths:
            parser.error("No checkpoint found; train first or supply --checkpoint")
        checkpoint_path = paths[-1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trained_dataset = checkpoint.get("training_config", {}).get("dataset")
    if args.dataset and trained_dataset and args.dataset != trained_dataset:
        parser.error("--dataset differs from checkpoint training dataset")
    dataset = trained_dataset or dataset
    device = torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    if precision == "bf16" and device.type != "cuda":
        parser.error("Use fp32 for the CPU benchmark")
    model = training.GPT(training.GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    encoder = tiktoken.get_encoding("gpt2")
    ids = encoder.encode_ordinary(args.prompt)
    if not ids or len(ids) + args.max_new_tokens > model.config.block_size:
        parser.error("Require a nonempty prompt fitting prompt+generation inside block_size")
    if args.max_new_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("Require positive max-new-tokens/repeats and nonnegative warmup")
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    result = benchmark(model, prompt, args.max_new_tokens, args.repeats, args.warmup, precision)
    result.update(dataset=dataset, checkpoint=str(checkpoint_path.resolve()), prompt=args.prompt,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  torch_version=str(torch.__version__))
    for run in result["runs"]:
        run["generated_text"] = encoder.decode(run["token_ids"])
    output = args.output_dir or checkpoint_path.resolve().parent.parent / "prediction"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = []
    with (output / "steps.jsonl").open("w", encoding="utf-8") as file:
        for run in result["runs"]:
            for step in run["steps"]:
                row = {"repeat": run["repeat"], "use_cache": run["use_cache"], **step}
                file.write(json.dumps(row) + "\n")
                lines.append(f"cache={run['use_cache']} run={run['repeat']} step={step['step']:3d} "
                             f"context={step['context_tokens']:4d} input={step['forward_tokens']:4d} "
                             f"dt={step['seconds']*1000:.2f}ms cached={step['cached_tokens']}")
    for name, metrics in result["modes"].items():
        lines.append(f"{name}: {metrics['tokens_per_second']:.2f} tok/s, "
                     f"TTFT={metrics['ttft_ms']:.2f}ms, decode_ms/token={metrics['decode_ms_per_token']}, "
                     f"peak_allocated={metrics['peak_allocated_bytes']}, cache_bytes={metrics['logical_cache_bytes']}")
    lines.append(f"speedup_total={result['speedup_total']:.3f}x; speedup_decode={result['speedup_decode']}; "
                 f"output_tokens_equal={result['output_tokens_equal']}")
    for run in result["runs"][:2]:
        lines.append(f"\ncache={run['use_cache']} sample:\n{run['generated_text']}")
    (output / "prediction_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved prediction logs to {output}")


if __name__ == "__main__":
    main()
