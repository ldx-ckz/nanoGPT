import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import predict
import tinystories

training = predict.training


class KVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.model = training.GPT(training.GPTConfig(block_size=24, vocab_size=128,
            n_layers=2, n_head=2, n_embed=32)).eval()

    def test_prefill_single_token_and_chunk_decode_match_full_forward(self):
        tokens = torch.randint(128, (3, 16))
        with torch.inference_mode():
            full, _, empty = self.model(tokens, past_kv=None, use_cache=False)
            self.assertIsNone(empty)
            for lengths in ([4] + [1] * 12, [3, 5, 8]):
                cache = None
                offset = 0
                parts = []
                for length in lengths:
                    out, _, cache = self.model(tokens[:, offset:offset+length], past_kv=cache, use_cache=True)
                    offset += length
                    self.assertTrue(all(k.shape == (3, 2, offset, 16) and v.shape == k.shape for k, v in cache))
                    parts.append(out)
                torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-6, rtol=1e-5)
            changed = tokens.clone()
            changed[:, 8:] = (changed[:, 8:] + 1) % 128
            out, _, _ = self.model(changed)
            torch.testing.assert_close(full[:, :8], out[:, :8])

    def test_generation_benchmark_tracks_work_and_equal_outputs(self):
        prompt = torch.tensor([[1, 2, 3]])
        result = predict.benchmark(self.model, prompt, 6, repeats=2, warmup=0)
        self.assertTrue(result["output_tokens_equal"])
        self.assertEqual(result["modes"]["no_cache"]["forward_tokens"], sum(range(3, 9)))
        self.assertEqual(result["modes"]["kv_cache"]["forward_tokens"], 8)
        self.assertEqual(result["modes"]["kv_cache"]["logical_cache_bytes"], 2*2*1*2*8*16*4)
        self.assertGreater(result["speedup_total"], 0)
        with self.assertRaises(ValueError):
            predict.generate(self.model, prompt, 24, True)

    def test_prediction_cli_loads_checkpoint_and_writes_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save({"config": training.asdict(self.model.config), "model": self.model.state_dict(),
                        "training_config": {"dataset": "tinystory"}}, checkpoint)
            # Small test vocabulary: supply byte-range token IDs through a stub encoder.
            class Encoder:
                def encode_ordinary(self, text):
                    return [1, 2, 3]

                def decode(self, tokens):
                    return str(tokens)

            argv = ["predict.py", "--dataset", "tinystory", "--checkpoint", str(checkpoint),
                    "--max-new-tokens", "4", "--repeats", "1", "--warmup", "0",
                    "--device", "cpu", "--output-dir", str(root / "output")]
            with patch.object(sys, "argv", argv), patch.object(predict.tiktoken, "get_encoding", return_value=Encoder()), \
                    contextlib.redirect_stdout(io.StringIO()):
                predict.main()
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertEqual(summary["dataset"], "tinystory")
            self.assertTrue(summary["output_tokens_equal"])
            self.assertEqual(len((root / "output/steps.jsonl").read_text().splitlines()), 8)


class TinyStoriesTests(unittest.TestCase):
    def test_split_is_stable_and_duplicates_cannot_leak(self):
        rows = [{"text": f"Story {i}"} for i in range(20)]
        rows += [{"text": "Story 0"}, {"text": " Story 0 "}]
        splits, hashes = tinystories.split_validation(rows)
        again, _ = tinystories.split_validation(reversed(rows))
        val_hashes = {tinystories.text_hash(x) for x in splits["val"]}
        test_hashes = {tinystories.text_hash(x) for x in splits["test"]}
        self.assertFalse(val_hashes & test_hashes)
        self.assertEqual(val_hashes, {tinystories.text_hash(x) for x in again["val"]})
        self.assertEqual(list(tinystories.training_texts([{"text": "Story 0"}, {"text": "New"}], hashes)), ["New"])
        arrays = [np.arange(3), np.arange(4), np.arange(5)]
        self.assertEqual(sum(len(x) for x in tinystories.limit_tokens(arrays, 8)), 7)

    def test_download_pipeline_and_dataset_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = {"train": [{"text": f"Train story {i}"} for i in range(4)],
                    "validation": [{"text": f"Held out {i}"} for i in range(30)]}
            argv = ["tinystories.py", "--workers", "1", "--output-dir", directory, "--train-tokens", "0"]
            with patch.object(sys, "argv", argv), patch.object(tinystories, "load_dataset", return_value=rows), \
                    patch.object(tinystories, "HfApi") as api:
                api.return_value.dataset_info.return_value.sha = "fixed_revision"
                tinystories.main()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(set(manifest["splits"]), {"train", "val", "test"})
            self.assertEqual(manifest["splits"]["test"]["source_split"], "validation")
            self.assertTrue(list(Path(directory).glob("tinystories_train_*.npy")))
        with patch.object(sys, "argv", ["GPT-2.py", "--dataset", "tinystory"]):
            args = training.parse_args()
        self.assertEqual((args.data_dir, args.output_dir), ("tinystories", "runs/tinystories"))


if __name__ == "__main__":
    unittest.main()
