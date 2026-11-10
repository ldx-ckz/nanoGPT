import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import wikitext

spec = importlib.util.spec_from_file_location("gpt_wikitext_tests", PROJECT_DIR / "GPT-2.py")
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)


class WikiTextTests(unittest.TestCase):
    def test_article_reconstruction_keeps_sections_and_whitespace(self):
        rows = [{"text": value} for value in (
            "", " = First Article = \n", "Body\n", "\n", " = = Section = = \n",
            "More text\n", " = Second Article = \n", "End\n")]
        articles = list(wikitext.iter_articles(rows))
        self.assertEqual(len(articles), 2)
        self.assertIn("Body\n\n = = Section = = \nMore text\n", articles[0])
        self.assertTrue(articles[1].startswith(" = Second Article = \n"))
        tokens = wikitext.tokenize(articles[0])
        self.assertEqual(tokens.dtype, np.uint16)
        self.assertEqual(tokens[0], wikitext.eot)
        self.assertEqual(wikitext.enc.decode(tokens[1:].tolist()), articles[0])

    def test_large_document_spans_shards_without_loss_or_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [np.arange(25, dtype=np.uint16), np.array([42, 43], dtype=np.uint16)]
            info = wikitext.write_split(train, root, "train", 8)
            wikitext.write_split([np.array([99], dtype=np.uint16)], root, "val", 8)
            actual = np.concatenate([np.load(root / name) for name in info["files"]])
            np.testing.assert_array_equal(actual, np.concatenate(train))
            self.assertEqual(info["tokens"], 27)
            self.assertEqual(len(info["files"]), 4)
            self.assertEqual(np.load(root / "wikitext_val_000000.npy").tolist(), [99])

    def test_loader_handles_tiny_tail_and_two_rank_epoch_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "wikitext_train_000000.npy", np.arange(13, dtype=np.uint16))
            np.save(root / "wikitext_train_000001.npy", np.array([13], dtype=np.uint16))
            loaders = [training.DataLoaderLite(1, 2, rank, 2, "train", root) for rank in range(2)]
            for step in range(20):
                for rank, loader in enumerate(loaders):
                    x, y = loader.next_batch()
                    expected = (torch.arange(3) + step * 4 + rank * 2) % 14
                    torch.testing.assert_close(x.flatten(), expected[:-1])
                    torch.testing.assert_close(y.flatten(), expected[1:])
                self.assertEqual(loaders[0].position, loaders[1].position)
            loaders[0].reset()
            self.assertEqual(loaders[0].next_batch()[0].tolist(), [[0, 1]])

    def test_defaults_use_only_wikitext_and_bound_token_budget(self):
        with patch.object(sys, "argv", ["GPT-2.py"]):
            wiki = training.parse_args()
        self.assertEqual((wiki.max_steps, wiki.total_batch_size), (4578, 65536))
        self.assertEqual(wiki.data_dir, "wikitext103")
        self.assertEqual(wiki.output_dir, "runs/wikitext103")
        self.assertFalse(hasattr(wiki, "hellaswag"))
        self.assertFalse(hasattr(training, "iterate_examples"))
        self.assertLess(wiki.warmup_steps, wiki.max_steps)

    def test_main_prepares_all_official_splits_and_records_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = {split: [{"text": f" = {split} = \n"}, {"text": "Some text.\n"}]
                    for split in ("train", "validation", "test")}
            argv = ["wikitext.py", "--output-dir", directory, "--workers", "1", "--shard-size", "8"]
            with patch.object(sys, "argv", argv), patch.object(wikitext, "HfApi") as api, \
                    patch.object(wikitext, "load_dataset", return_value=rows) as load:
                api.return_value.dataset_info.return_value.sha = "fixed_revision"
                wikitext.main()
            load.assert_called_once_with("Salesforce/wikitext", "wikitext-103-raw-v1", revision="fixed_revision")
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(set(manifest["splits"]), {"train", "val", "test"})
            self.assertEqual(manifest["splits"]["val"]["source_split"], "validation")

    def test_two_optimizer_steps_use_no_test_data_and_save_final_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                np.save(root / f"wikitext_{split}_000000.npy", np.arange(129, dtype=np.uint16))
            # A corrupt test file must never be opened by the training loop.
            (root / "wikitext_test_000000.npy").write_text("not training data")
            (root / "manifest.json").write_text("{}")
            original_gpt = training.GPT
            models = []

            def small_model(config):
                model = original_gpt(training.GPTConfig(block_size=8, vocab_size=50304,
                                                       n_layers=2, n_head=2, n_embed=32))
                models.append(model)
                return model

            argv = ["GPT-2.py", "--data-dir", str(root),
                    "--output-dir", str(root / "run"), "--max-steps", "2", "--warmup-steps", "0",
                    "--batch-size", "1", "--seq-len", "8", "--total-batch-size", "16", "--val-batches", "1"]
            with patch.object(sys, "argv", argv), patch.object(training, "GPT", side_effect=small_model), \
                    contextlib.redirect_stdout(io.StringIO()):
                training.main()
            checkpoint = torch.load(root / "run/checkpoints/model_00002.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["completed_steps"], 2)
            self.assertEqual(checkpoint["trained_tokens"], 32)
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(checkpoint["model"][key], value.cpu())
            lines = (root / "run/log/log.txt").read_text().splitlines()
            self.assertEqual(sum(" train " in line for line in lines), 2)


if __name__ == "__main__":
    unittest.main()
