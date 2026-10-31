import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hellaswag


class HellaSwagDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "hellaswag_val.jsonl"
        self.patch_cache = patch.object(hellaswag, "DATA_CACHE_DIR", self.directory.name)
        self.patch_cache.start()
        self.addCleanup(self.patch_cache.stop)
        self.example = {"ctx": "A person opens a book.", "endings": ["a", "b", "c", "d"], "label": "2"}

    def test_404_cache_is_replaced_with_valid_examples(self):
        self.cache.write_text("404: Not Found", encoding="utf-8")
        with patch.object(hellaswag, "load_dataset", return_value=[self.example]) as load:
            examples = list(hellaswag.iterate_examples("val"))
        self.assertEqual(examples[0]["label"], 2)
        self.assertEqual(json.loads(self.cache.read_text(encoding="utf-8"))["ctx"], self.example["ctx"])
        self.assertIn("validation-00000-of-00001.parquet", load.call_args.kwargs["data_files"]["val"])

    def test_valid_cache_works_without_network(self):
        self.cache.write_text(json.dumps(self.example) + "\n", encoding="utf-8")
        with patch.object(hellaswag, "load_dataset", side_effect=AssertionError("Unexpected network access")):
            self.assertEqual(list(hellaswag.iterate_examples("val"))[0]["label"], 2)

    def test_interrupted_download_does_not_publish_partial_cache(self):
        def interrupted_rows():
            yield self.example
            raise ConnectionError("Download interrupted")

        with patch.object(hellaswag, "load_dataset", return_value=interrupted_rows()):
            with self.assertRaisesRegex(ConnectionError, "interrupted"):
                hellaswag.download("val")
        self.assertFalse(self.cache.exists())
        self.assertEqual(list(Path(self.directory.name).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
