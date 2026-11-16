import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leaf_baseline as baseline


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        for label, channel in (("healthy", 1), ("diseased", 0)):
            folder = self.data / label
            folder.mkdir(parents=True)
            for index in range(8):
                color = [0, 0, 0]
                color[channel] = 150 + index
                Image.new("RGB", (12, 12), tuple(color)).save(folder / f"{index}.png")

    def tearDown(self):
        self.temp.cleanup()

    def test_splits_are_deterministic_and_keep_duplicates_together(self):
        original = self.data / "healthy" / "0.png"
        (self.data / "healthy" / "copy.png").write_bytes(original.read_bytes())
        first = baseline.make_splits(self.data, ["healthy", "diseased"], 42)
        self.assertEqual(first, baseline.make_splits(self.data, ["healthy", "diseased"], 42))
        copies = [row for row in first if row["sha256"] == baseline.image_hash(original)]
        self.assertEqual(len(copies), 2)
        self.assertEqual(len({row["split"] for row in copies}), 1)
        self.assertEqual({row["split"] for row in first}, {"train", "validation", "test"})

    def test_cross_class_duplicate_is_rejected(self):
        source = self.data / "healthy" / "0.png"
        (self.data / "diseased" / "copy.png").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "different labels"):
            baseline.make_splits(self.data, ["healthy", "diseased"], 42)

    def test_metrics_and_cli_results(self):
        scores = baseline.metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(scores["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)
        self.assertAlmostEqual(scores["per_class"]["b"]["precision"], 0.5)
        output = self.root / "results.json"
        subprocess.run(
            [sys.executable, str(Path(baseline.__file__)), "--data", str(self.data),
             "--classes", "healthy", "diseased", "--size", "12", "--output", str(output)],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(set(result["validation"]), {"color", "color_edge"})
        self.assertEqual(set(result["test"]), {"original", "dark", "bright"})
        self.assertEqual(len(result["test"]["original"]["predictions"]), 2)
        self.assertEqual(len(result["split_records"]), 16)


if __name__ == "__main__":
    unittest.main()
