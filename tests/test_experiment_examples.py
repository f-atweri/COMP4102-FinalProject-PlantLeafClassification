import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment import select_changed_examples, select_original_examples


class ExampleSelectionTests(unittest.TestCase):
    def test_original_examples_cover_classes_and_prefer_different_leaves(self):
        rows = [
            {"label": "a", "image_path": "a1", "leaf_id": "a-leaf"},
            {"label": "a", "image_path": "a2", "leaf_id": "a-leaf"},
            {"label": "a", "image_path": "a3", "leaf_id": "another-a-leaf"},
            {"label": "b", "image_path": "b1", "leaf_id": "b-leaf"},
            {"label": "b", "image_path": "b2", "leaf_id": "b-leaf"},
        ]
        self.assertEqual(select_original_examples(rows, ["a", "b"], [0] * 5, True), [0, 2, 1])
        self.assertEqual(select_original_examples(rows, ["a", "b"], [0] * 5, False), [3, 4])
        self.assertEqual(select_original_examples(rows, ["a", "b"], [0, 0, 0, 1, 1], True),
                         [0, 3, 2, 4])

    def test_changed_examples_include_both_directions_and_distinct_leaves(self):
        rows = [
            {"label": "a", "image_path": "a1", "leaf_id": "a-leaf"},
            {"label": "a", "image_path": "a2", "leaf_id": "a-leaf"},
            {"label": "b", "image_path": "b1", "leaf_id": "b-leaf"},
            {"label": "c", "image_path": "c1", "leaf_id": "c-leaf"},
        ]
        conditions = [("Original", 1.0), ("Brightness 0.5", 0.5), ("Brightness 1.5", 1.5)]
        results = {
            "Original": {"predictions": [0, 0, 0, 0]},
            "Brightness 0.5": {"predictions": [1, 1, 1, 1]},
            "Brightness 1.5": {"predictions": [1, 1, 1, 2]},
        }
        selected = select_changed_examples(rows, ["a", "b", "c"], conditions, results)
        self.assertEqual(selected[:3], [(0, "Brightness 0.5", 0.5),
                                       (2, "Brightness 1.5", 1.5),
                                       (3, "Brightness 0.5", 0.5)])
        self.assertEqual(len(selected), 4)
        self.assertEqual(select_changed_examples(rows, ["a", "b", "c"],
                                                 [conditions[0]], results), [])

    def test_missing_leaf_id_uses_image_path(self):
        rows = [{"label": "a", "image_path": "first"},
                {"label": "a", "image_path": "second"}]
        self.assertEqual(select_original_examples(rows, ["a"], [0, 0], True), [0, 1])


if __name__ == "__main__":
    unittest.main()
