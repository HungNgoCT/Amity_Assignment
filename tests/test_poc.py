import math
import random
import unittest
from collections import Counter

from src.baselines import answer_code_map, random_preds
from src.data.build_jsonl import person_split
from src.evaluate import score
from src.leak_test import assert_prompts_leak_free


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"pid": 1, "col": "q", "target": "1", "qtype": "MC"},
            {"pid": 2, "col": "q", "target": "1", "qtype": "MC"},
        ]
        self.ranges = {"q": (1.0, 2.0)}
        self.codes = {"q": ["1", "2"]}

    def test_invalid_outputs_remain_in_denominator(self):
        result = score(
            self.examples,
            {(1, "q"): "1", (2, "q"): "not-a-code"},
            self.ranges,
            self.codes,
        )
        self.assertEqual(result["slice_mean_mad"], 0.5)
        self.assertEqual(result["parse_rate"], 0.5)
        self.assertEqual(result["n_people"], 2)

    def test_nan_and_illegal_codes_score_zero(self):
        for output in ("nan", "inf", "999"):
            with self.subTest(output=output):
                result = score(
                    self.examples,
                    {(1, "q"): output, (2, "q"): "1"},
                    self.ranges,
                    self.codes,
                )
                self.assertTrue(math.isfinite(result["slice_mean_mad"]))
                self.assertEqual(result["slice_mean_mad"], 0.5)
                self.assertEqual(result["parse_rate"], 0.5)


class BaselineTests(unittest.TestCase):
    def test_random_uses_train_codes_uniformly(self):
        train = [
            {"col": "q", "target": code}
            for code in ("1", "2", "3")
        ]
        examples = [{"pid": pid, "col": "q"} for pid in range(30_000)]
        codes = answer_code_map(train)
        predictions = random_preds(examples, codes, random.Random(42))
        counts = Counter(predictions.values())
        self.assertEqual(set(counts), {"1", "2", "3"})
        for count in counts.values():
            self.assertGreater(count, 9_500)
            self.assertLess(count, 10_500)


class LeakageTests(unittest.TestCase):
    def test_known_numbers_in_unrelated_fields_are_safe(self):
        row = {
            "pid": 1,
            "col": "QID154",
            "prompt": (
                "Persona:\nage: 82\nprior_count: 70\n\n"
                "Question:\nHow many lawyers?\nOptions: []"
            ),
        }
        self.assertEqual(assert_prompts_leak_free([row]), 1)

    def test_repeated_target_column_is_rejected(self):
        row = {
            "pid": 1,
            "col": "QID154",
            "prompt": "Persona:\nQID154: 70\n\nQuestion:\nHow many lawyers?",
        }
        with self.assertRaises(AssertionError):
            assert_prompts_leak_free([row])

    def test_answer_bearing_keys_are_rejected_on_any_row(self):
        row = {
            "pid": 99,
            "col": "QID1",
            "prompt": "Persona:\nage: 40\n\nQuestion:\nValues: [1]",
        }
        with self.assertRaises(AssertionError):
            assert_prompts_leak_free([row], require_hit=False)


class SplitTests(unittest.TestCase):
    def test_person_split_is_complete_and_disjoint(self):
        split = person_split(list(range(2_058)))
        train, val, test = (set(split[name]) for name in ("train", "val", "test"))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(len(train | val | test), 2_058)


if __name__ == "__main__":
    unittest.main()
