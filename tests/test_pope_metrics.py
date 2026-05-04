from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from src.eval.pope_metrics import get_metrics, print_metrics


class PopeMetricsTest(unittest.TestCase):
    def test_extended_binary_metrics(self):
        results = [
            {"ground_truth": "yes", "best_answer": "yes"},
            {"ground_truth": "no", "best_answer": "no"},
            {"ground_truth": "no", "best_answer": "yes"},
            {"ground_truth": "yes", "best_answer": "no"},
            {"ground_truth": "yes", "best_answer": "maybe"},
            {"ground_truth": "unknown", "best_answer": "yes"},
        ]

        metrics = get_metrics(results)

        self.assertEqual(metrics["N"], 5)
        self.assertEqual(metrics["TP"], 1)
        self.assertEqual(metrics["TN"], 1)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 2)
        self.assertEqual(metrics["Unknown Pred"], 1)
        self.assertEqual(metrics["Unknown GT"], 1)
        self.assertAlmostEqual(metrics["Accuracy"], 40.0)
        self.assertAlmostEqual(metrics["Precision"], 50.0)
        self.assertAlmostEqual(metrics["Recall"], 100.0 / 3.0)
        self.assertAlmostEqual(metrics["F1"], 40.0)
        self.assertAlmostEqual(metrics["Yes Ratio"], 40.0)
        self.assertAlmostEqual(metrics["FPR"], 50.0)
        self.assertAlmostEqual(metrics["TNR"], 50.0)
        self.assertAlmostEqual(metrics["Specificity"], 50.0)
        self.assertAlmostEqual(metrics["FNR"], 200.0 / 3.0)
        self.assertAlmostEqual(metrics["Balanced Accuracy"], 125.0 / 3.0)

    def test_print_metrics_includes_extended_fields(self):
        metrics = get_metrics(
            [
                {"ground_truth": "yes", "best_answer": "yes"},
                {"ground_truth": "no", "best_answer": "yes"},
            ]
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_metrics("TEST", metrics)
        output = buffer.getvalue()

        self.assertIn("Unknown Pred", output)
        self.assertIn("FPR", output)
        self.assertIn("FNR", output)
        self.assertIn("TNR / Specificity", output)
        self.assertIn("Balanced Accuracy", output)


if __name__ == "__main__":
    unittest.main()
