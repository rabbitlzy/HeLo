import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from verify_resume import compare_runs


class ResumeComparisonTests(unittest.TestCase):
    def write_run(self, folder):
        folder.mkdir()
        files = {
            "manifest.json": {"config": {"epochs": 3, "subjects": [1]}},
            "summary.json": {"status": "complete", "completed_subjects": [1], "planned_subjects": [1]},
            "subject_1.json": {"subject": 1, "samples": 3, "train": 2, "test": 1, "epochs": 3,
                               "metrics": [.1, .2, .3, .4, .8, .9]},
        }
        for name, content in files.items():
            (folder / name).write_text(json.dumps(content), encoding="utf-8")
        np.savez(folder / "split_1.npz", train=[0, 1], test=[2], vids=[[0], [1], [2]])
        np.savez(folder / "predictions_1.npz", predictions=[[.4, .6]], labels=[[.5, .5]], indices=[2], video_ids=[[2]])
        torch.save({"weight": torch.tensor([.2, .5])}, folder / "model_1.pt")
        history = [{"epoch": epoch, "lr_next": .001 * .99**epoch} for epoch in range(1, 4)]
        (folder / "history_1.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")

    def test_identical_runs_pass_with_zero_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, resumed = Path(temporary) / "reference", Path(temporary) / "resumed"
            self.write_run(reference)
            self.write_run(resumed)
            report = compare_runs(reference, resumed)
            self.assertTrue(report["all_compared_arrays_exact"])
            self.assertEqual(report["prediction_max_abs_difference"], 0)
            self.assertEqual(report["weight_max_abs_difference"], 0)

    def test_changed_split_weights_nonfinite_predictions_and_schedule_are_rejected(self):
        cases = {"split": "Split differs", "weight": "Weight weight", "nan": "non-finite", "schedule": "schedule differs"}
        for case, message in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                reference, resumed = Path(temporary) / "reference", Path(temporary) / "resumed"
                self.write_run(reference)
                self.write_run(resumed)
                if case == "split":
                    np.savez(resumed / "split_1.npz", train=[1, 0], test=[2], vids=[[0], [1], [2]])
                elif case == "weight":
                    torch.save({"weight": torch.tensor([.3, .5])}, resumed / "model_1.pt")
                elif case == "nan":
                    np.savez(resumed / "predictions_1.npz", predictions=[[np.nan, .6]], labels=[[.5, .5]], indices=[2], video_ids=[[2]])
                else:
                    history = [{"epoch": epoch, "lr_next": .001} for epoch in range(1, 4)]
                    (resumed / "history_1.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    compare_runs(reference, resumed)


if __name__ == "__main__":
    unittest.main()
