import io
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from prepare_dmer import JoinedParts, archive_parts, parse_panas, read_subject, validate_features, write_subject
from scipy.io import loadmat, savemat


class PreparationTests(unittest.TestCase):
    def panas(self):
        scores = np.array([[(video + column) % 5 + 1 for column in range(10)] for video in range(32)], dtype=float)
        rows = [str(video) + "," + ",".join(str(int(value)) for value in scores[video]) for video in reversed(range(32))]
        return ("\n".join(rows)).encode(), scores

    def test_shuffled_ids_are_used_instead_of_row_numbers(self):
        payload, expected = self.panas()
        ordered, distribution, reordered = parse_panas(payload)
        self.assertTrue(reordered)
        np.testing.assert_array_equal(ordered, expected)
        np.testing.assert_allclose(distribution.sum(axis=1), 1)
        np.testing.assert_allclose(distribution[0], expected[0] / expected[0].sum())

    def test_duplicate_video_id_is_rejected(self):
        payload, _ = self.panas()
        rows = payload.decode().splitlines()
        rows[-1] = "1," + rows[-1].split(",", 1)[1]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            parse_panas("\n".join(rows).encode())

    def test_fractional_feature_id_and_nonfinite_feature_are_rejected(self):
        features = np.ones((2, 913))
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_features({"feas": features, "vids": np.array([[0.5], [1]])})
        features[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_features({"feas": features, "vids": np.array([[0], [1]])})

    def test_out_of_range_score_is_reported_without_preparing_subject(self):
        payload, _ = self.panas()
        rows = payload.decode().splitlines()
        cells = rows[0].split(",")
        cells[1] = "13"
        rows[0] = ",".join(cells)
        bad = "\n".join(rows).encode()
        mat = io.BytesIO()
        savemat(mat, {"feas": np.ones((2, 913)), "vids": np.array([[0], [1]])})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("1.mat", mat.getvalue())
            archive.writestr("1/Panas.csv", bad)
        with zipfile.ZipFile(buffer) as archive:
            prepared, metadata = read_subject(archive, archive, "1.mat", "1/Panas.csv")
        self.assertIsNone(prepared)
        self.assertEqual(metadata["status"], "invalid_labels")
        self.assertIn("13", metadata["error"])
        self.assertEqual(metadata["feature_shape"], [2, 913])
        scores, distribution, _ = parse_panas(bad, allow_out_of_range=True)
        self.assertEqual(scores[31, 0], 13)
        self.assertAlmostEqual(distribution[31, 0], 13 / scores[31].sum())
        with zipfile.ZipFile(buffer) as archive:
            prepared, metadata = read_subject(archive, archive, "1.mat", "1/Panas.csv", allow_out_of_range=True)
        self.assertIsNotNone(prepared)
        self.assertEqual(metadata["ratings_above_9"], [{"video_id": 31, "emotion_column_1based": 1, "score": 13.0}])

    def test_split_archive_reads_across_boundaries_and_checks_crc(self):
        buffer = io.BytesIO()
        payload = bytes(range(256)) * 3
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("1/1/Panas.csv", payload)
        blob = buffer.getvalue()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = [0, 37, 111, len(blob)]
            for index in range(3):
                (root / ("Raw_data.zip.%03d" % (index + 1))).write_bytes(blob[boundaries[index]:boundaries[index + 1]])
            with ExitStack() as stack:
                joined = JoinedParts(archive_parts(root, "Raw_data.zip"), stack)
                with zipfile.ZipFile(joined) as archive:
                    self.assertEqual(archive.read("1/1/Panas.csv"), payload)

    def test_saved_mat_roundtrip_and_overwrite_protection(self):
        payload, _ = self.panas()
        scores, distribution, _ = parse_panas(payload)
        prepared = {"feas": np.ones((2, 913)), "vids": np.array([[31], [0]], dtype=np.uint8), "dis_label": distribution, "panas_scores": scores}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "1.mat"
            write_subject(target, prepared)
            loaded = loadmat(target)
            for key in prepared:
                np.testing.assert_array_equal(loaded[key], prepared[key])
            with self.assertRaises(FileExistsError):
                write_subject(target, prepared)


if __name__ == "__main__":
    unittest.main()
