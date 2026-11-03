import io
import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path

import numpy as np

from prepare_dmer import DirectorySource, JoinedParts, archive_parts, main, parse_panas, read_subject, validate_features, write_subject
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

    def test_extracted_and_archive_inputs_produce_identical_arrays_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "extracted" / "features"
            raw = root / "extracted" / "raw"
            archives = root / "archives"
            features.mkdir(parents=True)
            archives.mkdir()
            original_bytes = {}
            expected_scores = {}
            with zipfile.ZipFile(archives / "Features.zip", "w") as feature_zip, \
                    zipfile.ZipFile(archives / "Raw_data.zip", "w") as raw_zip:
                for subject in (1, 36):
                    feature_file = features / (str(subject) + ".mat")
                    feature_array = np.arange(2 * 913, dtype=np.float64).reshape(2, 913) + subject
                    savemat(feature_file, {"feas": feature_array, "vids": np.array([[31], [0]], dtype=np.uint8)})
                    payload, scores = self.panas()
                    if subject == 36:
                        # The first CSV row is video 31, not video 0.
                        rows = payload.decode().splitlines()
                        cells = rows[0].split(",")
                        cells[1] = "13"
                        rows[0] = ",".join(cells)
                        payload = "\n".join(rows).encode()
                        scores[31, 0] = 13
                    expected_scores[subject] = scores
                    # Exercise both supported extracted layouts in one cohort.
                    member = "1/Panas.csv" if subject == 1 else "36/36/Panas.csv"
                    panas_file = raw / member
                    panas_file.parent.mkdir(parents=True)
                    panas_file.write_bytes(payload)
                    original_bytes[feature_file] = feature_file.read_bytes()
                    original_bytes[panas_file] = payload
                    feature_zip.writestr(feature_file.name, original_bytes[feature_file])
                    raw_zip.writestr(member, payload)

            directory_args = ["--features-dir", str(features), "--panas-dir", str(raw)]
            strict_output = root / "strict_output"
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "invalid labels"):
                main(directory_args + ["--output", str(strict_output), "--subjects", "1", "36"])
            self.assertFalse(strict_output.exists())

            reports = []
            outputs = []
            for name, source_args in (("directory", directory_args), ("archive", ["--source", str(archives)])):
                output = root / (name + "_output")
                outputs.append(output)
                args = source_args + ["--output", str(output), "--subjects", "1", "36", "--allow-out-of-range"]
                with redirect_stdout(io.StringIO()):
                    main(args)
                reports.append(json.loads((output / "preparation_report.json").read_text(encoding="utf-8")))
                for subject, scores in expected_scores.items():
                    actual = loadmat(output / (str(subject) + ".mat"))
                    original = loadmat(features / (str(subject) + ".mat"))
                    np.testing.assert_array_equal(actual["feas"], original["feas"])
                    np.testing.assert_array_equal(actual["vids"], original["vids"])
                    np.testing.assert_array_equal(actual["panas_scores"], scores)
                    np.testing.assert_allclose(actual["dis_label"], scores / scores.sum(axis=1, keepdims=True))
                with self.assertRaises(FileExistsError):
                    main(args)

            self.assertEqual(reports[0]["source_format"], "directories")
            self.assertEqual(reports[1]["source_format"], "archives")
            self.assertEqual(reports[0]["source_content_sha256"], reports[1]["source_content_sha256"])
            self.assertEqual(reports[0]["total_feature_samples"], 4)
            self.assertEqual(reports[0]["audit"]["36"]["rating_max"], 13)
            for subject in (1, 36):
                left, right = (loadmat(output / (str(subject) + ".mat")) for output in outputs)
                for field in ("feas", "vids", "panas_scores", "dis_label"):
                    np.testing.assert_array_equal(left[field], right[field])
            for path, expected in original_bytes.items():
                self.assertEqual(path.read_bytes(), expected)

    def test_duplicate_or_conflicting_extracted_questionnaires_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "1" / "1").mkdir(parents=True)
            (root / "1" / "Panas.csv").write_text("unused")
            (root / "1" / "1" / "Panas.csv").write_text("unused")
            with self.assertRaisesRegex(ValueError, "Duplicate panas"):
                DirectorySource(root).members("panas")
            (root / "2" / "3").mkdir(parents=True)
            (root / "2" / "3" / "Panas.csv").write_text("unused")
            with self.assertRaisesRegex(ValueError, "Conflicting PANAS"):
                DirectorySource(root).members("panas")

    def test_extracted_sources_cannot_be_used_as_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features, raw = root / "features", root / "raw"
            features.mkdir()
            raw.mkdir()
            for output in (features, raw / "generated"):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                    main(["--features-dir", str(features), "--panas-dir", str(raw), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
