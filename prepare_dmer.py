"""Audit DMER archives and prepare selected subjects for the HeLo loader.

Explicit baseline: normalize each trial's ten raw PANAS scores by their sum.
This is a documented reproduction assumption, not verified author preprocessing.
Raw and aligned video archives are never extracted in full.
"""

import argparse
import bisect
import csv
import hashlib
import io
import json
import os
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import loadmat, savemat


class JoinedParts(io.RawIOBase):
    """Seek across byte-split ZIP parts without copying the large archive."""

    def __init__(self, paths, stack):
        self.files = [stack.enter_context(path.open("rb")) for path in paths]
        self.offsets = [0]
        for path in paths:
            self.offsets.append(self.offsets[-1] + path.stat().st_size)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        origins = (0, self.position, self.offsets[-1])
        position = origins[whence] + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        available = max(0, self.offsets[-1] - self.position)
        size = available if size is None or size < 0 else min(size, available)
        result = []
        while size:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            handle = self.files[index]
            handle.seek(self.position - self.offsets[index])
            chunk = handle.read(min(size, self.offsets[index + 1] - self.position))
            if not chunk:
                raise EOFError("Unexpected end of archive part")
            result.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(result)


def archive_parts(root, stem):
    whole = root / stem
    if whole.is_file():
        return [whole]
    parts = sorted(path for path in root.glob(stem + ".*") if path.suffix[1:].isdigit())
    if not parts or [int(path.suffix[1:]) for path in parts] != list(range(1, len(parts) + 1)):
        raise ValueError("Missing or nonconsecutive archive parts: " + stem)
    return parts


def member_map(archive, kind):
    result = {}
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        path = PurePosixPath(entry.filename.replace("\\", "/"))
        if kind == "features":
            if path.suffix.lower() != ".mat" or not path.stem.isdigit():
                continue
            subject = int(path.stem)
        else:
            if path.name.lower() != "panas.csv":
                continue
            if not path.parent.name.isdigit():
                raise ValueError("Cannot identify PANAS subject: " + entry.filename)
            subject = int(path.parent.name)
        if subject in result:
            raise ValueError("Duplicate %s member for subject %d" % (kind, subject))
        result[subject] = entry.filename
    return result


def parse_panas(payload, allow_out_of_range=False):
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    table = np.asarray(rows, dtype=np.float64)
    if table.shape != (32, 11) or not np.isfinite(table).all():
        raise ValueError("PANAS must have 32 finite rows and 11 columns, without a header")
    ids = table[:, 0]
    if not np.equal(ids, np.floor(ids)).all() or set(ids.tolist()) != set(range(32)):
        raise ValueError("PANAS video IDs must contain each ID 0..31 exactly once")
    scores = table[:, 1:]
    if not np.equal(scores, np.floor(scores)).all() or np.any(scores < 1) or (np.any(scores > 9) and not allow_out_of_range):
        raise ValueError("PANAS scores must be integer ratings between 1 and 9; observed range %g..%g" % (scores.min(), scores.max()))
    # The observed archive's actual range is reported, rather than assumed to be 1..5.
    ordered = np.empty((32, 10), dtype=np.float64)
    ordered[ids.astype(np.int64)] = scores
    distribution = ordered / ordered.sum(axis=1, keepdims=True)
    return ordered, distribution, not np.array_equal(ids, np.arange(32))


def validate_features(data):
    if "feas" not in data or "vids" not in data:
        raise ValueError("Missing feas or vids")
    if "dis_label" in data:
        raise ValueError("Source already has dis_label; compare its meaning before replacing it")
    features = np.asarray(data["feas"])
    ids = np.asarray(data["vids"])
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 913:
        raise ValueError("Expected nonempty N x 913 features")
    if ids.shape != (len(features), 1):
        raise ValueError("Expected N x 1 video IDs")
    if features.dtype.kind not in "fiu" or ids.dtype.kind not in "fiu":
        raise ValueError("Features and IDs must be real numeric arrays")
    if not np.isfinite(features).all() or not np.isfinite(ids).all():
        raise ValueError("Features or IDs contain NaN/Inf")
    if not np.equal(ids, np.floor(ids)).all() or np.any(ids < 0) or np.any(ids > 31):
        raise ValueError("Feature video IDs must be integers in 0..31")
    return features, ids


def read_subject(feature_zip, raw_zip, feature_name, panas_name, allow_out_of_range=False):
    feature_bytes = feature_zip.read(feature_name)
    panas_bytes = raw_zip.read(panas_name)
    features, ids = validate_features(loadmat(io.BytesIO(feature_bytes)))
    counts = np.bincount(ids[:, 0].astype(np.int64), minlength=32)
    metadata = {
        "feature_member": feature_name,
        "panas_member": panas_name,
        "feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "panas_sha256": hashlib.sha256(panas_bytes).hexdigest(),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "sample_count_per_video": counts.tolist(),
        "feature_video_count": int(np.count_nonzero(counts)),
    }
    try:
        scores, distribution, reordered = parse_panas(panas_bytes, allow_out_of_range)
    except ValueError as error:
        metadata.update(status="invalid_labels", error=str(error))
        return None, metadata
    metadata.update(status="valid", rating_min=float(scores.min()), rating_max=float(scores.max()),
                    panas_reordered_by_video_id=reordered)
    metadata["ratings_above_9"] = [{"video_id": int(i), "emotion_column_1based": int(j + 1), "score": float(scores[i, j])}
                                   for i, j in np.argwhere(scores > 9)]
    return {"feas": features, "vids": ids, "dis_label": distribution, "panas_scores": scores}, metadata


def write_subject(target, prepared):
    if target.exists():
        raise FileExistsError("Will not overwrite existing prepared data: " + str(target))
    temporary = target.with_suffix(".mat.tmp")
    with temporary.open("xb") as handle:
        savemat(handle, prepared, do_compression=True)
    loaded = loadmat(temporary)
    for key, expected in prepared.items():
        if not np.array_equal(loaded[key], expected):
            raise ValueError("Saved MAT round-trip mismatch: " + key)
    if target.exists():
        raise FileExistsError("Destination appeared while saving: " + str(target))
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "Features")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--allow-out-of-range", action="store_true", help="Explicit sensitivity policy: preserve positive integer ratings above 9, record them, and do not clip.")
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    output = args.output.resolve()
    selected = sorted(set(args.subjects))
    if output == root or root in output.parents:
        raise ValueError("Prepared output must be outside the original dataset directory")
    feature_path = root / "Features.zip"
    raw_paths = archive_parts(root, "Raw_data.zip")
    report_path = output / "preparation_report.json"
    if report_path.exists() or any((output / (str(subject) + ".mat")).exists() for subject in selected):
        raise FileExistsError("Output already exists; use a fresh output directory to preserve provenance")

    with ExitStack() as stack:
        feature_zip = stack.enter_context(zipfile.ZipFile(feature_path))
        raw_zip = stack.enter_context(zipfile.ZipFile(JoinedParts(raw_paths, stack)))
        features = member_map(feature_zip, "features")
        panas = member_map(raw_zip, "panas")
        if not features or set(features) != set(panas):
            raise ValueError("Feature subjects and PANAS subjects differ")
        if not set(selected).issubset(features):
            raise ValueError("Requested subject is absent from the dataset")
        print("Auditing %d subjects; preparing %s" % (len(features), selected), flush=True)
        audit = {}
        for number, subject in enumerate(sorted(features), 1):
            _, metadata = read_subject(feature_zip, raw_zip, features[subject], panas[subject], args.allow_out_of_range)
            audit[str(subject)] = metadata
            if number % 10 == 0 or number == len(features):
                print("Audited %d/%d" % (number, len(features)), flush=True)
        invalid = [int(subject) for subject, item in audit.items() if item["status"] != "valid"]
        if invalid:
            print("Subjects with invalid labels (not repaired):", invalid, flush=True)
        if set(selected).intersection(invalid):
            raise ValueError("A requested subject has invalid labels: " + str({s: audit[str(s)]["error"] for s in selected if s in invalid}))
        valid_audit = [item for item in audit.values() if item["status"] == "valid"]
        # Audit everyone first; never prepare a subject whose labels failed validation.
        output.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for subject in selected:
            prepared, _ = read_subject(feature_zip, raw_zip, features[subject], panas[subject], args.allow_out_of_range)
            target = output / (str(subject) + ".mat")
            write_subject(target, prepared)
            outputs[str(subject)] = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            print("Prepared subject %d: %d samples" % (subject, len(prepared["feas"])), flush=True)
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(root),
            "archives": [{"path": str(path), "size_bytes": path.stat().st_size} for path in [feature_path] + raw_paths],
            "label_rule": "raw PANAS score / sum of ten raw PANAS scores for the same subject and video",
            "label_rule_status": "explicit reproduction assumption; not verified HeLo author preprocessing",
            "label_order": "Panas.csv columns 2..11, preserved; emotion names not inferred",
            "baseline_subtraction": False,
            "allow_out_of_range": args.allow_out_of_range,
            "out_of_range_policy": "preserve original positive integer scores and flag; never clip" if args.allow_out_of_range else "reject scores above 9",
            "feature_transformation": "none; original 913 columns and dtype preserved",
            "subject_count": len(audit),
            "total_feature_samples": sum(item["feature_shape"][0] for item in audit.values()),
            "invalid_label_subjects": invalid,
            "valid_label_subject_count": len(valid_audit),
            "valid_subject_rating_range": [min(item["rating_min"] for item in valid_audit), max(item["rating_max"] for item in valid_audit)],
            "audit": audit,
            "prepared_outputs": outputs,
        }
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("Rating range (valid subjects only):", report["valid_subject_rating_range"])
        print("Total feature samples:", report["total_feature_samples"])
        print("Report:", report_path)
        print("DATA PREPARATION PASSED; label transformation is a documented assumption.")


if __name__ == "__main__":
    main()
