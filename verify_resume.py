"""Verify disk checkpoint recovery against a completed subject-1 reference run.

Creates a separate run, exits immediately after the epoch-25 checkpoint, then
resumes in a fresh process. Training/model source files are never edited.
Only load checkpoints from your own local runs.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STOP_EPOCH = 25
STOP_EXIT_CODE = 75
ATOL = 1e-6
STOP_DRIVER = """
import run_reproduction as training
original_save = training.save_checkpoint
def save_then_stop(path, state):
    original_save(path, state)
    if path.name == 'resume_1.pt' and state.get('epoch') == 25:
        print('INTENTIONAL_CHECKPOINT_STOP: epoch=25', flush=True)
        raise SystemExit(75)
training.save_checkpoint = save_then_stop
training.main()
"""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_local_checkpoint(path):
    options = {"map_location": "cpu"}
    # PyTorch 1.12 lacks this explicit parameter; newer versions can restrict
    # loading to tensors and the primitive containers in our own checkpoints.
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)


def finite_max_difference(left, right, description):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.size > 0, description + ": shape mismatch or empty array")
    require(np.isfinite(left).all() and np.isfinite(right).all(), description + ": non-finite values")
    difference = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    require(difference <= ATOL, "%s: max difference %.9g exceeds %.9g" % (description, difference, ATOL))
    return difference


def compare_runs(reference, resumed):
    config = read_json(reference / "manifest.json")["config"]
    require(read_json(resumed / "manifest.json")["config"] == config, "Run configurations/source hashes differ")
    for folder in (reference, resumed):
        summary = read_json(folder / "summary.json")
        require(summary["status"] == "complete" and summary["completed_subjects"] == [1]
                and summary["planned_subjects"] == [1], "Expected a completed subject-1 run: " + str(folder))
    with np.load(reference / "split_1.npz", allow_pickle=False) as left, \
            np.load(resumed / "split_1.npz", allow_pickle=False) as right:
        for name in ("train", "test", "vids"):
            require(np.array_equal(left[name], right[name]), "Split differs: " + name)
    with np.load(reference / "predictions_1.npz", allow_pickle=False) as left, \
            np.load(resumed / "predictions_1.npz", allow_pickle=False) as right:
        for name in ("labels", "indices", "video_ids"):
            require(np.array_equal(left[name], right[name]), "Evaluation data differs: " + name)
        prediction_difference = finite_max_difference(left["predictions"], right["predictions"], "Predictions")
    left_result, right_result = (read_json(folder / "subject_1.json") for folder in (reference, resumed))
    for name in ("subject", "samples", "train", "test", "epochs"):
        require(left_result[name] == right_result[name], "Result field differs: " + name)
    require(right_result["epochs"] == config["epochs"], "Result has the wrong epoch count")
    metric_difference = finite_max_difference(left_result["metrics"], right_result["metrics"], "Metrics")

    left_state, right_state = (load_local_checkpoint(folder / "model_1.pt") for folder in (reference, resumed))
    require(left_state.keys() == right_state.keys() and bool(left_state), "Model parameter keys differ or are empty")
    weight_difference = 0.0
    for name, left in left_state.items():
        right = right_state[name]
        require(left.dtype == right.dtype, "Model parameter dtype differs: " + name)
        weight_difference = max(weight_difference, finite_max_difference(left.numpy(), right.numpy(), "Weight " + name))

    histories = [[json.loads(line) for line in (folder / "history_1.jsonl").read_text(encoding="utf-8").splitlines()]
                 for folder in (reference, resumed)]
    for history in histories:
        require([row["epoch"] for row in history] == list(range(1, config["epochs"] + 1)),
                "History must contain every epoch exactly once")
    require([row["lr_next"] for row in histories[0]] == [row["lr_next"] for row in histories[1]],
            "Learning-rate schedule differs")
    return {"prediction_max_abs_difference": prediction_difference,
            "metric_max_abs_difference": metric_difference,
            "weight_max_abs_difference": weight_difference,
            "split_and_evaluation_data_identical": True, "learning_rates_identical": True,
            "all_compared_arrays_exact": prediction_difference == metric_difference == weight_difference == 0.0}


def verify_reference(reference):
    manifest = read_json(reference / "manifest.json")
    config = manifest["config"]
    require(config["subjects"] == [1] and config["implementation"] == "code-repaired",
            "Use a subject-1-only code-repaired reference run")
    require(config["epochs"] > STOP_EPOCH, "Reference must train beyond epoch 25")
    summary = read_json(reference / "summary.json")
    require(summary["status"] == "complete" and summary["completed_subjects"] == [1], "Reference is incomplete")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    environment = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "gpu": torch.cuda.get_device_name(0)}
    for name, current in environment.items():
        require(manifest.get(name) == current, "Reference environment differs for %s: %r vs %r" %
                (name, manifest.get(name), current))
    require(bool(config["sha256"]), "Reference has no source hashes")
    for name, expected in config["sha256"].items():
        if name.startswith("data/"):
            path = Path(config["data"]) / Path(name).name
        else:
            path = (ROOT / name).resolve()
            require(ROOT in path.parents, "Unsafe reference source path: " + name)
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "Source/data changed since reference: " + name)
    return config, environment


def run_child(command, log_path):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Completed uninterrupted subject-1 run.")
    parser.add_argument("--output", type=Path, required=True, help="New directory reserved for this validation.")
    args = parser.parse_args()
    reference, output = args.reference.resolve(strict=True), args.output.resolve()
    require(not output.exists(), "Validation output already exists; use a new directory")
    require(reference not in output.parents, "Validation output must be outside the reference directory")
    config, environment = verify_reference(reference)
    resumed = output / "resumed_run"
    output.mkdir(parents=True)
    arguments = ["--implementation", "code-repaired", "--subjects", "1", "--data", config["data"],
                 "--output", str(resumed), "--epochs", str(config["epochs"]),
                 "--batch-size", str(config["batch_size"]), "--seed", str(config["seed"]),
                 "--threads", str(config["threads"])]
    plan = {"reference": str(reference), "output": str(output), "config": config,
            "environment": environment, "stop_epoch": STOP_EPOCH, "absolute_tolerance": ATOL,
            "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (output / "test_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("STEP 1: Train a separate run; stop after saving epoch 25.", flush=True)
    interrupted_log = output / "interrupted.log"
    code = run_child([sys.executable, "-u", "-c", STOP_DRIVER] + arguments, interrupted_log)
    require(code == STOP_EXIT_CODE, "Expected intentional exit 75, received %d; inspect interrupted.log" % code)
    require("INTENTIONAL_CHECKPOINT_STOP: epoch=25" in interrupted_log.read_text(encoding="utf-8"),
            "Intentional stop marker is missing")
    checkpoint_path = resumed / "resume_1.pt"
    checkpoint = load_local_checkpoint(checkpoint_path)
    require(checkpoint["epoch"] == STOP_EPOCH, "Checkpoint has the wrong epoch")
    require(bool(checkpoint["optimizer"]["state"]), "Checkpoint lacks optimizer state")
    require(checkpoint["scheduler"]["last_epoch"] == STOP_EPOCH, "Checkpoint lacks the expected scheduler state")
    require(checkpoint["cpu_rng"].numel() > 0 and checkpoint["cuda_rng"].numel() > 0, "Checkpoint lacks RNG states")
    require(not (resumed / "subject_1.json").exists(), "Interrupted run unexpectedly completed")
    shutil.copy2(checkpoint_path, output / "interrupted_checkpoint.pt")
    del checkpoint

    print("STEP 2: Resume in a new process from epoch 25.", flush=True)
    resumed_log = output / "resumed.log"
    code = run_child([sys.executable, "-u", str(ROOT / "run_reproduction.py")] + arguments + ["--resume"], resumed_log)
    require(code == 0, "Resume failed with exit %d; inspect resumed.log" % code)
    require("epoch=25/%d implementation=code-repaired" % config["epochs"] in resumed_log.read_text(encoding="utf-8"),
            "Training did not report resuming from epoch 25")

    print("STEP 3: Compare with the completed uninterrupted reference.", flush=True)
    comparison = compare_runs(reference, resumed)
    report = dict(plan, status="passed", **comparison)
    report["scope"] = "Controlled stop after checkpoint save; OS logoff/disconnection and interrupted writes are not tested."
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)
    print("RESUME_VERIFICATION_PASSED:", output / "verification.json", flush=True)


if __name__ == "__main__":
    main()
