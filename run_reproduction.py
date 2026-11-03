"""Resumable DMER subject-dependent experiment and automatic paper comparison.

Uses 80/20 sample split, 300 epochs, batch 128, Adam 1e-3. Never selects an
epoch/hyperparameter using test data. The final epoch is the reported model.
Both implementations remain explicitly qualified, not exact reproductions.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import DMERDataset
from models.model import HeLo
from models.paper_model import PaperHeLo
from params import Config
from reproduction_metrics import NAMES, PAPER_DMER_SD, distribution_metrics

ROOT = Path(__file__).resolve().parent


def source_revision():
    """Allow a checked transfer archive without requiring a .git directory."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance = ROOT / "SOURCE_PROVENANCE.json"
        if provenance.exists():
            return json.loads(provenance.read_text(encoding="utf-8")).get("git_head")
        return None


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def evaluate(model, tensors, indices, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for index in indices.split(batch_size):
            # True labels are used only to construct a diagnostic loss target,
            # not as an input to the prediction branch of either implementation.
            output = model(*(field[index] for field in tensors))[0]
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite predictions")
            predictions.append(output.cpu().numpy())
    pred = np.concatenate(predictions)
    label = tensors[-1][indices].cpu().numpy()
    return pred, label, distribution_metrics(pred, label).mean(0)


def write_comparison(output, subjects, implementation, epochs):
    results = []
    for subject in subjects:
        path = output / ("subject_%d.json" % subject)
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        return
    mean = np.mean([row["metrics"] for row in results], axis=0)
    baseline = np.mean([row["train_mean_baseline"] for row in results], axis=0)
    clean = [row for row in results if row["subject"] != 36]
    complete = len(results) == len(subjects)
    summary = {"status": "complete" if complete else "running", "implementation": implementation,
               "completed_subjects": [r["subject"] for r in results], "planned_subjects": subjects,
               "epochs_per_subject": epochs, "metrics_order": NAMES, "subject_macro_average": mean.tolist(),
               "paper_dmer_subject_dependent": PAPER_DMER_SD.tolist(), "difference_from_paper": (mean - PAPER_DMER_SD).tolist(),
               "train_mean_baseline": baseline.tolist(),
               "excluding_subject_36": np.mean([r["metrics"] for r in clean], axis=0).tolist() if clean else None,
               "qualification": "Not an exact reproduction: label preprocessing and equation/code discrepancies remain unresolved. Partial cohorts are not comparable to the paper's 73-subject average."}
    save_json(output / "summary.json", summary)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "direction", "paper_73_subjects", "current_subject_mean", "difference", "train_mean_baseline", "completed_subjects"])
        for i, name in enumerate(NAMES):
            writer.writerow([name, "lower" if i < 4 else "higher", PAPER_DMER_SD[i], mean[i], mean[i] - PAPER_DMER_SD[i], baseline[i], len(results)])
    lines = ["# DMER 被试内实验对比", "", "状态：%s；已完成 %d/%d 个被试，每人 %d 轮。" %
             ("本次任务完成" if complete else "运行中", len(results), len(subjects), epochs),
             "", "实现：`%s`。这是有明确假设的复现实验，不能标为作者设置已完全复现。" % implementation,
             "原始评分归一化、公式解释和公开实现的差异见 `REPRODUCTION_PROTOCOL.md`。少量被试结果不能与论文全体平均值作等价性能结论。",
             "", "| 指标 | 方向 | 论文73人 | 本次已完成被试均值 | 差值（本次−论文） | 训练集均值基线 |",
             "|---|---|---:|---:|---:|---:|"]
    for i, name in enumerate(NAMES):
        lines.append("| %s | %s | %.4f | %.6f | %+.6f | %.6f |" %
                     (name, "↓" if i < 4 else "↑", PAPER_DMER_SD[i], mean[i], mean[i]-PAPER_DMER_SD[i], baseline[i]))
    lines += ["", "每个被试先对全部测试样本计算均值，再对被试等权平均。基线仅使用该被试的训练标签均值。",
              "不使用测试集挑选最佳轮数；报告固定最后一轮。每人拆分索引、预测、标签、指标和最终权重均保存在本目录。",
              "", "36号的原始评分13如被纳入则原样保留，summary.json同时给出去掉36号的均值；这不能消除评分来源待核实的问题。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "DMER_raw_preserved")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=["paper-equations", "code-repaired"], default="paper-equations")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if options.epochs <= 0 or options.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the configured CUDA environment")
    torch.set_num_threads(options.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    subjects = options.subjects or sorted(int(p.stem) for p in options.data.glob("*.mat") if p.stem.isdigit())
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Subject list must be nonempty and unique")
    source_files = ["run_reproduction.py", "reproduction_metrics.py", "dataset.py", "params.py", "models/model.py",
                    "models/paper_model.py", "models/transformer_layers.py", "models/OT_torch_.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    hashes.update({"data/%d.mat" % subject: hashlib.sha256((options.data / (str(subject) + ".mat")).read_bytes()).hexdigest() for subject in subjects})
    config = {"implementation": options.implementation, "subjects": subjects, "epochs": options.epochs,
              "batch_size": options.batch_size, "seed": options.seed, "threads": options.threads,
              "train_ratio": 0.8, "split": "within-subject random samples", "lr": 0.001,
              "weight_decay": 1e-5, "lr_gamma": 0.99,
              "loss": "KL sum + batch-mean squared Frobenius CC" if options.implementation == "paper-equations" else "KL sum + element-mean MSE CC",
              "sinkhorn_epsilon": 0.1, "sinkhorn_iterations": 50,
              "data": str(options.data.resolve()), "sha256": hashes}
    output = options.output.resolve()
    manifest = output / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if not options.resume or existing["config"] != config:
            raise ValueError("Existing run needs --resume with identical configuration, source code and data")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty")
        output.mkdir(parents=True, exist_ok=True)
        save_json(manifest, {"config": config, "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                            "gpu": torch.cuda.get_device_name(0),
                            "git_head": source_revision()})
        for name in source_files + ["REPRODUCTION_PROTOCOL.md"]:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    cfg = Config(); cfg.initialize(); args = cfg.parser.parse_args([])
    for subject in subjects:
        result_path = output / ("subject_%d.json" % subject)
        if result_path.exists():
            continue
        start = time.perf_counter()
        random.seed(options.seed); np.random.seed(options.seed)
        torch.manual_seed(options.seed); torch.cuda.manual_seed_all(options.seed)
        ds = DMERDataset(str(options.data / (str(subject) + ".mat")))
        tensors = tuple(torch.as_tensor(np.array(getattr(ds, name), copy=True), dtype=torch.float32, device="cuda")
                        for name in ("eeg", "gsr", "ppg", "video", "label"))
        permutation = torch.randperm(len(ds), generator=torch.Generator().manual_seed(options.seed))
        n_train = int(0.8 * len(ds))
        train_cpu, test_cpu = permutation[:n_train], permutation[n_train:]
        train_index, test_index = train_cpu.cuda(), test_cpu.cuda()
        np.savez_compressed(output / ("split_%d.npz" % subject), train=train_cpu.numpy(), test=test_cpu.numpy(), vids=ds.data["vids"])
        train_mean = ds.label[train_cpu.numpy()].mean(0)
        baseline = distribution_metrics(np.broadcast_to(train_mean, (len(test_cpu), 10)), ds.label[test_cpu.numpy()]).mean(0)
        model = (PaperHeLo(args) if options.implementation == "paper-equations" else HeLo(args)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        checkpoint = output / ("resume_%d.pt" % subject)
        first_epoch = 0
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
            first_epoch = state["epoch"]
        torch.cuda.reset_peak_memory_stats()
        print("START subject=%d samples=%d train=%d test=%d epoch=%d/%d implementation=%s" %
              (subject, len(ds), n_train, len(test_cpu), first_epoch, options.epochs, options.implementation), flush=True)
        for epoch in range(first_epoch, options.epochs):
            model.train()
            order = train_index[torch.randperm(n_train, device="cuda")]
            totals = torch.zeros(3, device="cuda")
            for index in order.split(options.batch_size):
                optimizer.zero_grad(set_to_none=True)
                pred, corr, target = model(*(field[index] for field in tensors))
                kl = F.kl_div(pred.log(), tensors[-1][index], reduction="sum")
                cc = ((corr-target)**2).sum((1, 2)).mean() if options.implementation == "paper-equations" else F.mse_loss(corr, target)
                loss = kl + cc
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss at subject %d epoch %d" % (subject, epoch + 1))
                loss.backward()
                optimizer.step()
                totals += torch.stack((kl.detach(), cc.detach(), loss.detach()))
            scheduler.step()
            values = (totals / math.ceil(n_train/options.batch_size)).cpu().tolist()
            with (output / ("history_%d.jsonl" % subject)).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch + 1, "kl_batch_mean": values[0], "cc_batch_mean": values[1], "loss_batch_mean": values[2], "lr_next": scheduler.get_last_lr()[0]}) + "\n")
            if (epoch + 1) % 25 == 0 or epoch + 1 == options.epochs:
                save_checkpoint(checkpoint, {"epoch": epoch + 1, "model": model.state_dict(),
                                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()})
                save_json(output / "progress.json", {"subject": subject, "epoch": epoch + 1, "epochs": options.epochs,
                                                     "elapsed_subject_seconds": time.perf_counter()-start})
                print("subject=%d epoch=%d/%d loss=%.6f elapsed=%.1fs" %
                      (subject, epoch+1, options.epochs, values[2], time.perf_counter()-start), flush=True)
        pred, label, metrics = evaluate(model, tensors, test_index, options.batch_size)
        torch.cuda.synchronize()
        np.savez_compressed(output / ("predictions_%d.npz" % subject), predictions=pred, labels=label,
                            indices=test_cpu.numpy(), video_ids=ds.data["vids"][test_cpu.numpy()])
        save_checkpoint(output / ("model_%d.pt" % subject), model.state_dict())
        result = {"subject": subject, "samples": len(ds), "train": n_train, "test": len(test_cpu),
                  "epochs": options.epochs, "metrics": metrics.tolist(), "train_mean_baseline": baseline.tolist(),
                  "elapsed_seconds_this_invocation": time.perf_counter()-start,
                  "peak_allocated_MiB": torch.cuda.max_memory_allocated()/1024**2,
                  "shared_video_ids_train_test": sorted(set(ds.data["vids"][train_cpu.numpy(), 0].tolist()) & set(ds.data["vids"][test_cpu.numpy(), 0].tolist()))}
        save_json(result_path, result)
        # Final weights and result exist; the rolling optimizer checkpoint is no longer needed.
        checkpoint.unlink(missing_ok=True)
        print("DONE subject=%d metrics=%s" % (subject, metrics.tolist()), flush=True)
        write_comparison(output, subjects, options.implementation, options.epochs)
        del model, optimizer, scheduler, tensors
        torch.cuda.empty_cache()
    write_comparison(output, subjects, options.implementation, options.epochs)
    print("EXPERIMENT COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
