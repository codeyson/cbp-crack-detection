"""
evaluate.py - measure one design against the design constraints. Run on the rig laptop, one design per run
(separate processes, so one model's memory does not count against another).

Each call does 50 runs by default (--runs): 50 fresh processes, one inspection cycle each, rows sharing one
batch id. tradeoff.py averages the newest batch per design and plots every run (runs.png).

Final (needs the labeled dataset and trained weights):
    python evaluation/evaluate.py --design d1
    python evaluation/evaluate.py --design d2
    python evaluation/evaluate.py --design d3
    python evaluation/tradeoff.py                        # table, winner per constraint, charts

Provisional speed/memory before the dataset exists (untrained model, any block crops):
    python evaluation/evaluate.py --design d1 --img_dir crops      # -> results/provisional/
    python evaluation/tradeoff.py --results evaluation/results/provisional

Self-test (fake data, a few minutes on CPU):
    python evaluation/evaluate.py --selftest

Misclassification Rate (%) = (1 - accuracy_score) x 100        on dataset/test, first cycle
Average Inference Time (s) = mean of perf_counter() around every predict() call
Average Memory Usage (%)   = mean of psutil memory_percent() after every predict(), over all cycles
Maintainability Index      = radon mi_visit() of the design's source files (designs.source_files), SLOC-weighted
Training Time (s)          = from results/training_time.csv (train.py)
"""
import argparse
import csv
import datetime
import glob
import os
import platform
import statistics
import subprocess
import sys
import time

import cv2
import psutil
from radon.metrics import mi_visit
from radon.raw import analyze
from sklearn.metrics import accuracy_score, confusion_matrix

import designs
from designs import DESIGNS
from make_split import EXT
from train import HERE, append_row


def test_items(data):
    """[(path, true_grade)] from data/test/{A,B,C}/."""
    root = os.path.join(data, "test")
    return [(os.path.join(root, cls, f), cls) for cls in sorted(os.listdir(root))
            for f in sorted(os.listdir(os.path.join(root, cls))) if f.lower().endswith(EXT)]


def maintainability(design):
    """Maintainability Index of all the design's source files, weighted by source lines of code."""
    mis, slocs = [], []
    for path in designs.source_files(design):
        with open(path, encoding="utf-8") as f:
            code = f.read()
        mis.append(mi_visit(code, multi=True))
        slocs.append(analyze(code).sloc)
    return sum(m * s for m, s in zip(mis, slocs)) / sum(slocs)


def evaluate(design, data="dataset", weights=None, img_dir=None, cycles=1, warmup=5, results=f"{HERE}/results",
             batch=""):
    """img_dir given: no labels, so misclassification is skipped (provisional speed/memory only) and the row
    goes to results/provisional/, never into the final table."""
    if img_dir:
        results = os.path.join(results, "provisional")
    items = ([(p, None) for p in sorted(glob.glob(f"{img_dir}/**/*", recursive=True)) if p.lower().endswith(EXT)]
             if img_dir else test_items(data))
    assert items, f"no images found in {img_dir or os.path.join(data, 'test')}"
    predict = designs.load(design, weights)
    proc = psutil.Process()
    for _ in range(warmup):   # first calls are slow (lazy init): not timed
        predict(cv2.imread(items[0][0]))

    times, mems, y_true, y_pred = [], [], [], []
    for cycle in range(cycles):
        for path, grade in items:
            img = cv2.imread(path)   # disk read is outside the timer
            start = time.perf_counter()
            pred = predict(img)
            times.append(time.perf_counter() - start)
            mems.append(proc.memory_percent())
            if cycle == 0 and grade is not None:
                y_true.append(grade)
                y_pred.append(pred)

    mi = maintainability(design)
    row = dict(design=design, model=DESIGNS[design], n_images=len(items), cycles=cycles,
               misclassification_pct=round((1 - accuracy_score(y_true, y_pred)) * 100, 2) if y_true else "",
               inference_mean_s=round(statistics.mean(times), 5), inference_std_s=round(statistics.pstdev(times), 5),
               memory_mean_pct=round(statistics.mean(mems), 3), memory_max_pct=round(max(mems), 3),
               maintainability_index=round(mi, 2), ram_total_gb=round(psutil.virtual_memory().total / 1e9, 1),
               cpu=platform.processor(), device=platform.node(), weights=weights or "untrained",
               when=datetime.datetime.now().isoformat(timespec="seconds"))
    row["batch"] = batch or row["when"]   # one id per --runs call; tradeoff.py averages the newest batch
    append_row(os.path.join(results, "evaluation.csv"), row)
    if y_true:
        labels = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        with open(os.path.join(results, f"confusion_{design}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true \\ pred"] + labels)
            w.writerows([lab] + list(r) for lab, r in zip(labels, cm))
    print("  ".join(f"{k}={v}" for k, v in row.items()))
    return row


def selftest():
    import tempfile

    import numpy as np

    import tradeoff
    import train

    tmp = tempfile.mkdtemp()
    data, results = os.path.join(tmp, "data"), os.path.join(tmp, "results")
    rng = np.random.default_rng(0)
    for part, n in dict(train=6, val=3, test=3).items():
        for cls, shade in dict(A=200, B=128, C=40).items():   # learnable: grade = brightness
            os.makedirs(os.path.join(data, part, cls))
            for i in range(n):
                img = np.clip(rng.normal(shade, 10, (64, 64, 3)), 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(data, part, cls, f"{i}.png"), img)
    for d in DESIGNS:
        r = train.run(d, data, epochs=1, imgsz=64, batch=4, out=os.path.join(tmp, "models"), results=results,
                      pretrained=False)
        assert r["training_time_s"] > 0 and os.path.exists(r["weights"]), r
        e = evaluate(d, img_dir=os.path.join(data, "test"), cycles=1, warmup=1, results=results)   # no labels
        assert e["misclassification_pct"] == "" and e["weights"] == "untrained", e
        e = evaluate(d, data, weights=r["weights"], cycles=2, warmup=1, results=results)
        assert 0 <= e["misclassification_pct"] <= 100 and e["inference_mean_s"] > 0, e
        assert 0 < e["memory_mean_pct"] <= 100 and 0 <= e["maintainability_index"] <= 100, e
        assert os.path.exists(os.path.join(results, f"confusion_{d}.csv"))
    assert all(r["weights"] == "untrained" for r in csv.DictReader(open(os.path.join(results, "provisional", "evaluation.csv"))))
    assert all(r["weights"] != "untrained" for r in csv.DictReader(open(os.path.join(results, "evaluation.csv"))))
    assert sorted(tradeoff.run(results, plots=False)["ranked"]) == sorted(DESIGNS)
    print("selftest: ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=DESIGNS)
    ap.add_argument("--data", default="dataset", help="needs dataset/test/{A,B,C}/")
    ap.add_argument("--weights", help="default: the design's file in models/")
    ap.add_argument("--img_dir", help="any block crops: no labels, untrained model, speed/memory only")
    ap.add_argument("--runs", type=int, default=50, help="separate runs (fresh process each), averaged by tradeoff.py")
    ap.add_argument("--cycles", type=int, default=1, help="full passes over the images per run (inspection cycles)")
    ap.add_argument("--batch", default="", help=argparse.SUPPRESS)   # set by the --runs loop
    ap.add_argument("--mi", action="store_true", help="print the Maintainability Index of each design")
    ap.add_argument("--results", default=f"{HERE}/results", help="e.g. evaluation/results/provisional")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.mi:
        for d in DESIGNS:
            print(f"{d} {DESIGNS[d]:20} MI {maintainability(d):.2f}  files: "
                  + ", ".join(os.path.basename(p) for p in designs.source_files(d)))
    elif a.design and a.runs > 1:
        batch = datetime.datetime.now().isoformat(timespec="seconds")
        cmd = [sys.executable, __file__, "--design", a.design, "--data", a.data, "--cycles", str(a.cycles),
               "--results", a.results, "--runs", "1", "--batch", batch]
        cmd += ["--weights", a.weights] if a.weights else []
        cmd += ["--img_dir", a.img_dir] if a.img_dir else []
        for i in range(a.runs):
            print(f"run {i + 1}/{a.runs}", flush=True)
            subprocess.run(cmd, check=True)
    elif a.design:
        evaluate(a.design, a.data, a.weights or (None if a.img_dir else designs.weights_path(a.design)),
                 a.img_dir, a.cycles, results=a.results, batch=a.batch)
    else:
        ap.error("give --design, --mi or --selftest")
