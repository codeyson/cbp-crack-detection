"""
train.py - train one design and record its training time (Manufacturability). Run in Colab, same GPU for all 3.

    python evaluation/train.py --design d1 --data dataset --epochs 30
    python evaluation/train.py --design d2 --data dataset --epochs 30
    python evaluation/train.py --design d3 --data dataset --epochs 30

Training Time (s) = End Time - Start Time, from time.perf_counter() around train().
Pretrained weights are downloaded BEFORE the timer starts, so download speed is not counted.
"""
import argparse
import csv
import datetime
import os
import platform
import time

import torch

import designs
from designs import DESIGNS

HERE = os.path.dirname(os.path.abspath(__file__))


def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    if not new:
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != list(row):   # columns changed: appending would shift values under the wrong headers
            raise SystemExit(f"{path} has old columns: rename it (e.g. to .old) and run again")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(design, data, epochs, imgsz=224, batch=32, out="models", results=f"{HERE}/results", pretrained=True):
    if pretrained:
        designs.build(design, pretrained=True)   # download/cache ImageNet weights outside the timer
    start = time.perf_counter()
    weights = designs.train(design, data, epochs=epochs, imgsz=imgsz, out=out, batch=batch, pretrained=pretrained)
    end = time.perf_counter()
    row = dict(design=design, model=DESIGNS[design], training_time_s=round(end - start, 2), epochs=epochs,
               imgsz=imgsz, batch=batch, n_train=sum(len(fs) for _, _, fs in os.walk(os.path.join(data, "train"))),
               hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
               device=platform.node(), weights=weights, when=datetime.datetime.now().isoformat(timespec="seconds"))
    append_row(os.path.join(results, "training_time.csv"), row)
    print(f"{design}: training time {row['training_time_s']} s on {row['hardware']} -> {weights}")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=DESIGNS)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    run(a.design, a.data, a.epochs, a.imgsz, a.batch, a.out)
