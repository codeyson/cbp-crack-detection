"""
make_split.py - split hand-sorted crops into train/val/test (70/15/15 per grade, fixed seed).

    python evaluation/make_split.py raw dataset      # raw/{A,B,C}/*.png -> dataset/{train,val,test}/{A,B,C}/

Run once. The test split is frozen after this: never tune on it.
"""
import argparse
import os
import random
import shutil

EXT = (".png", ".jpg", ".jpeg", ".bmp")


def split(src, dst, val=0.15, test=0.15, seed=0):
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists: delete it first (re-splitting changes the test set)")
    rng = random.Random(seed)
    for cls in sorted(os.listdir(src)):
        files = sorted(f for f in os.listdir(os.path.join(src, cls)) if f.lower().endswith(EXT))
        rng.shuffle(files)
        n_test, n_val = round(len(files) * test), round(len(files) * val)
        parts = dict(test=files[:n_test], val=files[n_test:n_test + n_val], train=files[n_test + n_val:])
        for part, names in parts.items():
            os.makedirs(os.path.join(dst, part, cls), exist_ok=True)
            for f in names:
                shutil.copy2(os.path.join(src, cls, f), os.path.join(dst, part, cls, f))
        print(f"{cls}: " + "  ".join(f"{p} {len(n)}" for p, n in parts.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with one subfolder per grade, e.g. raw/A raw/B raw/C")
    ap.add_argument("dst", help="output, e.g. dataset")
    a = ap.parse_args()
    split(a.src, a.dst)
