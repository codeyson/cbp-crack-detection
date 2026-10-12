"""
make_split.py - split hand-sorted crops into train/val/test (70/15/15 per grade, fixed seed).

    python evaluation/make_split.py raw dataset      # raw/{A,B,C}/*.png -> dataset/{train,val,test}/{A,B,C}/
    python evaluation/make_split.py --selftest

Byte-identical images are kept once (copies would land in both train and test and inflate accuracy). An image found
under more than one grade is dropped (conflicting label).
Run once. The test split is frozen after this: never tune on it.
"""
import argparse
import hashlib
import os
import random
import shutil
import tempfile
from collections import defaultdict

EXT = (".png", ".jpg", ".jpeg", ".bmp")


def unique_files(src):
    """{grade: [file, ...]} with one file per distinct image, images in more than one grade removed."""
    seen = defaultdict(dict)                                          # hash -> {grade: first file}
    for cls in sorted(os.listdir(src)):
        for f in sorted(os.listdir(os.path.join(src, cls))):
            if f.lower().endswith(EXT):
                with open(os.path.join(src, cls, f), "rb") as fh:
                    seen[hashlib.md5(fh.read()).hexdigest()].setdefault(cls, f)
    out = defaultdict(list)
    for grades in seen.values():
        if len(grades) == 1:
            (cls, f), = grades.items()
            out[cls].append(f)
    conflicts = sum(len(g) > 1 for g in seen.values())
    print(f"{len(seen)} distinct images, {conflicts} dropped (in more than one grade)")
    return {cls: sorted(files) for cls, files in out.items()}


def split(src, dst, val=0.15, test=0.15, seed=0):
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists: delete it first (re-splitting changes the test set)")
    rng = random.Random(seed)
    for cls, files in sorted(unique_files(src).items()):
        total = len([f for f in os.listdir(os.path.join(src, cls)) if f.lower().endswith(EXT)])
        rng.shuffle(files)
        n_test, n_val = round(len(files) * test), round(len(files) * val)
        parts = dict(test=files[:n_test], val=files[n_test:n_test + n_val], train=files[n_test + n_val:])
        for part, names in parts.items():
            os.makedirs(os.path.join(dst, part, cls), exist_ok=True)
            for f in names:
                shutil.copy2(os.path.join(src, cls, f), os.path.join(dst, part, cls, f))
        print(f"{cls}: {len(files)} unique of {total}  " + "  ".join(f"{p} {len(n)}" for p, n in parts.items()))


def selftest():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "raw")
        for cls, names in dict(A=["a1", "a2", "dup", "x"], B=["b1", "x"]).items():
            os.makedirs(os.path.join(src, cls))
            for i, n in enumerate(names):                           # A/a2 and A/dup are the same bytes
                with open(os.path.join(src, cls, f"{i}.png"), "w") as fh:
                    fh.write("a2" if n == "dup" else n)
        got = unique_files(src)
        assert got == dict(A=["0.png", "1.png"], B=["0.png"]), got   # copy kept once, "x" in both grades dropped
        split(src, os.path.join(d, "dataset"))
        out = [f for _, _, fs in os.walk(os.path.join(d, "dataset")) for f in fs]
        assert len(out) == 3, out
    print("selftest: ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("src", nargs="?", help="folder with one subfolder per grade, e.g. raw/A raw/B raw/C")
    ap.add_argument("dst", nargs="?", help="output, e.g. dataset")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif not (a.src and a.dst):
        ap.error("src and dst are required")
    else:
        split(a.src, a.dst)
