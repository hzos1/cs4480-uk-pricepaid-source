#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build UK scaling inputs: small samples via NumPy; large via file reuse / concat."""
from __future__ import print_function

import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAR = os.path.join(ROOT, "data", "parallel")
FULL = os.path.join(PAR, "uk_features_scaled_2018plus.txt")
SAMPLE = os.path.join(PAR, "uk_features_scaled_2018plus_sample10pct.txt")


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def write_sample(path, n, src_path, seed=42):
    rng = np.random.RandomState(seed)
    # reservoir from text to avoid loading full into RAM twice when possible
    print("sampling", n, "from", src_path)
    # For speed on Windows host: loadtxt once from SAMPLE if enough rows
    src = np.loadtxt(src_path, dtype=np.float64)
    if src.ndim == 1:
        src = src.reshape(1, -1)
    src = src[:, :6]
    if n <= src.shape[0]:
        idx = rng.choice(src.shape[0], size=n, replace=False)
        arr = src[idx]
    else:
        raise SystemExit("need larger source for n=%d" % n)
    np.savetxt(path, arr, fmt="%.8f")


def concat_copies(out_path, src_path, copies):
    print("concat", copies, "x", src_path, "->", out_path)
    if os.path.isfile(out_path):
        os.remove(out_path)
    with open(out_path, "wb") as out:
        for i in range(copies):
            print("  copy", i + 1, "/", copies)
            with open(src_path, "rb") as inp:
                shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)


def main():
    if not os.path.isfile(FULL) or not os.path.isfile(SAMPLE):
        raise SystemExit("missing FULL or SAMPLE under data/parallel")

    n_full = count_lines(FULL)
    n_sample = count_lines(SAMPLE)
    print("full=", n_full, "sample=", n_sample)

    jobs = []

    # 500k from sample
    p5 = os.path.join(PAR, "uk_features_scaled_scale_n5e5.txt")
    if not os.path.isfile(p5) or count_lines(p5) != 500000:
        write_sample(p5, 500000, SAMPLE)
    jobs.append(("n5e5", 500000, p5))

    # 855k = existing sample (alias copy once for uniform naming)
    p855 = os.path.join(PAR, "uk_features_scaled_scale_n855k.txt")
    if not os.path.isfile(p855):
        print("copy sample ->", p855)
        shutil.copyfile(SAMPLE, p855)
    jobs.append(("n855k", n_sample, p855))

    # 2M from full
    p2 = os.path.join(PAR, "uk_features_scaled_scale_n2e6.txt")
    if not os.path.isfile(p2) or abs(count_lines(p2) - 2000000) > 0:
        write_sample(p2, 2000000, FULL)
    jobs.append(("n2e6", 2000000, p2))

    # ~8.55M = full (pointer copy)
    p8 = os.path.join(PAR, "uk_features_scaled_scale_n8p5e6.txt")
    if not os.path.isfile(p8):
        print("copy full ->", p8)
        shutil.copyfile(FULL, p8)
    jobs.append(("n8p5e6", n_full, p8))

    # ~17M = 2x full (disk stress for in-memory Python)
    p17 = os.path.join(PAR, "uk_features_scaled_scale_n1p7e7.txt")
    want17 = n_full * 2
    if not os.path.isfile(p17) or count_lines(p17) != want17:
        concat_copies(p17, FULL, 2)
    jobs.append(("n1p7e7", want17, p17))

    # ~25.6M = 3x full
    p25 = os.path.join(PAR, "uk_features_scaled_scale_n2p6e7.txt")
    want25 = n_full * 3
    if not os.path.isfile(p25) or count_lines(p25) != want25:
        concat_copies(p25, FULL, 3)
    jobs.append(("n2p6e7", want25, p25))

    man = os.path.join(PAR, "scaling_manifest.csv")
    with open(man, "w") as f:
        f.write("tag,points,path\n")
        for tag, n, path in jobs:
            actual = count_lines(path)
            f.write("%s,%d,%s\n" % (tag, actual, path.replace("\\", "/")))
            print("OK", tag, "points=", actual, "MB=", round(os.path.getsize(path) / 1e6, 1))
    print("wrote", man)


if __name__ == "__main__":
    main()
