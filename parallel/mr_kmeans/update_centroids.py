#!/usr/bin/env python
"""Merge reducer output into a full k-centroid file; print SHIFT=..."""
from __future__ import print_function
import sys

DIM = 6


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()[:DIM]])
    return rows


def main():
    old_path, new_path, out_path, eps_s, k_s = sys.argv[1:6]
    eps = float(eps_s)
    k = int(k_s)
    old = load_rows(old_path)
    while len(old) < k:
        old.append([0.0] * DIM)
    new = [list(c) for c in old]
    with open(new_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, rest = line.split("\t", 1)
            cid = int(float(key))
            vals = [float(x) for x in rest.split()[:DIM]]
            if 0 <= cid < k:
                new[cid] = vals
    shift = 0.0
    for a, b in zip(old, new):
        d = sum((x - y) * (x - y) for x, y in zip(a, b)) ** 0.5
        if d > shift:
            shift = d
    with open(out_path, "w") as f:
        for row in new:
            f.write(" ".join("%.8f" % x for x in row) + "\n")
    print("SHIFT=%.8f" % shift)
    with open(old_path + ".shift", "w") as f:
        f.write("%.8f" % shift)
    with open(old_path + ".moved", "w") as f:
        f.write("1" if shift > eps else "0")


if __name__ == "__main__":
    main()
