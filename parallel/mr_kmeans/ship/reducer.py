#!/usr/bin/env python
"""Hadoop Streaming reducer: average points per cluster.
Input: cluster_id \\t x1..x6 1
Output: cluster_id \\t mean1..mean6
"""
from __future__ import print_function
import sys

DIM = 6


def flush(cid, sums, count):
    if cid is None or count == 0:
        return
    means = [s / count for s in sums]
    print("%d\t%s" % (int(cid), " ".join("%.8f" % m for m in means)))


def main():
    cur = None
    sums = [0.0] * DIM
    count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        key, rest = line.split("\t", 1)
        parts = rest.split()
        vals = [float(x) for x in parts[:DIM]]
        ones = float(parts[DIM]) if len(parts) > DIM else 1.0
        if cur is None:
            cur = key
        if key != cur:
            flush(cur, sums, count)
            cur = key
            sums = [0.0] * DIM
            count = 0
        for i in range(DIM):
            sums[i] += vals[i]
        count += ones
    flush(cur, sums, count)


if __name__ == "__main__":
    main()
