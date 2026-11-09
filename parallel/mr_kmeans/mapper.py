#!/usr/bin/env python
"""Hadoop Streaming mapper: assign each point to nearest centroid.
Centroids file path via env CENTROIDS_FILE (local path on task nodes).
Input line: 6 floats. Output: cluster_id \\t x1 x2 x3 x4 x5 x6 1
"""
from __future__ import print_function
import os
import sys

DIM = 6


def load_centroids(path):
    cents = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cents.append([float(x) for x in parts[:DIM]])
    return cents


def dist2(a, b):
    s = 0.0
    for i in range(DIM):
        d = a[i] - b[i]
        s += d * d
    return s


def closest(p, cents):
    best = 0
    best_d = dist2(p, cents[0])
    for i in range(1, len(cents)):
        d = dist2(p, cents[i])
        if d < best_d:
            best_d = d
            best = i
    return best


def main():
    path = os.environ.get("CENTROIDS_FILE", "centroids.txt")
    cents = load_centroids(path)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        p = [float(x) for x in line.split()[:DIM]]
        c = closest(p, cents)
        print("%d\t%s 1" % (c, " ".join("%.8f" % x for x in p)))


if __name__ == "__main__":
    main()
