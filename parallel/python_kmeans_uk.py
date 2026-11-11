#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python parallel Lloyd k-means (ProcessPool + NumPy), same protocol as Spark/MR.

Shared init centroids from data/parallel/fair/, k from fair_config.env (default 3).
Stop when max centre shift <= 1e-4. max_iter=300 is only a safety cap.
The timing file records this machine's CPU and memory.
"""
from __future__ import print_function


import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machine_info

DIM = 6
EPS = 1e-4


def load_matrix(path):
    return np.loadtxt(path, dtype=np.float64)


def chunk_ranges(n, workers):
    workers = max(1, min(int(workers), int(n)))
    base, rem = divmod(n, workers)
    ranges = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _assign_chunk(args):
    """Worker: assign one row slice to the nearest centroid; return sums/counts.

    The parent passes only that slice. On Windows, multiprocessing pickles every
    argument, so passing the full matrix once per worker would copy it many times.
    """
    chunk, centroids, k = args
    # (n, k) squared distances via ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c
    x2 = np.sum(chunk * chunk, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1)
    dots = chunk.dot(centroids.T)
    d2 = x2 + c2 - 2.0 * dots
    labels = np.argmin(d2, axis=1)
    sums = np.zeros((k, DIM), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for c in range(k):
        mask = labels == c
        counts[c] = int(mask.sum())
        if counts[c] > 0:
            sums[c] = chunk[mask].sum(axis=0)
    return sums, counts


def run_kmeans(points, centroids, k, max_iter, workers):
    n = points.shape[0]
    ranges = chunk_ranges(n, workers)
    # Pass read-only views; on Windows spawn, arrays are pickled once per call.
    pool = mp.Pool(processes=len(ranges)) if workers > 1 else None
    cents = centroids.copy()
    iter_done = 0
    try:
        for it in range(1, max_iter + 1):
            iter_done = it
            payloads = [
                (np.ascontiguousarray(points[lo:hi]), cents, k) for lo, hi in ranges
            ]
            if pool is None:
                parts = [_assign_chunk(payloads[0])]
            else:
                parts = pool.map(_assign_chunk, payloads)

            sums = np.zeros((k, DIM), dtype=np.float64)
            counts = np.zeros(k, dtype=np.int64)
            for ps, pc in parts:
                sums += ps
                counts += pc

            updated = cents.copy()
            for c in range(k):
                if counts[c] > 0:
                    updated[c] = sums[c] / float(counts[c])

            shift = float(np.sqrt(np.sum((cents - updated) ** 2, axis=1).max()))
            print("Iter %d max shift = %.8f" % (it, shift))
            sys.stdout.flush()
            cents = updated
            if shift <= EPS:
                print("Stop: max shift %.8f <= epsilon %.1e" % (shift, EPS))
                break
        else:
            print("Stop: reached max_iter=%d before epsilon %.1e" % (max_iter, EPS))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return cents, iter_done


def main():
    # Prefer repo-relative defaults when run on the Windows host
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_housing = os.environ.get("HOUSING", here)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=os.path.join(default_housing, "data", "parallel", "fair", "n200000.txt"),
    )
    ap.add_argument(
        "--centroids",
        default=os.path.join(
            default_housing, "data", "parallel", "fair", "init_centroids_k3.txt"
        ),
    )
    ap.add_argument("--k", type=int, default=int(os.environ.get("KMEANS_K", "3")))
    ap.add_argument("--max-iter", type=int, default=int(os.environ.get("KMEANS_MAX_ITER", "300")))
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PYTHON_KMEANS_WORKERS") or mp.cpu_count() or 1),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(default_housing, "data", "processed", "python_parallel_timing.txt"),
    )
    args = ap.parse_args()

    if not os.path.isfile(args.data) or not os.path.isfile(args.centroids):
        raise SystemExit(
            "Missing fair input.\n  data=%s\n  centroids=%s\n"
            "On one machine run: python parallel/prepare_fair_inputs.py\n"
            "Then copy data/parallel/uk_features_scaled_all.txt and data/parallel/fair/ "
            "with the project." % (args.data, args.centroids)
        )

    print("engine=Python ProcessPool+NumPy")
    print("data=%s" % args.data)
    print("centroids=%s" % args.centroids)
    print("k=%d max_iter=%d workers=%d" % (args.k, args.max_iter, args.workers))
    sys.stdout.flush()

    t_load0 = time.time()
    points = load_matrix(args.data)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[:, :DIM]
    centroids = load_matrix(args.centroids)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    centroids = centroids[:, :DIM]
    if centroids.shape[0] != args.k:
        raise SystemExit("expected %d centroids, got %d" % (args.k, centroids.shape[0]))
    t_load1 = time.time()
    print("points=%d load_seconds=%.3f" % (points.shape[0], t_load1 - t_load0))
    sys.stdout.flush()

    t0 = time.time()
    final, iters = run_kmeans(points, centroids, args.k, args.max_iter, args.workers)
    wall = time.time() - t0
    print("=== Python parallel k-means done ===")
    print("iters=%d wall_seconds=%.3f workers=%d" % (iters, wall, args.workers))
    for i, row in enumerate(final):
        print("Centroid %d: %s" % (i, " ".join("%.8f" % x for x in row)))

    peak_self, peak_children = machine_info.peak_rss_pair()
    rows = [
        ("engine", "Python_ProcessPool_NumPy"),
        ("tag", args.tag),
        ("points", str(points.shape[0])),
        ("k", str(args.k)),
        ("epsilon", "%.1e" % EPS),
        ("max_iter", str(args.max_iter)),
        ("iters", str(iters)),
        ("workers", str(args.workers)),
        ("wall_seconds", "%.6f" % wall),
        ("load_seconds", "%.6f" % (t_load1 - t_load0)),
        ("peak_rss_bytes", peak_self),
        ("peak_rss_children_bytes", peak_children),
        ("data", args.data),
        ("centroids", args.centroids),
    ]
    machine_info.write_result(args.out, rows)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    mp.freeze_support()
    main()
