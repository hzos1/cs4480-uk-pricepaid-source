# -*- coding: utf-8 -*-
"""Local Lloyd on one row piece, and the shared stitch of those centres.

Each piece starts from the same two centres and stops at the same 1e-4 rule.
The stitch is Lloyd again, on the local centres only, from those same centres.
"""
from __future__ import print_function

import argparse
import json
import time

import numpy as np

DIM = 6
EPS = 1e-4


def chunk_ranges(n, pieces):
    pieces = max(1, min(int(pieces), int(n)))
    base, rem = divmod(int(n), pieces)
    ranges = []
    start = 0
    for i in range(pieces):
        size = base + (1 if i < rem else 0)
        ranges.append((start, size))
        start += size
    return ranges


def lloyd_numpy(points, centroids, k, max_iter, eps):
    """Same update as the single-process Python run. Returns centres, counts, iters, shift, seconds."""
    cents = np.array(centroids, dtype=np.float64, copy=True)
    counts = np.zeros(int(k), dtype=np.int64)
    iters = 0
    shift = float("nan")
    t0 = time.perf_counter()
    for it in range(1, int(max_iter) + 1):
        iters = it
        x2 = np.sum(points * points, axis=1, keepdims=True)
        c2 = np.sum(cents * cents, axis=1)
        labels = np.argmin(x2 + c2 - 2.0 * points.dot(cents.T), axis=1)
        updated = cents.copy()
        counts[:] = 0
        for c in range(int(k)):
            mask = labels == c
            counts[c] = int(mask.sum())
            if counts[c] > 0:
                updated[c] = points[mask].sum(axis=0) / np.float64(counts[c])
        delta = updated - cents
        shift = float(np.sqrt(np.sum(delta * delta, axis=1)).max())
        cents = updated
        if shift <= eps:
            break
    return cents, counts, iters, shift, time.perf_counter() - t0


def cpu_piece(payload):
    """Spark task. Read one contiguous piece and run Lloyd inside it."""
    raw, start, nrows, centroids, k, max_iter, eps, piece = payload
    t0 = time.perf_counter()
    with open(raw, "rb") as handle:
        handle.seek(int(start) * DIM * 8)
        buf = handle.read(int(nrows) * DIM * 8)
    points = np.frombuffer(buf, dtype="<f8").reshape(int(nrows), DIM).copy()
    read_s = time.perf_counter() - t0
    cents, counts, iters, shift, lloyd_s = lloyd_numpy(points, centroids, k, max_iter, eps)
    return {
        "piece": int(piece),
        "start": int(start),
        "rows": int(nrows),
        "iters": int(iters),
        "shift": float(shift),
        "read_seconds": float(read_s),
        "lloyd_seconds": float(lloyd_s),
        "counts": [int(c) for c in counts],
        "centres": cents.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--piece", type=int, required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--eps", type=float, default=EPS)
    args = ap.parse_args()
    cents = np.loadtxt(args.centroids, dtype=np.float64)
    if cents.ndim == 1:
        cents = cents.reshape(1, -1)
    result = cpu_piece(
        (args.raw, args.start, args.rows, cents[:, :DIM].tolist(), args.k, args.max_iter, args.eps, args.piece)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
