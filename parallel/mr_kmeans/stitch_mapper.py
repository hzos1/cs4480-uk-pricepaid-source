#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One map task, one piece. Lloyd stays inside this split. Emit the local centres."""
from __future__ import print_function

import array
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stitch_piece

DIM = 6


def piece_id():
    name = os.environ.get("mapreduce_map_input_file", "")
    base = os.path.basename(name.replace("\\", "/"))
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        return int(digits)
    raw = os.environ.get("mapreduce_task_partition", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def main():
    cents = np.loadtxt("centroids.txt", dtype=np.float64)
    if cents.ndim == 1:
        cents = cents.reshape(1, -1)
    cents = np.ascontiguousarray(cents[:, :DIM], dtype=np.float64)
    buf = array.array("d")
    rows = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < DIM:
            continue
        buf.extend(float(parts[d]) for d in range(DIM))
        rows += 1
    points = np.frombuffer(buf, dtype=np.float64).reshape(rows, DIM).copy()
    centres, counts, iters, shift, lloyd_s = stitch_piece.lloyd_numpy(
        points, cents, cents.shape[0], 300, stitch_piece.EPS
    )
    payload = {
        "piece": piece_id(),
        "rows": int(rows),
        "iters": int(iters),
        "shift": float(shift),
        "read_seconds": 0.0,
        "lloyd_seconds": float(lloyd_s),
        "counts": [int(c) for c in counts],
        "centres": centres.tolist(),
    }
    sys.stdout.write("0\t%s\n" % json.dumps(payload))


if __name__ == "__main__":
    main()
