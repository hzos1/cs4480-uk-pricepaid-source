#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One reducer. Lloyd on the local centres, same initial centres as the mappers."""
from __future__ import print_function

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stitch_piece

DIM = 6


def main():
    init = np.loadtxt("centroids.txt", dtype=np.float64)
    if init.ndim == 1:
        init = init.reshape(1, -1)
    init = np.ascontiguousarray(init[:, :DIM], dtype=np.float64)
    pieces = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        payload = line.split("\t", 1)[-1]
        pieces.append(json.loads(payload))
    rows = []
    for piece in sorted(pieces, key=lambda item: item["piece"]):
        for c, centre in enumerate(piece["centres"]):
            if piece["counts"][c] > 0:
                rows.append(centre)
    points = np.asarray(rows, dtype=np.float64)
    cents, counts, iters, shift, seconds = stitch_piece.lloyd_numpy(
        points, init, init.shape[0], 300, stitch_piece.EPS
    )
    result = {
        "pieces": pieces,
        "stitch_iters": int(iters),
        "stitch_seconds": float(seconds),
        "stitch_shift": float(shift),
        "stitch_counts": [int(c) for c in counts],
        "centres": cents.tolist(),
        "local_centres_used": int(points.shape[0]),
    }
    sys.stdout.write("result\t%s\n" % json.dumps(result))


if __name__ == "__main__":
    main()
