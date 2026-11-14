# -*- coding: utf-8 -*-
"""Split the full file into pieces, Lloyd inside each piece, then stitch.

CPU: one Spark local[N] job. Each task reads its own rows and does not
shuffle the full file on later rounds.
GPU: the same pieces, one after another, on the RTX 3070. The stitch is the
same Lloyd step on the local centres.
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machine_info
import stitch_piece

DIM = stitch_piece.DIM
EPS = stitch_piece.EPS
N_FULL = 31702685
RAW_BYTES = N_FULL * DIM * 8

# Rounded centres from the earlier full-file CUDA run (86 rounds). Used only
# to measure how far the stitched centres sit from that result.
REFERENCE = np.array(
    [
        [-0.12239597, 0.09728258, -0.01315756, 1.18514829, 0.15859577, -1.65436867],
        [0.03957855, -0.03145776, 0.00425469, -0.38323526, -0.05128429, 0.53496462],
    ],
    dtype=np.float64,
)


def load_centroids(path, k):
    cents = np.loadtxt(path, dtype=np.float64)
    if cents.ndim == 1:
        cents = cents.reshape(1, -1)
    cents = np.ascontiguousarray(cents[:, :DIM], dtype=np.float64)
    if cents.shape[0] != k:
        raise SystemExit("expected %d centroids, got %d" % (k, cents.shape[0]))
    return cents


def prepare_raw(text_path, raw_path):
    if os.path.isfile(raw_path) and os.path.getsize(raw_path) == RAW_BYTES:
        print("raw exists %s" % raw_path)
        return
    print("reading %s" % text_path)
    sys.stdout.flush()
    t0 = time.perf_counter()
    points = np.loadtxt(text_path, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = np.ascontiguousarray(points[:, :DIM], dtype=np.float64)
    if points.shape != (N_FULL, DIM):
        raise SystemExit("expected %d x %d, got %s" % (N_FULL, DIM, points.shape))
    tmp = raw_path + ".tmp"
    points.tofile(tmp)
    os.replace(tmp, raw_path)
    print("wrote %s rows=%d prepare_seconds=%.3f" % (raw_path, points.shape[0], time.perf_counter() - t0))


def reference_shift(centres):
    """Max distance after matching the two centres either way."""
    a = np.linalg.norm(centres - REFERENCE, axis=1).max()
    swapped = np.linalg.norm(centres[::-1] - REFERENCE, axis=1).max()
    return float(min(a, swapped))


def write_result(path, rows, pieces):
    lines = list(rows)
    for piece in pieces:
        i = piece["piece"]
        lines.append(("piece%d_rows" % i, str(piece["rows"])))
        lines.append(("piece%d_iters" % i, str(piece["iters"])))
        lines.append(("piece%d_shift" % i, "%.8e" % piece["shift"]))
        lines.append(("piece%d_read_seconds" % i, "%.6f" % piece["read_seconds"]))
        lines.append(("piece%d_lloyd_seconds" % i, "%.6f" % piece["lloyd_seconds"]))
        lines.append(("piece%d_counts" % i, ",".join(str(c) for c in piece["counts"])))
        for c, centre in enumerate(piece["centres"]):
            if piece["counts"][c] <= 0:
                continue
            lines.append(("piece%d_centre%d" % (i, c), " ".join("%.8f" % x for x in centre)))
    machine_info.write_result(path, lines)
    print("wrote %s" % path)


def stitch_from_pieces(pieces, init, k, max_iter):
    rows = []
    for piece in sorted(pieces, key=lambda item: item["piece"]):
        for c, centre in enumerate(piece["centres"]):
            if piece["counts"][c] > 0:
                rows.append(centre)
    points = np.asarray(rows, dtype=np.float64)
    if points.shape[0] < k:
        raise SystemExit("only %d local centres to stitch" % points.shape[0])
    cents, counts, iters, shift, seconds = stitch_piece.lloyd_numpy(points, init, k, max_iter, EPS)
    return cents, counts, iters, shift, seconds, int(points.shape[0])


def result_rows(engine, tag, pieces, final, counts, stitch_iters, stitch_shift, stitch_seconds, wall, extra):
    iters = [piece["iters"] for piece in pieces]
    rows = [
        ("engine", engine),
        ("tag", tag),
        ("points", str(sum(piece["rows"] for piece in pieces))),
        ("k", "2"),
        ("pieces", str(len(pieces))),
        ("epsilon", "%.1e" % EPS),
        ("max_iter", "300"),
        ("piece_iters_min", str(min(iters))),
        ("piece_iters_max", str(max(iters))),
        ("local_centres_used", str(sum(1 for p in pieces for c in p["counts"] if c > 0))),
        ("stitch_iters", str(stitch_iters)),
        ("stitch_seconds", "%.6f" % stitch_seconds),
        ("stitch_shift", "%.8e" % stitch_shift),
        ("stitch_counts", ",".join(str(int(c)) for c in counts)),
        ("wall_seconds", "%.6f" % wall),
        ("final_shift_vs_full_file", "%.8e" % reference_shift(final)),
        ("stitch", "unweighted Lloyd on the local centres, same initial centres"),
    ]
    rows.extend(extra)
    for i, centre in enumerate(final):
        rows.append(("centroid%d" % i, " ".join("%.8f" % x for x in centre)))
    return rows


def _lloyd_cuda(cp, kernel, points, centroids, k, max_iter, eps, partial_sums, partial_counts):
    n = int(points.shape[0])
    nblocks = (n + 256 - 1) // 256
    cents = cp.asarray(centroids, dtype=cp.float64)
    shifts = []
    iters = 0
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        iters = it
        partial_sums.fill(0)
        partial_counts.fill(0)
        kernel(
            (nblocks,),
            (256,),
            (points, np.int32(n), cents, np.int32(k), partial_sums, partial_counts),
        )
        sums = partial_sums.sum(axis=0)
        counts = partial_counts.sum(axis=0)
        updated = cents.copy()
        for c in range(k):
            cnt = int(counts[c].item())
            if cnt > 0:
                updated[c] = sums[c] / np.float64(cnt)
        shift = float(cp.sqrt(((updated - cents) ** 2).sum(axis=1)).max().item())
        cents = updated
        shifts.append(shift)
        if shift <= eps:
            break
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(cents), iters, time.perf_counter() - t0, shifts


def run_gpu(raw_path, init, pieces, out_path, max_iter):
    import cupy as cp
    import cuda_kmeans_uk as cuda

    kernel = cp.RawKernel(cuda.LLOYD_SRC, "lloyd_partial")
    warm_n = 256
    kernel(
        (1,),
        (256,),
        (
            cp.zeros((warm_n, DIM), dtype=cp.float64),
            np.int32(warm_n),
            cp.zeros((2, DIM), dtype=cp.float64),
            np.int32(2),
            cp.zeros((1, 2, DIM), dtype=cp.float64),
            cp.zeros((1, 2), dtype=cp.int32),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    nblocks = (max(nrows for _, nrows in pieces) + 255) // 256
    partial_sums = cp.zeros((nblocks, 2, DIM), dtype=cp.float64)
    partial_counts = cp.zeros((nblocks, 2), dtype=cp.int32)

    matrix = np.memmap(raw_path, dtype="<f8", mode="r", shape=(N_FULL, DIM))
    done = []
    h2d = 0.0
    local = 0.0
    for piece, (start, nrows) in enumerate(pieces):
        host = np.ascontiguousarray(matrix[start:start + nrows])
        t0 = time.perf_counter()
        device = cp.asarray(host)
        cp.cuda.Stream.null.synchronize()
        copy_s = time.perf_counter() - t0
        final, iters, lloyd_s, shifts = _lloyd_cuda(
            cp, kernel, device, init, 2, max_iter, EPS, partial_sums, partial_counts
        )
        del device
        counts = _counts_for(host, final)
        shift = shifts[-1] if shifts else float("nan")
        done.append({
            "piece": piece,
            "start": start,
            "rows": nrows,
            "iters": int(iters),
            "shift": float(shift),
            "read_seconds": 0.0,
            "lloyd_seconds": float(lloyd_s),
            "h2d_seconds": float(copy_s),
            "counts": counts,
            "centres": final.tolist(),
        })
        h2d += copy_s
        local += lloyd_s
        print("gpu piece %d rows=%d iters=%d shift=%.8e lloyd=%.3f h2d=%.3f" % (piece, nrows, iters, shift, lloyd_s, copy_s))
        sys.stdout.flush()
    stitched, counts, stitch_iters, stitch_shift, stitch_s, used = stitch_from_pieces(done, init, 2, max_iter)
    wall = local + stitch_s
    rows = result_rows(
        "CUDA_stitch",
        "stitch_gpu_12",
        done,
        stitched,
        counts,
        stitch_iters,
        stitch_shift,
        stitch_s,
        wall,
        [
            ("h2d_seconds", "%.6f" % h2d),
            ("local_lloyd_seconds", "%.6f" % local),
            ("dtype", "float64"),
            ("runtime", "one CUDA piece at a time, then the same stitch"),
            ("data", raw_path),
        ],
    )
    for piece in done:
        rows.append(("piece%d_h2d_seconds" % piece["piece"], "%.6f" % piece["h2d_seconds"]))
    write_result(out_path, rows, done)
    print("gpu wall_seconds=%.3f stitch_seconds=%.3f centres_used=%d" % (wall, stitch_s, used))
    for i, row in enumerate(stitched):
        print("centroid %d: %s" % (i, " ".join("%.8f" % x for x in row)))


def _counts_for(points, centres):
    x2 = np.sum(points * points, axis=1, keepdims=True)
    c2 = np.sum(centres * centres, axis=1)
    labels = np.argmin(x2 + c2 - 2.0 * points.dot(centres.T), axis=1)
    return [int(np.sum(labels == c)) for c in range(centres.shape[0])]


def run_cpu(raw_path, init, pieces, out_path, max_iter, master):
    root = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(master)
        .appName("kmeans-stitch")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.dir", r"C:\cs4480_data\spark-tmp")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    payloads = [
        (raw_path, start, nrows, init.tolist(), 2, max_iter, EPS, piece)
        for piece, (start, nrows) in enumerate(pieces)
    ]
    t0 = time.perf_counter()
    done = spark.sparkContext.parallelize(payloads, len(payloads)).map(stitch_piece.cpu_piece).collect()
    stage = time.perf_counter() - t0
    spark.stop()
    done = sorted(done, key=lambda item: item["piece"])
    for piece in done:
        print(
            "cpu piece %d rows=%d iters=%d shift=%.8e read=%.3f lloyd=%.3f"
            % (piece["piece"], piece["rows"], piece["iters"], piece["shift"], piece["read_seconds"], piece["lloyd_seconds"])
        )
    stitched, counts, stitch_iters, stitch_shift, stitch_s, used = stitch_from_pieces(done, init, 2, max_iter)
    wall = stage + stitch_s
    rows = result_rows(
        "Spark_stitch",
        "stitch_cpu_%s" % master,
        done,
        stitched,
        counts,
        stitch_iters,
        stitch_shift,
        stitch_s,
        wall,
        [
            ("stage_seconds", "%.6f" % stage),
            ("local_lloyd_max_seconds", "%.6f" % max(piece["lloyd_seconds"] for piece in done)),
            ("read_max_seconds", "%.6f" % max(piece["read_seconds"] for piece in done)),
            ("spark_master", master),
            ("runtime", "one Spark job, Lloyd stays inside each piece"),
            ("data", raw_path),
        ],
    )
    write_result(out_path, rows, done)
    print("cpu wall_seconds=%.3f stage_seconds=%.3f stitch_seconds=%.3f centres_used=%d" % (wall, stage, stitch_s, used))
    for i, row in enumerate(stitched):
        print("centroid %d: %s" % (i, " ".join("%.8f" % x for x in row)))


def finish_cpu(pieces_path, init, out_path, max_iter):
    pieces = []
    stage = 0.0
    master = ""
    with open(pieces_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("stage_seconds="):
                stage = float(line.split("=", 1)[1])
            elif line.startswith("spark_master="):
                master = line.split("=", 1)[1]
            elif line.startswith("{"):
                pieces.append(json.loads(line))
    pieces = sorted(pieces, key=lambda item: item["piece"])
    for piece in pieces:
        print(
            "cpu piece %d rows=%d iters=%d shift=%.8e read=%.3f lloyd=%.3f"
            % (piece["piece"], piece["rows"], piece["iters"], piece["shift"], piece["read_seconds"], piece["lloyd_seconds"])
        )
    stitched, counts, stitch_iters, stitch_shift, stitch_s, used = stitch_from_pieces(pieces, init, 2, max_iter)
    wall = stage + stitch_s
    rows = result_rows(
        "Spark_stitch",
        "stitch_cpu_%s" % (master or "local"),
        pieces,
        stitched,
        counts,
        stitch_iters,
        stitch_shift,
        stitch_s,
        wall,
        [
            ("stage_seconds", "%.6f" % stage),
            ("local_lloyd_max_seconds", "%.6f" % max(piece["lloyd_seconds"] for piece in pieces)),
            ("read_max_seconds", "%.6f" % max(piece["read_seconds"] for piece in pieces)),
            ("spark_master", master),
            ("runtime", "one Spark job, Lloyd stays inside each piece"),
        ],
    )
    write_result(out_path, rows, pieces)
    print("cpu wall_seconds=%.3f stage_seconds=%.3f stitch_seconds=%.3f centres_used=%d" % (wall, stage, stitch_s, used))
    for i, row in enumerate(stitched):
        print("centroid %d: %s" % (i, " ".join("%.8f" % x for x in row)))


def main():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "gpu", "cpu", "finish"))
    ap.add_argument("--text", default=r"C:\cs4480_data\parallel\uk_features_scaled_all.txt")
    ap.add_argument("--raw", default=r"C:\cs4480_data\parallel\uk_features_scaled_all.f64")
    ap.add_argument("--centroids", default=r"C:\cs4480_data\parallel\fair\init_centroids_k2.txt")
    ap.add_argument("--pieces", type=int, default=12)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--rows", type=int, default=0, help="smoke test: use only the first N rows")
    ap.add_argument("--master", default="local[12]")
    ap.add_argument("--pieces-file", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.command == "prepare":
        prepare_raw(args.text, args.raw)
        return

    init = load_centroids(args.centroids, 2)
    if args.command == "finish":
        finish_cpu(args.pieces_file, init, args.out, args.max_iter)
        return
    if args.rows:
        smoke = os.path.join(os.path.dirname(args.raw), "stitch_smoke.f64")
        rows = int(args.rows)
        if not os.path.isfile(smoke) or os.path.getsize(smoke) != rows * DIM * 8:
            print("reading first %d rows for the smoke test" % rows)
            points = np.loadtxt(args.text, dtype=np.float64, max_rows=rows)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            np.ascontiguousarray(points[:, :DIM], dtype=np.float64).tofile(smoke)
        pieces = stitch_piece.chunk_ranges(rows, args.pieces)
        # cpu_piece seeks the smoke file. Tell the runners which file.
        args.raw = smoke
        # The GPU path checks N_FULL. Smoke is CPU only.
        run_cpu(args.raw, init, pieces, args.out, args.max_iter, args.master)
        return

    if not os.path.isfile(args.raw) or os.path.getsize(args.raw) != RAW_BYTES:
        raise SystemExit("missing raw matrix %s" % args.raw)
    pieces = stitch_piece.chunk_ranges(N_FULL, args.pieces)
    if args.command == "gpu":
        run_gpu(args.raw, init, pieces, args.out, args.max_iter)
    else:
        run_cpu(args.raw, init, pieces, args.out, args.max_iter, args.master)


if __name__ == "__main__":
    main()
