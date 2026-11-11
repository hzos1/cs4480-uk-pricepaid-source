#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CUDA Lloyd k-means on one NVIDIA GPU.

Same update as Python, Spark, and Hadoop Streaming: assign each sale to the
nearest centre, replace the centre by sum/count, stop when the largest centre
movement is <= 1e-4. max_iter is only a cap.

wall_seconds is the Lloyd loop after the matrix and the initial centres are
on the device. load_seconds is the host read. h2d_seconds is the copy onto
the GPU and is not included in wall_seconds, matching the separate load time
recorded for the other engines.
"""
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machine_info

DIM = 6
EPS = 1e-4
BLOCK = 256

LLOYD_SRC = r"""
extern "C" __global__
void lloyd_partial(const double* points, int n, const double* cents, int k,
                   double* partial_sums, int* partial_counts) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double sh_sum[48];
    __shared__ int sh_cnt[8];
    if (tid < k) {
        sh_cnt[tid] = 0;
        for (int d = 0; d < 6; d++) sh_sum[tid * 6 + d] = 0.0;
    }
    __syncthreads();
    if (i < n) {
        int best = 0;
        double bestD = 1.0e300;
        for (int c = 0; c < k; c++) {
            double dist = 0.0;
            for (int d = 0; d < 6; d++) {
                double diff = points[i * 6 + d] - cents[c * 6 + d];
                dist += diff * diff;
            }
            if (dist < bestD) {
                bestD = dist;
                best = c;
            }
        }
        atomicAdd(&sh_cnt[best], 1);
        for (int d = 0; d < 6; d++) {
            atomicAdd(&sh_sum[best * 6 + d], points[i * 6 + d]);
        }
    }
    __syncthreads();
    if (tid < k) {
        partial_counts[blockIdx.x * k + tid] = sh_cnt[tid];
        for (int d = 0; d < 6; d++) {
            partial_sums[(blockIdx.x * k + tid) * 6 + d] = sh_sum[tid * 6 + d];
        }
    }
}
"""


def load_matrix(path):
    return np.loadtxt(path, dtype=np.float64)


def gpu_name(cp):
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    return str(name).strip()


def run_kmeans(cp, points, centroids, k, max_iter, eps):
    kernel = cp.RawKernel(LLOYD_SRC, "lloyd_partial")
    n = int(points.shape[0])
    nblocks = (n + BLOCK - 1) // BLOCK
    partial_sums = cp.zeros((nblocks, k, DIM), dtype=cp.float64)
    partial_counts = cp.zeros((nblocks, k), dtype=cp.int32)
    cents = cp.asarray(centroids, dtype=cp.float64)
    shifts = []
    iters = 0
    # One launch outside the clock. The first call compiles the kernel.
    warm_n = BLOCK
    warm_points = cp.zeros((warm_n, DIM), dtype=cp.float64)
    warm_cents = cp.zeros((k, DIM), dtype=cp.float64)
    warm_sums = cp.zeros((1, k, DIM), dtype=cp.float64)
    warm_counts = cp.zeros((1, k), dtype=cp.int32)
    kernel(
        (1,),
        (BLOCK,),
        (
            warm_points,
            np.int32(warm_n),
            warm_cents,
            np.int32(k),
            warm_sums,
            warm_counts,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        iters = it
        partial_sums.fill(0)
        partial_counts.fill(0)
        kernel(
            (nblocks,),
            (BLOCK,),
            (
                points,
                np.int32(n),
                cents,
                np.int32(k),
                partial_sums,
                partial_counts,
            ),
        )
        sums = partial_sums.sum(axis=0)
        counts = partial_counts.sum(axis=0)
        updated = cents.copy()
        for c in range(k):
            cnt = int(counts[c].item())
            if cnt > 0:
                updated[c] = sums[c] / np.float64(cnt)
        diff = updated - cents
        shift = float(cp.sqrt((diff * diff).sum(axis=1)).max().item())
        cents = updated
        shifts.append(shift)
        if shift <= eps:
            break
    cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0
    final = cp.asnumpy(cents)
    return final, iters, wall, shifts


def main():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_housing = os.environ.get("HOUSING", here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--centroids",
        default=os.path.join(default_housing, "data", "parallel", "fair", "init_centroids_k2.txt"),
    )
    ap.add_argument("--k", type=int, default=int(os.environ.get("KMEANS_K", "2")))
    ap.add_argument("--max-iter", type=int, default=int(os.environ.get("KMEANS_MAX_ITER", "300")))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.isfile(args.out):
        with open(args.out, "r") as handle:
            if "wall_seconds=" in handle.read():
                print("skip existing %s" % args.out)
                return

    import cupy as cp

    if not os.path.isfile(args.data) or not os.path.isfile(args.centroids):
        raise SystemExit("Missing input data=%s centroids=%s" % (args.data, args.centroids))

    name = gpu_name(cp)
    print("engine=CUDA Lloyd")
    print("gpu=%s" % name)
    print("data=%s" % args.data)
    print("k=%d max_iter=%d block=%d" % (args.k, args.max_iter, BLOCK))
    sys.stdout.flush()

    t_load0 = time.perf_counter()
    points = load_matrix(args.data)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = np.ascontiguousarray(points[:, :DIM], dtype=np.float64)
    centroids = load_matrix(args.centroids)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    centroids = np.ascontiguousarray(centroids[:, :DIM], dtype=np.float64)
    if centroids.shape[0] != args.k:
        raise SystemExit("expected %d centroids, got %d" % (args.k, centroids.shape[0]))
    t_load1 = time.perf_counter()
    print("points=%d load_seconds=%.3f" % (points.shape[0], t_load1 - t_load0))
    sys.stdout.flush()

    t_h2d0 = time.perf_counter()
    device_points = cp.asarray(points)
    cp.cuda.Stream.null.synchronize()
    h2d = time.perf_counter() - t_h2d0

    final, iters, wall, shifts = run_kmeans(
        cp, device_points, centroids, args.k, args.max_iter, EPS
    )
    final_shift = shifts[-1] if shifts else float("nan")
    print("=== CUDA k-means done ===")
    print("iters=%d wall_seconds=%.6f h2d_seconds=%.6f final_shift=%.8e" % (iters, wall, h2d, final_shift))
    for i, row in enumerate(final):
        print("Centroid %d: %s" % (i, " ".join("%.8f" % x for x in row)))
    sys.stdout.flush()

    props = cp.cuda.runtime.getDeviceProperties(0)
    rows = [
        ("engine", "CUDA_Lloyd"),
        ("tag", args.tag),
        ("points", str(points.shape[0])),
        ("k", str(args.k)),
        ("epsilon", "%.1e" % EPS),
        ("max_iter", str(args.max_iter)),
        ("iters", str(iters)),
        ("wall_seconds", "%.6f" % wall),
        ("h2d_seconds", "%.6f" % h2d),
        ("load_seconds", "%.6f" % (t_load1 - t_load0)),
        ("final_shift", "%.8e" % final_shift),
        ("gpu_name", name),
        ("gpu_memory_bytes", str(int(props["totalGlobalMem"]))),
        ("dtype", "float64"),
        ("block_size", str(BLOCK)),
        ("runtime", "CUDA SIMD threadIdx/blockIdx"),
        ("data", args.data),
        ("centroids", args.centroids),
    ]
    machine_info.write_result(args.out, rows)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
