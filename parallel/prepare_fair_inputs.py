#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the shared k-means inputs used on every machine.

Sampling uses a fixed 64-bit LCG and Vitter's Method S, so the selected row
numbers do not depend on the NumPy version. Run this once, then copy
data/parallel/uk_features_scaled_all.txt and data/parallel/fair/ with the code.
A second machine can run this script again; fair/MANIFEST.txt records the
source checksum so the two copies can be checked before timing.
"""
from __future__ import print_function

import array
import hashlib
import os
import sys

MASK = (1 << 64) - 1


class LCG(object):
    def __init__(self, seed):
        self.state = int(seed) & MASK
        if self.state == 0:
            self.state = 1

    def _next(self):
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & MASK
        return self.state

    def below(self, need, span):
        if span <= 1:
            return True
        return (self._next() % span) < need


def script_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path):
    cfg = {
        "K": "3",
        "MAX_ITER": "300",
        "EPS": "0.0001",
        "SEED": "42",
        "SIZES": "200000 1000000 5000000 10000000 20000000",
    }
    with open(path) as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            value = value.strip()
            if value.startswith("${"):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            cfg[key.strip()] = value
    return cfg


def sample_positions(n, k, rng):
    """Increasing positions in 0..n-1, exactly min(k, n) of them."""
    out = array.array("I")
    if n <= 0 or k <= 0:
        return out
    if k > n:
        k = n
    selected = 0
    i = 0
    while i < n and selected < k:
        need = k - selected
        span = n - i
        if rng.below(need, span):
            out.append(i)
            selected += 1
        i += 1
    if selected != k:
        raise SystemExit("sampler returned %d rows, expected %d" % (selected, k))
    return out


def take_positions(population, k, rng):
    positions = sample_positions(len(population), k, rng)
    out = array.array("I")
    for pos in positions:
        out.append(population[pos])
    return out


def iter_rows(path):
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                if not line.endswith(b"\n"):
                    line += b"\n"
                yield line


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_and_hash(path):
    digest = hashlib.sha256()
    n = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                if not line.endswith(b"\n"):
                    line += b"\n"
                digest.update(line)
                n += 1
    return n, digest.hexdigest()


def write_selected(src, dest, chosen, expected):
    digest = hashlib.sha256()
    pos = 0
    written = 0
    folder = os.path.dirname(dest)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(dest, "wb") as out:
        for i, line in enumerate(iter_rows(src)):
            if pos < len(chosen) and i == chosen[pos]:
                out.write(line)
                digest.update(line)
                written += 1
                pos += 1
    if written != expected:
        raise SystemExit("%s wrote %d rows, expected %d" % (dest, written, expected))
    return digest.hexdigest()


def write_centroids(sample_path, dest, k, rng):
    rows = []
    with open(sample_path, "rb") as handle:
        for line in handle:
            if line.strip():
                if not line.endswith(b"\n"):
                    line += b"\n"
                rows.append(line)
    picked = sample_positions(len(rows), k, rng)
    digest = hashlib.sha256()
    with open(dest, "wb") as out:
        for pos in picked:
            out.write(rows[pos])
            digest.update(rows[pos])
    return len(picked), digest.hexdigest()


def manifest_value(path, key):
    if not os.path.isfile(path):
        return ""
    prefix = key + "="
    with open(path) as handle:
        for line in handle:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return ""


def main():
    root = os.environ.get("HOUSING", script_root())
    cfg = load_config(os.path.join(os.path.dirname(__file__), "fair_config.env"))
    source = os.path.join(root, "data", "parallel", "uk_features_scaled_all.txt")
    out_dir = os.path.join(root, "data", "parallel", "fair")
    manifest = os.path.join(out_dir, "MANIFEST.txt")
    force = "--force" in sys.argv
    k = int(cfg["K"])
    seed = int(cfg["SEED"])
    sizes = [int(x) for x in cfg["SIZES"].split()]
    if not os.path.isfile(source):
        raise SystemExit("Missing %s. Copy uk_features_scaled_all.txt with the project." % source)

    print("Hashing %s" % source)
    sys.stdout.flush()
    n_rows, source_sha = count_and_hash(source)
    print("source_rows=%d" % n_rows)
    if (not force) and manifest_value(manifest, "source_sha256") == source_sha and manifest_value(manifest, "seed") == str(seed) and manifest_value(manifest, "k") == str(k):
        print("fair inputs already match this source file: %s" % manifest)
        return

    sizes = sorted(set(size for size in sizes if size > 0))
    if not sizes:
        raise SystemExit("No sizes listed in fair_config.env")
    largest = sizes[-1]
    if largest > n_rows:
        largest = n_rows
    rng = LCG(seed)
    print("Sampling %d row numbers" % min(largest, n_rows))
    sys.stdout.flush()
    by_size = {}
    largest_idx = sample_positions(n_rows, min(largest, n_rows), rng)
    by_size[len(largest_idx)] = largest_idx
    for size in sizes:
        if size == len(largest_idx):
            continue
        if size >= n_rows:
            continue
        print("Sampling %d from the %d-row set" % (size, len(largest_idx)))
        by_size[size] = take_positions(largest_idx, size, rng)

    lines = []
    lines.append("source=%s" % source)
    lines.append("source_rows=%d" % n_rows)
    lines.append("source_sha256=%s" % source_sha)
    lines.append("seed=%d" % seed)
    lines.append("sampler=lcg64_method_s")
    lines.append("k=%d" % k)
    lines.append("epsilon=%s" % cfg["EPS"])
    lines.append("max_iter=%s" % cfg["MAX_ITER"])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    smallest_path = None
    for size in sizes:
        if size >= n_rows:
            lines.append("n%d=use_full_file" % size)
            continue
        chosen = by_size[size]
        dest = os.path.join(out_dir, "n%d.txt" % size)
        print("Writing %s" % dest)
        sys.stdout.flush()
        digest = write_selected(source, dest, chosen, size)
        lines.append("n%d_rows=%d" % (size, size))
        lines.append("n%d_sha256=%s" % (size, digest))
        if smallest_path is None:
            smallest_path = dest

    if smallest_path is None:
        raise SystemExit("No subset was written.")
    cent_path = os.path.join(out_dir, "init_centroids_k%d.txt" % k)
    n_cent, cent_sha = write_centroids(smallest_path, cent_path, k, rng)
    lines.append("centroids=%s" % cent_path)
    lines.append("centroids_rows=%d" % n_cent)
    lines.append("centroids_sha256=%s" % cent_sha)
    lines.append("all_rows_file=%s" % source)
    with open(manifest, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("Wrote %s" % manifest)
    print("Copy data/parallel/fair/ and uk_features_scaled_all.txt to the other machine.")


if __name__ == "__main__":
    main()
