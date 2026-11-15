# -*- coding: utf-8 -*-
"""One MapReduce job: each map task clusters one piece, one reducer stitches."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import machine_info
import stitch_piece

N_FULL = 31702685
PIECES = 12
TEXT = Path(r"C:\cs4480_data\parallel\uk_features_scaled_all.txt")
CENT = Path(r"C:\cs4480_data\parallel\fair\init_centroids_k2.txt")
PARTS = Path(r"C:\cs4480_data\parallel\stitch_parts")
SHIP = Path(r"C:\cs4480_data\mr_kmeans\stitch_ship")
JAR = Path(r"C:\cs4480_tools\hadoop\hadoop-3.3.6\share\hadoop\tools\lib\hadoop-streaming-3.3.6.jar")
OUT = Path(r"C:\cs4480_data\processed\fair_timing\goodpc\stitch_mapreduce.txt")
HDFS_IN = "/user/jkjk2/uk_kmeans_stitch_in"
HDFS_OUT = "/user/jkjk2/uk_kmeans_stitch_out"


def run(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, check=False)


def hdfs(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run("hdfs dfs " + " ".join(args))


def split_text() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    ranges = stitch_piece.chunk_ranges(N_FULL, PIECES)
    expected = [size for _, size in ranges]
    ready = True
    for i, size in enumerate(expected):
        path = PARTS / ("part-%02d.txt" % i)
        if not path.is_file():
            ready = False
            break
        with path.open("rb") as handle:
            lines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))
        if lines != size:
            ready = False
            break
    if ready:
        print("parts already split", flush=True)
        return
    handles = [(PARTS / ("part-%02d.txt" % i)).open("wb") for i in range(PIECES)]
    piece = 0
    left = expected[0]
    seen = 0
    try:
        with TEXT.open("rb") as src:
            for line in src:
                if not line.strip():
                    continue
                handles[piece].write(line if line.endswith(b"\n") else line + b"\n")
                seen += 1
                left -= 1
                if left == 0 and piece + 1 < PIECES:
                    piece += 1
                    left = expected[piece]
    finally:
        for handle in handles:
            handle.close()
    if seen != N_FULL:
        raise SystemExit("split saw %d rows, expected %d" % (seen, N_FULL))
    print("split %d rows into %d parts" % (seen, PIECES), flush=True)


def main() -> None:
    if not JAR.is_file():
        raise SystemExit("missing streaming jar")
    split_text()
    SHIP.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CENT, SHIP / "centroids.txt")
    for name in ("stitch_mapper.py", "stitch_reducer.py"):
        shutil.copyfile(ROOT / "mr_kmeans" / name, SHIP / name)
    shutil.copyfile(ROOT / "stitch_piece.py", SHIP / "stitch_piece.py")

    hdfs(["-rm", "-r", "-f", HDFS_IN, HDFS_OUT])
    hdfs(["-mkdir", "-p", HDFS_IN])
    put0 = time.perf_counter()
    put = hdfs(["-put", "-f", '"' + str(PARTS).replace("\\", "/") + '"/*', HDFS_IN])
    put_s = time.perf_counter() - put0
    if put.returncode != 0:
        raise SystemExit(put.stderr or put.stdout or "hdfs put failed")
    print("hdfs_put_seconds=%.3f" % put_s, flush=True)

    bat = Path(r"C:\cs4480_data\logs\kmeans_stitch_mr.cmd")
    bat.write_text(
        "@echo off\r\n"
        'cd /d "%s"\r\n' % SHIP
        + 'hadoop jar "%s" ' % JAR
        + "-D mapreduce.job.name=uk_kmeans_stitch "
        + "-D mapreduce.map.memory.mb=2048 "
        + "-D mapreduce.reduce.memory.mb=2048 "
        + "-D mapreduce.input.fileinputformat.split.minsize=536870912 "
        + "-D mapreduce.map.speculative=false "
        + "-D mapreduce.reduce.speculative=false "
        + "-D mapreduce.task.timeout=0 "
        + '-files "centroids.txt,stitch_mapper.py,stitch_reducer.py,stitch_piece.py" '
        + '-mapper "python stitch_mapper.py" '
        + '-reducer "python stitch_reducer.py" '
        + "-input %s -output %s -numReduceTasks 1\r\n" % (HDFS_IN, HDFS_OUT),
        encoding="ascii",
    )
    t0 = time.perf_counter()
    proc = run('cmd /c "%s"' % bat)
    wall = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = Path(r"C:\cs4480_data\logs\kmeans_stitch_mr.log")
    log_path.write_text(combined, encoding="utf-8", errors="replace")
    if "completed successfully" not in combined:
        raise SystemExit("streaming failed\n" + combined[-2500:])
    cat = hdfs(["-cat", HDFS_OUT + "/part-*"])
    if cat.returncode != 0:
        raise SystemExit(cat.stderr or cat.stdout or "hdfs cat failed")
    payload = ""
    for line in cat.stdout.splitlines():
        if line.startswith("result\t"):
            payload = line.split("\t", 1)[1]
    if not payload:
        raise SystemExit("reducer wrote no result\n" + cat.stdout[:1000])
    result = json.loads(payload)
    pieces = sorted(result["pieces"], key=lambda item: item["piece"])
    centres = result["centres"]
    rows = [
        ("engine", "MapReduce_stitch"),
        ("tag", "stitch_mapreduce_12"),
        ("points", str(sum(piece["rows"] for piece in pieces))),
        ("k", "2"),
        ("pieces", str(len(pieces))),
        ("epsilon", "1.0e-04"),
        ("max_iter", "300"),
        ("piece_iters_min", str(min(piece["iters"] for piece in pieces))),
        ("piece_iters_max", str(max(piece["iters"] for piece in pieces))),
        ("local_centres_used", str(result["local_centres_used"])),
        ("stitch_iters", str(result["stitch_iters"])),
        ("stitch_seconds", "%.6f" % result["stitch_seconds"]),
        ("stitch_shift", "%.8e" % result["stitch_shift"]),
        ("stitch_counts", ",".join(str(c) for c in result["stitch_counts"])),
        ("wall_seconds", "%.6f" % wall),
        ("hdfs_put_seconds", "%.6f" % put_s),
        ("map_memory_mb", "2048"),
        ("reducers", "1"),
        ("yarn_vcores", "8"),
        ("runtime", "one MapReduce job, Lloyd stays inside each piece"),
        ("data", str(TEXT)),
        ("centroids", str(CENT)),
    ]
    for i, centre in enumerate(centres):
        rows.append(("centroid%d" % i, " ".join("%.8f" % x for x in centre)))
    for piece in pieces:
        i = piece["piece"]
        rows.append(("piece%d_rows" % i, str(piece["rows"])))
        rows.append(("piece%d_iters" % i, str(piece["iters"])))
        rows.append(("piece%d_shift" % i, "%.8e" % piece["shift"]))
        rows.append(("piece%d_lloyd_seconds" % i, "%.6f" % piece["lloyd_seconds"]))
        rows.append(("piece%d_counts" % i, ",".join(str(c) for c in piece["counts"])))
    machine_info.write_result(OUT, rows)
    print("wall_seconds=%.3f hdfs_put_seconds=%.3f" % (wall, put_s), flush=True)
    for i, centre in enumerate(centres):
        print("centroid %d: %s" % (i, " ".join("%.8f" % x for x in centre)), flush=True)
    print("wrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
