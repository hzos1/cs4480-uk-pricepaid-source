"""Time Hadoop Streaming Count Distribution on the existing basket files.

The candidate rule matches local_python/apriori_cd_benchmark.py.
A size is recorded only after its frequent itemsets match the Python run.
"""
from __future__ import annotations

import csv
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASKETS = ROOT / "data" / "parallel" / "apriori_baskets"
OUT = ROOT / "data" / "processed" / "apriori_timing" / "goodpc"
REPORT = ROOT / "reports" / "apriori_count_distribution_timing.csv"
WORK = Path(r"C:\cs4480_data\mr_apriori")
HADOOP_HOME = Path(r"C:\cs4480_tools\hadoop\hadoop-3.3.6")
JAVA_HOME = Path(r"C:\cs4480_tools\jdk")
STREAMING = HADOOP_HOME / "share" / "hadoop" / "tools" / "lib" / "hadoop-streaming-3.3.6.jar"
MIN_SUPPORT = 0.05
MAX_K = 3
N_ITEMS = 11
TAGS = [
    ("n200000", 200000),
    ("n1000000", 1000000),
    ("n5000000", 5000000),
    ("n10000000", 10000000),
    ("n20000000", 20000000),
    ("all", 31702685),
]


def generate_candidates(prev: list[int], k: int) -> list[int]:
    prev_set = set(prev)
    found: set[int] = set()
    out: list[int] = []
    for i, left in enumerate(prev):
        for right in prev[i + 1 :]:
            union = left | right
            if union.bit_count() != k or union in found:
                continue
            bits = [1 << j for j in range(N_ITEMS) if union & (1 << j)]
            if all((union ^ bit) in prev_set for bit in bits):
                found.add(union)
                out.append(union)
    return out


def env() -> dict[str, str]:
    path = os.environ.get("PATH", "")
    merged = str(HADOOP_HOME / "bin") + ";" + str(JAVA_HOME / "bin") + ";" + path
    out = dict(os.environ)
    out["JAVA_HOME"] = str(JAVA_HOME)
    out["HADOOP_HOME"] = str(HADOOP_HOME)
    out["HADOOP_CONF_DIR"] = str(HADOOP_HOME / "etc" / "hadoop")
    out["PATH"] = merged
    return out


def run(cmd: list[str] | str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env(),
        text=True,
        capture_output=True,
        check=False,
        shell=True,
    )


def hdfs(args: list[str]) -> None:
    proc = run(["hdfs", "dfs", *args])
    if proc.returncode != 0 and "No such file" not in (proc.stderr or "") and "No such file" not in (proc.stdout or ""):
        # rm -f of a missing path is not fatal
        if args[:2] == ["-rm", "-r"] or (args and args[0] == "-rm"):
            return
        raise SystemExit(proc.stderr or proc.stdout or "hdfs failed")


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def read_freq(path: Path) -> dict[int, int]:
    freq: dict[int, int] = {}
    if not path.is_file():
        return freq
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("\t")
        if key:
            freq[int(key)] = int(value)
    return freq


def streaming_round(tag: str, k: int, hdfs_in: str, hdfs_out: str) -> tuple[dict[int, int], float, int]:
    hdfs(["-rm", "-r", "-f", hdfs_out])
    cmd = (
        f'hadoop jar "{STREAMING}" '
        f"-D mapreduce.job.name=apriori_{tag}_k{k} "
        "-D mapreduce.map.speculative=false "
        "-D mapreduce.reduce.speculative=false "
        "-D mapreduce.task.timeout=1800000 "
        '-files "mapper.py,reducer.py,candidates.txt" '
        '-mapper "python mapper.py" '
        '-reducer "python reducer.py" '
        f"-input {hdfs_in} -output {hdfs_out} -numReduceTasks 1"
    )
    t0 = time.perf_counter()
    proc = run(cmd, cwd=WORK)
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = OUT / f"mapreduce_{tag}_k{k}.log"
    log_path.write_text(combined, encoding="utf-8")
    if "completed successfully" not in combined:
        tail = combined[-2000:]
        raise SystemExit(f"Streaming failed for {tag} k={k}\n{tail}")
    maps = 0
    for line in combined.splitlines():
        if "Launched map tasks=" in line:
            maps = int(line.split("=", 1)[1].strip())
    cat = run(["hdfs", "dfs", "-cat", hdfs_out + "/part-*"])
    if cat.returncode != 0:
        raise SystemExit(cat.stderr or "hdfs cat failed")
    counts: dict[int, int] = {}
    for line in cat.stdout.splitlines():
        key, _, value = line.partition("\t")
        if key:
            counts[int(key)] = counts.get(int(key), 0) + int(value)
    return counts, elapsed, maps


def one(tag: str, rows_drawn: int) -> dict:
    local = BASKETS / f"{tag}.txt"
    if not local.is_file():
        raise SystemExit(f"Missing baskets {local}")
    baskets = count_lines(local)
    min_count = max(1, math.ceil(MIN_SUPPORT * baskets))
    hdfs_in = f"/user/jkjk2/apriori_{tag}"
    print(f"MapReduce {tag} baskets={baskets} min_count={min_count}", flush=True)
    t_put = time.perf_counter()
    hdfs(["-rm", "-r", "-f", hdfs_in])
    hdfs(["-mkdir", "-p", hdfs_in])
    put = run(["hdfs", "dfs", "-put", "-f", str(local), hdfs_in + "/baskets.txt"])
    put_seconds = time.perf_counter() - t_put
    if put.returncode != 0:
        raise SystemExit(put.stderr or "hdfs put failed")
    prev = [1 << i for i in range(N_ITEMS)]
    frequent: dict[int, int] = {}
    rounds = 0
    counted = 0
    job_seconds = 0.0
    maps = 0
    for k in range(1, MAX_K + 1):
        cands = prev if k == 1 else generate_candidates(prev, k)
        if not cands:
            break
        (WORK / "candidates.txt").write_text(
            "".join(f"{c}\n" for c in cands), encoding="utf-8"
        )
        counts, elapsed, maps = streaming_round(tag, k, hdfs_in, f"/user/jkjk2/apriori_out_{tag}_k{k}")
        job_seconds += elapsed
        rounds += 1
        counted += len(cands)
        prev = []
        for cand in cands:
            count = counts.get(cand, 0)
            if count >= min_count:
                frequent[cand] = count
                prev.append(cand)
        print(f"  k={k} candidates={len(cands)} frequent={len(prev)} job={elapsed:.1f}s", flush=True)
        if not prev:
            break
    expected = read_freq(OUT / f"python_{tag}_freq.txt")
    if frequent != expected:
        missing = sorted(set(expected) - set(frequent))[:8]
        extra = sorted(set(frequent) - set(expected))[:8]
        raise SystemExit(
            f"{tag} frequent itemsets do not match Python. "
            f"mr={len(frequent)} python={len(expected)} missing={missing} extra={extra}"
        )
    host = socket.gethostname()
    rows = [
        ("engine", "MapReduce"),
        ("algorithm", "Apriori-Count-Distribution"),
        ("status", "ok"),
        ("tag", tag),
        ("rows_drawn", str(rows_drawn)),
        ("baskets", str(baskets)),
        ("wall_seconds", f"{job_seconds:.3f}"),
        ("load_seconds", f"{put_seconds:.3f}"),
        ("count_seconds", f"{job_seconds:.3f}"),
        ("rounds", str(rounds)),
        ("candidates_counted", str(counted)),
        ("min_count", str(min_count)),
        ("min_support", str(MIN_SUPPORT)),
        ("max_itemset", str(MAX_K)),
        ("workers", str(maps)),
        ("spark_master", "yarn"),
        ("host", host),
        ("note", "YARN Streaming; wall is the sum of the count jobs, not the HDFS put"),
    ]
    text = "".join(f"{key}={value}\n" for key, value in rows)
    (OUT / f"mapreduce_{tag}.txt").write_text(text, encoding="utf-8")
    freq_lines = [f"{mask}\t{frequent[mask]}" for mask in sorted(frequent)]
    (OUT / f"mapreduce_{tag}_freq.txt").write_text("\n".join(freq_lines) + "\n", encoding="utf-8")
    print(f"MapReduce {tag} wall={job_seconds:.3f}s put={put_seconds:.3f}s maps={maps}", flush=True)
    return {
        "engine": "MapReduce",
        "tag": tag,
        "rows_drawn": rows_drawn,
        "baskets": baskets,
        "wall_seconds": round(job_seconds, 3),
        "load_seconds": round(put_seconds, 3),
        "count_seconds": round(job_seconds, 3),
        "process_seconds": "",
        "rounds": rounds,
        "workers": maps,
        "spark_master": "yarn",
        "status": "ok",
        "note": "YARN Streaming; wall is the sum of the count jobs, not the HDFS put; frequent itemsets match Python",
        "host": host,
        "cpu_model": "AMD Ryzen 5 5600X 6-Core Processor",
        "cpu_cores": 6,
        "cpu_threads": 12,
        "mem_total_gb": 31.889,
    }


def publish(rows: list[dict]) -> None:
    existing: list[dict] = []
    if REPORT.is_file():
        with REPORT.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    # Replace only the tags this run measured. Leave other MapReduce rows in place.
    done_tags = {row.get("tag") for row in rows}
    existing = [
        row
        for row in existing
        if not (row.get("engine") == "MapReduce" and row.get("tag") in done_tags)
    ]
    existing = [row for row in existing if not (row.get("engine") == "MapReduce" and row.get("status") != "ok")]
    merged = existing + rows
    fields = [
        "engine",
        "tag",
        "rows_drawn",
        "baskets",
        "wall_seconds",
        "load_seconds",
        "count_seconds",
        "process_seconds",
        "rounds",
        "workers",
        "spark_master",
        "status",
        "note",
        "host",
        "cpu_model",
        "cpu_cores",
        "cpu_threads",
        "mem_total_gb",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in merged:
            writer.writerow(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in merged:
            writer.writerow(row)
    sys.path.insert(0, str(ROOT / "local_python"))
    import pandas as pd
    import apriori_cd_benchmark as bench

    bench.plot_timing(pd.DataFrame(merged))


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("mr_apriori_mapper.py", "mr_apriori_reducer.py"):
        src = ROOT / "parallel" / name
        dest_name = "mapper.py" if "mapper" in name else "reducer.py"
        (WORK / dest_name).write_bytes(src.read_bytes())
    done = []
    only = sys.argv[1:]
    for tag, rows_drawn in TAGS:
        if only and tag not in only:
            continue
        done.append(one(tag, rows_drawn))
        publish(done if not only else done)
    if not only:
        publish(done)
    print("MapReduce Apriori finished", flush=True)


if __name__ == "__main__":
    main()
