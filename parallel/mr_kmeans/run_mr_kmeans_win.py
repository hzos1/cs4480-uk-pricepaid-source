"""Windows driver for the same Streaming Lloyd loop as run_mr_kmeans.sh.

Git Bash cannot pass Hadoop Streaming's comma-separated -files list through
hadoop.cmd. This process calls hadoop.cmd the same way the Apriori jobs did.
The mapper, reducer, and centroid update are unchanged.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def win_path(path: str) -> str:
    if len(path) >= 3 and path[0] == "/" and path[2] == "/" and path[1].isalpha():
        return path[1].upper() + ":" + path[2:].replace("/", "\\")
    return path


def run(cmd: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )


def hdfs(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run("hdfs dfs " + " ".join(args))


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing environment variable {name}")
    return value


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def main() -> None:
    data = Path(win_path(require("DATA")))
    init = Path(win_path(require("INIT")))
    work = Path(win_path(os.environ.get("WORKDIR", "")))
    if not (work / "mapper.py").is_file():
        # Git Bash can drop WORKDIR when the project path contains a space.
        work = Path(__file__).resolve().parent
    k = int(require("K"))
    max_iter = int(require("MAX_ITER"))
    eps = require("EPS")
    tag = os.environ.get("KMEANS_TAG", "run")
    # Git Bash rewrites /user/... into a Program Files path before Python starts.
    hdfs_in = f"/user/jkjk2/uk_kmeans_in_{tag}"
    hdfs_out_base = f"/user/jkjk2/uk_kmeans_out_{tag}"
    out_file = Path(win_path(require("OUT_FILE")))
    jar = win_path(require("STREAMING_JAR"))
    housing = os.environ.get("HOUSING", "")
    if not Path(jar).is_file():
        raise SystemExit(f"Streaming jar not found: {jar}")
    if not data.is_file():
        raise SystemExit(f"Missing data {data}")
    if not init.is_file():
        raise SystemExit(f"Missing centroids {init}")

    # Hadoop ships -files as a URI, which rejects the space in "4480 group".
    runtime = Path(r"C:\cs4480_data\mr_kmeans")
    ship = runtime / "ship"
    ship.mkdir(parents=True, exist_ok=True)
    (runtime / "logs").mkdir(parents=True, exist_ok=True)
    (runtime / "centroids").mkdir(parents=True, exist_ok=True)
    centroids = runtime / "centroids" / "centroids.txt"
    shutil.copyfile(init, centroids)
    shutil.copyfile(work / "mapper.py", ship / "mapper.py")
    shutil.copyfile(work / "reducer.py", ship / "reducer.py")

    hdfs(["-rm", "-r", "-f", hdfs_in])
    hdfs(["-mkdir", "-p", hdfs_in])
    local = str(data).replace("\\", "/")
    put0 = time.perf_counter()
    put = hdfs(["-put", "-f", f'"{local}"', f"{hdfs_in}/points.txt"])
    put1 = time.perf_counter()
    if put.returncode != 0:
        raise SystemExit(put.stderr or put.stdout or "hdfs put failed")
    print(f"Streaming jar: {jar}", flush=True)
    print(f"Data: {data}", flush=True)

    start = time.perf_counter()
    iters = 0
    moved = "1"
    while iters < max_iter and moved == "1":
        iters += 1
        hdfs_out = f"{hdfs_out_base}/iter_{iters}"
        hdfs(["-rm", "-r", "-f", hdfs_out])
        shutil.copyfile(centroids, ship / "centroids.txt")
        # A cmd script keeps the comma-separated -files list and the /user path intact.
        bat = Path(r"C:\cs4480_data\logs\kmeans_iter.cmd")
        bat.write_text(
            "@echo off\r\n"
            f'cd /d "{ship}"\r\n'
            f'hadoop jar "{jar}" '
            f"-D mapreduce.job.name=uk_kmeans_{tag}_iter_{iters} "
            "-D mapreduce.map.speculative=false "
            "-D mapreduce.reduce.speculative=false "
            "-D mapreduce.task.timeout=0 "
            '-files "centroids.txt,mapper.py,reducer.py" '
            '-mapper "python mapper.py" '
            '-reducer "python reducer.py" '
            f"-input {hdfs_in} -output {hdfs_out} -numReduceTasks {k}\r\n",
            encoding="ascii",
        )
        t0 = time.perf_counter()
        proc = run(f'cmd /c "{bat}"')
        elapsed = time.perf_counter() - t0
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path = runtime / "logs" / f"{tag}_iter_{iters}.log"
        log_path.write_text(combined, encoding="utf-8", errors="replace")
        if "completed successfully" not in combined:
            raise SystemExit(f"Streaming failed for {tag} iter {iters}\n{combined[-2000:]}")
        cat = hdfs(["-cat", f"{hdfs_out}/part-*"])
        if cat.returncode != 0:
            raise SystemExit(cat.stderr or cat.stdout or "hdfs cat failed")
        raw = runtime / "centroids" / "new_centroids_raw.txt"
        raw.write_text(cat.stdout, encoding="utf-8")
        upd = run(
            "python "
            f'"{work / "update_centroids.py"}" '
            f'"{centroids}" "{raw}" "{runtime / "centroids" / "centroids_next.txt"}" '
            f"{eps} {k}"
        )
        if upd.returncode != 0:
            raise SystemExit(upd.stderr or upd.stdout or "centroid update failed")
        shift = centroids.with_suffix(centroids.suffix + ".shift").read_text(encoding="utf-8").strip()
        moved = centroids.with_suffix(centroids.suffix + ".moved").read_text(encoding="utf-8").strip()
        shutil.move(str(runtime / "centroids" / "centroids_next.txt"), str(centroids))
        print(f"Iter {iters} max shift = {shift} moved={moved} job={elapsed:.1f}s", flush=True)
    end = time.perf_counter()
    points = count_lines(data)
    wall = end - start
    put_s = put1 - put0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "engine=Hadoop_Streaming\n"
        f"tag={tag}\n"
        f"points={points}\n"
        f"k={k}\n"
        f"epsilon={eps}\n"
        f"max_iter={max_iter}\n"
        f"iters={iters}\n"
        f"reducers={k}\n"
        f"wall_seconds={wall:.6f}\n"
        f"hdfs_put_seconds={put_s:.6f}\n"
        f"data={data}\n"
        f"centroids={init}\n"
    )
    out_file.write_text(text, encoding="utf-8")
    print(f"=== MapReduce k-means done ===", flush=True)
    print(f"iters={iters} wall_seconds={wall:.3f}", flush=True)
    print("Final centroids:", flush=True)
    print(centroids.read_text(encoding="utf-8"), flush=True)
    print(f"wrote {out_file}", flush=True)
    info = Path(win_path(housing)) / "parallel" / "machine_info.py" if housing else None
    if info and info.is_file():
        subprocess.run([sys.executable, str(info), "--append", str(out_file)], check=False)


if __name__ == "__main__":
    main()
