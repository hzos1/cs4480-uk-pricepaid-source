"""Merge measured Python and Spark Apriori times. Does not invent MapReduce seconds."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_python"))
from apriori_cd_benchmark import OUT, REPORT, plot_timing  # noqa: E402

TAGS = ["n200000", "n1000000", "n5000000", "n10000000", "n20000000", "all"]
DRAWN = {
    "n200000": 200000,
    "n1000000": 1000000,
    "n5000000": 5000000,
    "n10000000": 10000000,
    "n20000000": 20000000,
    "all": 31702685,
}


def parse_result(path: Path) -> dict[str, str]:
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("freq=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key] = value
    return meta


def main() -> None:
    rows = []
    for tag in TAGS:
        py = parse_result(OUT / f"python_{tag}.txt")
        sp = parse_result(OUT / f"spark_{tag}.txt")
        py_freq = {
            line.split("\t")[0]: line.split("\t")[1]
            for line in (OUT / f"python_{tag}_freq.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        sp_freq = {}
        for line in (OUT / f"spark_{tag}.txt").read_text(encoding="utf-8").splitlines():
            if line.startswith("freq="):
                key, value = line[5:].split("\t")
                sp_freq[key] = value
        if py_freq != sp_freq:
            raise SystemExit(f"Spark counts do not match Python for {tag}")
        rows.append(
            {
                "engine": "Python",
                "tag": tag,
                "rows_drawn": int(py["rows_drawn"]),
                "baskets": int(py["baskets"]),
                "wall_seconds": round(float(py["wall_seconds"]), 3),
                "load_seconds": round(float(py["load_seconds"]), 3),
                "count_seconds": round(float(py["count_seconds"]), 3),
                "process_seconds": round(float(py["wall_seconds"]), 3),
                "rounds": int(py["rounds"]),
                "workers": int(py["workers"]),
                "spark_master": "",
                "status": "ok",
                "note": "6 processes; wall is reread of the basket file plus three count rounds",
                "host": "goodpc",
                "cpu_model": "AMD Ryzen 5 5600X 6-Core Processor",
                "cpu_cores": 6,
                "cpu_threads": 12,
                "mem_total_gb": 31.889,
            }
        )
        rows.append(
            {
                "engine": "Spark",
                "tag": tag,
                "rows_drawn": DRAWN[tag],
                "baskets": int(sp["baskets"]),
                "wall_seconds": round(float(sp["wall_seconds"]), 3),
                "load_seconds": round(float(sp["load_seconds"]), 3),
                "count_seconds": round(float(sp["count_seconds"]), 3),
                "process_seconds": round(float(sp["process_seconds"]), 3),
                "rounds": int(sp["rounds"]),
                "workers": int(sp["workers"]),
                "spark_master": sp["spark_master"],
                "status": "ok",
                "note": "local[*], baskets cached between rounds; process_seconds includes JVM startup; not YARN",
                "host": "goodpc",
                "cpu_model": "AMD Ryzen 5 5600X 6-Core Processor",
                "cpu_cores": 6,
                "cpu_threads": 12,
                "mem_total_gb": 31.889,
            }
        )
    rows.append(
        {
            "engine": "MapReduce",
            "tag": "",
            "rows_drawn": "",
            "baskets": "",
            "wall_seconds": "",
            "load_seconds": "",
            "count_seconds": "",
            "process_seconds": "",
            "rounds": "",
            "workers": "",
            "spark_master": "",
            "status": "failed",
            "note": "Hadoop Streaming did not run. No seconds recorded.",
            "host": "goodpc",
            "cpu_model": "AMD Ryzen 5 5600X 6-Core Processor",
            "cpu_cores": 6,
            "cpu_threads": 12,
            "mem_total_gb": 31.889,
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT, index=False)
    plot_timing(frame[frame["status"] == "ok"])
    print(frame[frame["status"] == "ok"][
        ["engine", "tag", "baskets", "wall_seconds", "count_seconds", "process_seconds", "spark_master"]
    ].to_string(index=False))
    print(f"wrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
