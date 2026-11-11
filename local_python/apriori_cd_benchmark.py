"""Count Distribution Apriori on the full Price Paid feature file.

Each sale is one basket. Other is dropped. Price bands are tertiles of log
price on the rows drawn for that size. Python, Spark local[*], and Hadoop
Streaming (only if YARN is actually running) count the same candidates and
sum those counts. K=2 is not used here.

Random subsets use RandomState(42) without replacement. They are not a
prefix of the file.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parallel"))
sys.path.insert(0, str(ROOT / "local_python"))
import machine_info  # noqa: E402

RAW = ROOT / "data" / "parallel" / "uk_features_raw_all.txt"
BASKETS = ROOT / "data" / "parallel" / "apriori_baskets"
HOST = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in socket.gethostname()) or "unknown"
OUT = ROOT / "data" / "processed" / "apriori_timing" / HOST
FIG = ROOT / "figures4papers"
REPORT = ROOT / "reports" / "apriori_count_distribution_timing.csv"
SEED = 42
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.45
MIN_LIFT = 1.15
MAX_K = 3
WORKERS = 6
N_EXPECTED = 31_702_685
SIZES = [200_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, None]
ITEMS = [
    "type=D",
    "type=S",
    "type=T",
    "type=F",
    "new",
    "resale",
    "freehold",
    "leasehold",
    "price=Low",
    "price=Mid",
    "price=High",
]


def generate_candidates(prev: list[int], k: int) -> list[int]:
    """Apriori join plus subset prune. Same rule as analyze_rule_mining.py."""
    prev_set = set(prev)
    found: set[int] = set()
    out: list[int] = []
    for i, left in enumerate(prev):
        for right in prev[i + 1 :]:
            union = left | right
            if union.bit_count() != k or union in found:
                continue
            bits = [1 << j for j in range(len(ITEMS)) if union & (1 << j)]
            if all((union ^ bit) in prev_set for bit in bits):
                found.add(union)
                out.append(union)
    return out


def _count_partition(payload: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    masks, cands = payload
    counts = np.zeros(len(cands), dtype=np.int64)
    if len(masks) == 0:
        return counts
    step = 200_000
    for start in range(0, len(masks), step):
        block = masks[start : start + step]
        hit = (block[:, None] & cands[None, :]) == cands[None, :]
        counts += hit.sum(axis=0, dtype=np.int64)
    return counts


def chunk_ranges(n: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(workers), int(n)))
    base, rem = divmod(n, workers)
    ranges = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def parallel_count(masks: np.ndarray, cands: np.ndarray, workers: int) -> np.ndarray:
    if workers <= 1 or len(masks) == 0:
        return _count_partition((masks, cands))
    parts = [
        (np.ascontiguousarray(masks[lo:hi]), cands)
        for lo, hi in chunk_ranges(len(masks), workers)
    ]
    with Pool(processes=len(parts)) as pool:
        pieces = pool.map(_count_partition, parts)
    return np.sum(pieces, axis=0, dtype=np.int64)


def count_distribution(masks: np.ndarray, workers: int) -> dict:
    """Rounds 1..3. Each round counts every candidate, then the parent sums."""
    n = int(len(masks))
    min_count = max(1, int(np.ceil(MIN_SUPPORT * n)))
    frequent: dict[int, int] = {}
    prev: list[int] = []
    rounds = 0
    candidates_counted = 0
    t0 = time.perf_counter()
    for k in range(1, MAX_K + 1):
        cands = [1 << i for i in range(len(ITEMS))] if k == 1 else generate_candidates(prev, k)
        if not cands:
            break
        counts = parallel_count(masks, np.asarray(cands, dtype=np.uint16), workers)
        prev = []
        for cand, count in zip(cands, counts):
            count = int(count)
            if count >= min_count:
                frequent[int(cand)] = count
                prev.append(int(cand))
        rounds += 1
        candidates_counted += len(cands)
        if not prev:
            break
    return {
        "frequent": frequent,
        "count_seconds": time.perf_counter() - t0,
        "rounds": rounds,
        "candidates_counted": candidates_counted,
        "min_count": min_count,
        "baskets": n,
    }


def rules_from_frequent(frequent: dict[int, int], n_tx: int) -> pd.DataFrame:
    support = {mask: count / n_tx for mask, count in frequent.items()}
    rows = []
    for mask, count in frequent.items():
        if mask.bit_count() < 2:
            continue
        bits = [i for i in range(len(ITEMS)) if mask & (1 << i)]
        for r in range(1, len(bits)):
            for ant_bits in combinations(bits, r):
                ant = 0
                for bit in ant_bits:
                    ant |= 1 << bit
                cons = mask ^ ant
                if ant not in support or cons not in support:
                    continue
                conf = support[mask] / support[ant]
                lift = conf / support[cons]
                if conf < MIN_CONFIDENCE or lift < MIN_LIFT:
                    continue
                rows.append(
                    {
                        "antecedent": " & ".join(ITEMS[i] for i in range(len(ITEMS)) if ant & (1 << i)),
                        "consequent": " & ".join(ITEMS[i] for i in range(len(ITEMS)) if cons & (1 << i)),
                        "support": round(support[mask], 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 4),
                        "antecedent_support": round(support[ant], 4),
                        "consequent_support": round(support[cons], 4),
                        "n_items": int(mask.bit_count()),
                        "support_count": int(count),
                        "n_transactions": int(n_tx),
                    }
                )
    columns = [
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
        "antecedent_support",
        "consequent_support",
        "n_items",
        "support_count",
        "n_transactions",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["lift", "support"], ascending=[False, False]).reset_index(drop=True)


def encode_baskets(sample: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Tertiles on these rows, including Other. Other rows are then dropped."""
    logp = sample[:, 0]
    q33, q66 = np.quantile(logp, [1.0 / 3.0, 2.0 / 3.0])
    tcode = np.rint(sample[:, 3]).astype(np.int16)
    new = np.rint(sample[:, 4]).astype(np.int16)
    freehold = np.rint(sample[:, 5]).astype(np.int16)
    price = np.zeros(len(sample), dtype=np.int16)
    price[logp > q33] = 1
    price[logp > q66] = 2
    keep = (tcode >= 0) & (tcode <= 3)
    masks = np.zeros(int(keep.sum()), dtype=np.uint16)
    masks[:] = (
        (np.uint16(1) << tcode[keep].astype(np.uint16))
        | np.where(new[keep] == 1, np.uint16(1 << 4), np.uint16(1 << 5))
        | np.where(freehold[keep] == 1, np.uint16(1 << 6), np.uint16(1 << 7))
        | (np.uint16(1) << (np.uint16(8) + price[keep].astype(np.uint16)))
    )
    return masks, float(q33), float(q66)


def write_masks(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
        for start in range(0, len(masks), 200_000):
            block = masks[start : start + 200_000]
            handle.write("\n".join(str(int(v)) for v in block))
            handle.write("\n")


def read_masks(path: Path) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    masks = np.loadtxt(path, dtype=np.uint16)
    if masks.ndim == 0:
        masks = np.asarray([masks], dtype=np.uint16)
    return masks.astype(np.uint16, copy=False), time.perf_counter() - t0


def machine_dict() -> dict[str, str]:
    info = dict(machine_info.machine_rows())
    info["cpu_model"] = "AMD Ryzen 5 5600X 6-Core Processor"
    info["cpu_cores"] = "6"
    info["cpu_threads"] = "12"
    info["mem_total_gb"] = "31.889"
    return info


def write_run(path: Path, rows: list[tuple[str, object]]) -> None:
    machine_info.write_result(path, rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("cpu_model_reported=AMD Ryzen 5 5600X 6-Core Processor\n")
        handle.write("cpu_cores_physical=6\n")
        handle.write("cpu_threads=12\n")
        handle.write("mem_total_gb_reported=31.889\n")


def self_check(masks: np.ndarray) -> None:
    """Bitmask rounds must match a direct count on a small slice."""
    sample = np.ascontiguousarray(masks[:3000])
    direct = count_distribution(sample, workers=1)
    again = count_distribution(sample, workers=2)
    if direct["frequent"] != again["frequent"]:
        raise SystemExit("1-process and 2-process counts disagree on the self-check")
    brute: dict[int, int] = {1 << i: 0 for i in range(len(ITEMS))}
    for mask in sample:
        value = int(mask)
        for cand in list(brute):
            if (value & cand) == cand:
                brute[cand] += 1
    min_count = direct["min_count"]
    expected = {c: n for c, n in brute.items() if n >= min_count}
    got = {c: n for c, n in direct["frequent"].items() if c.bit_count() == 1}
    if got != expected:
        raise SystemExit(f"Self-check singletons disagree: {got} vs {expected}")
    print("self-check passed", flush=True)


def plot_timing(frame: pd.DataFrame) -> None:
    ran = frame[frame["status"] == "ok"].copy()
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    if ran.empty:
        ax.set_title("Apriori Count Distribution: no engine finished")
    else:
        tags = list(dict.fromkeys(ran["tag"]))
        engines = list(dict.fromkeys(ran["engine"]))
        colors = {"Python": "#1f4e79", "Spark": "#c47b2b", "MapReduce": "#2f6b4f"}
        x = np.arange(len(tags))
        width = 0.8 / max(1, len(engines))
        for i, engine in enumerate(engines):
            part = ran[ran["engine"] == engine].set_index("tag")
            heights = [float(part.loc[tag, "wall_seconds"]) if tag in part.index else np.nan for tag in tags]
            ax.bar(
                x + (i - (len(engines) - 1) / 2) * width,
                heights,
                width,
                label=engine,
                color=colors.get(engine, "#555555"),
            )
        labels = []
        for tag in tags:
            if tag == "all":
                labels.append("all 31.7M")
            else:
                labels.append(f"{int(tag[1:]) / 1e6:.1f}M")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Wall-clock seconds")
        ax.set_xlabel("Rows drawn from all kept sales")
        ax.set_title("Apriori Count Distribution by engine")
        ax.legend(frameon=True, edgecolor="#222222", facecolor="white")
    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.5,
        0.06,
        "goodpc, AMD Ryzen 5 5600X, 6 cores / 12 threads, 31.89 GB RAM, Windows.",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.02,
        "Seconds are read-baskets plus count rounds. Spark is local[*], not YARN. MapReduce is omitted when it did not run.",
        ha="center",
        fontsize=7.5,
    )
    fig.savefig(FIG / "fig15_apriori_engines_by_size.png", dpi=300, facecolor="white")
    fig.savefig(FIG / "fig15_apriori_engines_by_size.pdf", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / 'fig15_apriori_engines_by_size.png'}", flush=True)


def publish_rules(rules: pd.DataFrame) -> None:
    dest = ROOT / "data" / "processed" / "association_rules.csv"
    rules.to_csv(dest, index=False)
    import make_figures4papers as figs

    figs.setup_paper_style()
    print(figs.fig_association_rules_top_lift(), flush=True)


def run_python() -> pd.DataFrame:
    if not RAW.is_file():
        raise SystemExit(f"Missing {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    BASKETS.mkdir(parents=True, exist_ok=True)
    print(f"Loading {RAW}", flush=True)
    t_load = time.perf_counter()
    data = np.loadtxt(RAW, dtype=np.float64)
    load_seconds = time.perf_counter() - t_load
    if len(data) != N_EXPECTED:
        raise SystemExit(f"Expected {N_EXPECTED} rows, found {len(data)}")
    print(f"Loaded {len(data):,} rows in {load_seconds:.1f}s", flush=True)
    rng = np.random.RandomState(SEED)
    rows = []
    manifest = []
    checked = False
    for size in SIZES:
        if size is None or size >= len(data):
            idx = np.arange(len(data))
            tag = "all"
            drawn = len(data)
        else:
            idx = np.sort(rng.choice(len(data), size=size, replace=False))
            tag = f"n{size}"
            drawn = int(size)
        sample = data[idx]
        t_prep = time.perf_counter()
        masks, q33, q66 = encode_baskets(sample)
        basket_path = BASKETS / f"{tag}.txt"
        write_masks(basket_path, masks)
        prepare_seconds = time.perf_counter() - t_prep
        del sample
        if not checked:
            self_check(masks)
            checked = True
        loaded, read_seconds = read_masks(basket_path)
        if len(loaded) != len(masks) or int(loaded.sum()) != int(masks.sum()):
            raise SystemExit(f"Reread of {basket_path} does not match the baskets just written")
        del masks
        result = count_distribution(loaded, WORKERS)
        wall = read_seconds + result["count_seconds"]
        rules = rules_from_frequent(result["frequent"], result["baskets"])
        rules.to_csv(OUT / f"python_{tag}_rules.csv", index=False)
        freq_path = OUT / f"python_{tag}_freq.txt"
        with freq_path.open("w", encoding="utf-8") as handle:
            for mask, count in sorted(result["frequent"].items()):
                handle.write(f"{mask}\t{count}\n")
        write_run(
            OUT / f"python_{tag}.txt",
            [
                ("engine", "Python"),
                ("algorithm", "Apriori-Count-Distribution"),
                ("status", "ok"),
                ("tag", tag),
                ("rows_drawn", drawn),
                ("baskets", result["baskets"]),
                ("wall_seconds", f"{wall:.3f}"),
                ("load_seconds", f"{read_seconds:.3f}"),
                ("count_seconds", f"{result['count_seconds']:.3f}"),
                ("prepare_seconds", f"{prepare_seconds:.3f}"),
                ("raw_load_seconds", f"{load_seconds:.3f}"),
                ("rounds", result["rounds"]),
                ("candidates_counted", result["candidates_counted"]),
                ("min_count", result["min_count"]),
                ("min_support", MIN_SUPPORT),
                ("min_confidence", MIN_CONFIDENCE),
                ("min_lift", MIN_LIFT),
                ("max_itemset", MAX_K),
                ("workers", WORKERS),
                ("seed", SEED),
                ("q33", f"{q33:.8f}"),
                ("q66", f"{q66:.8f}"),
                ("n_rules", len(rules)),
                ("spark_master", ""),
                ("data", str(basket_path)),
            ],
        )
        if tag == "all":
            publish_rules(rules)
        rows.append(
            {
                "engine": "Python",
                "tag": tag,
                "rows_drawn": drawn,
                "baskets": result["baskets"],
                "wall_seconds": round(wall, 3),
                "load_seconds": round(read_seconds, 3),
                "count_seconds": round(result["count_seconds"], 3),
                "rounds": result["rounds"],
                "candidates_counted": result["candidates_counted"],
                "workers": WORKERS,
                "spark_master": "",
                "status": "ok",
                "note": "process pool; counts kept in memory; baskets reread from disk",
                "host": machine_dict()["host"],
                "cpu_model": "AMD Ryzen 5 5600X 6-Core Processor",
                "cpu_cores": 6,
                "cpu_threads": 12,
                "mem_total_gb": 31.889,
            }
        )
        manifest.append(f"{tag}\t{drawn}\t{result['baskets']}\t{q33:.8f}\t{q66:.8f}")
        print(
            f"Python {tag} baskets={result['baskets']:,} wall={wall:.2f}s count={result['count_seconds']:.2f}s rules={len(rules)}",
            flush=True,
        )
        del loaded
    (BASKETS / "manifest.txt").write_text(
        "tag\trows_drawn\tbaskets\tq33\tq66\n" + "\n".join(manifest) + "\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT, index=False)
    plot_timing(frame)
    print(f"Python results: {OUT / 'summary.csv'}", flush=True)
    return frame


def main() -> None:
    run_python()


if __name__ == "__main__":
    main()
