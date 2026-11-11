"""Apriori at the same sizes as choose_k, counted in parallel.

Sizes are random subsets of every kept sale, plus the full file.
They are not a prefix of the feature file. Other (type code 4) is dropped
after the subset is drawn. Price bands are tertiles of log price inside
that subset.

Each size is timed twice: one process, then a process pool. Both counts
use every selected row. The pool splits the encoded rows. Workers do not
reload the raw file.
"""
from __future__ import annotations

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
import machine_info  # noqa: E402

RAW = ROOT / "data" / "parallel" / "uk_features_raw_all.txt"
OUT = ROOT / "data" / "processed" / "association_by_size"
FIG = ROOT / "figures4papers"
REPORT = ROOT / "reports" / "association_timing_by_size.csv"
SIZES = [200_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, None]
SEED = 42
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.45
MIN_LIFT = 1.15
MAX_K = 3
WORKERS = 6

ITEM_NAMES = [
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


def load_raw() -> np.ndarray:
    print(f"Loading {RAW} ...", flush=True)
    t0 = time.perf_counter()
    data = np.loadtxt(RAW, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    print(f"Loaded {len(data):,} rows in {time.perf_counter() - t0:.1f}s", flush=True)
    return data


def encode(logp: np.ndarray, type_code: np.ndarray, is_new: np.ndarray, is_fh: np.ndarray) -> np.ndarray:
    """Pack one basket into 11 bits. Rows with type Other become 0 and are unused."""
    q33, q66 = np.quantile(logp, [1.0 / 3.0, 2.0 / 3.0])
    tcode = np.rint(type_code).astype(np.int16)
    new = np.rint(is_new).astype(np.int16)
    fh = np.rint(is_fh).astype(np.int16)
    price = np.zeros(len(logp), dtype=np.int16)
    price[logp > q33] = 1
    price[logp > q66] = 2
    masks = np.zeros(len(logp), dtype=np.uint16)
    keep = (tcode >= 0) & (tcode <= 3)
    masks[keep] = (
        (np.uint16(1) << tcode[keep].astype(np.uint16))
        | np.where(new[keep] == 1, np.uint16(1 << 4), np.uint16(1 << 5))
        | np.where(fh[keep] == 1, np.uint16(1 << 6), np.uint16(1 << 7))
        | (np.uint16(1) << (np.uint16(8) + price[keep].astype(np.uint16)))
    )
    return masks[keep]


def candidates() -> list[int]:
    bits = list(range(len(ITEM_NAMES)))
    out: list[int] = []
    for k in range(1, MAX_K + 1):
        for combo in combinations(bits, k):
            # One value from each family: type, new/resale, tenure, price.
            def family(bit: int) -> int:
                if bit < 4:
                    return 0
                if bit < 6:
                    return 1
                if bit < 8:
                    return 2
                return 3

            families = [family(b) for b in combo]
            if len(families) != len(set(families)):
                continue
            mask = 0
            for b in combo:
                mask |= 1 << b
            out.append(mask)
    return out


def _count_chunk(payload: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    masks, cand = payload
    counts = np.zeros(len(cand), dtype=np.int64)
    if len(masks) == 0:
        return counts
    step = 250_000
    for start in range(0, len(masks), step):
        block = masks[start : start + step]
        hit = (block[:, None] & cand[None, :]) == cand[None, :]
        counts += hit.sum(axis=0, dtype=np.int64)
    return counts


def count_supports(masks: np.ndarray, cand: np.ndarray, workers: int) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    if workers <= 1 or len(masks) < 50_000:
        counts = _count_chunk((masks, cand))
        return counts, time.perf_counter() - t0
    ranges = chunk_ranges(len(masks), workers)
    parts = [(np.ascontiguousarray(masks[lo:hi]), cand) for lo, hi in ranges]
    with Pool(processes=len(parts)) as pool:
        pieces = pool.map(_count_chunk, parts)
    counts = np.sum(pieces, axis=0, dtype=np.int64)
    return counts, time.perf_counter() - t0


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


def rules_from_counts(cand: np.ndarray, counts: np.ndarray, n_tx: int) -> pd.DataFrame:
    support = {int(m): float(c) / n_tx for m, c in zip(cand, counts) if c > 0}
    rows = []
    frequent = [m for m, c in zip(cand, counts) if c >= MIN_SUPPORT * n_tx and bin(int(m)).count("1") >= 2]
    for mask in frequent:
        bits = [i for i in range(len(ITEM_NAMES)) if mask & (1 << i)]
        for r in range(1, len(bits)):
            for ant_bits in combinations(bits, r):
                ant = 0
                for b in ant_bits:
                    ant |= 1 << b
                cons = mask ^ ant
                if ant not in support or cons not in support or mask not in support:
                    continue
                conf = support[mask] / support[ant]
                lift = conf / support[cons]
                if conf < MIN_CONFIDENCE or lift < MIN_LIFT:
                    continue
                rows.append(
                    {
                        "antecedent": " & ".join(ITEM_NAMES[i] for i in range(len(ITEM_NAMES)) if ant & (1 << i)),
                        "consequent": " & ".join(ITEM_NAMES[i] for i in range(len(ITEM_NAMES)) if cons & (1 << i)),
                        "support": round(support[mask], 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 4),
                        "antecedent_support": round(support[ant], 4),
                        "consequent_support": round(support[cons], 4),
                        "n_items": int(bin(int(mask)).count("1")),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
                "antecedent_support",
                "consequent_support",
                "n_items",
            ]
        )
    return pd.DataFrame(rows).sort_values(["lift", "support"], ascending=[False, False]).reset_index(drop=True)


def plot_runtime(timing: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(timing))
    width = 0.36
    ax.bar(x - width / 2, timing["seconds_1worker"], width, label="Python, 1 process", color="#4c6a80")
    ax.bar(x + width / 2, timing["seconds_parallel"], width, label=f"Python, {WORKERS} processes", color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)/1e6:.1f}M" if n < 31e6 else "all 31.7M" for n in timing["rows_drawn"]])
    ax.set_ylabel("Count time (seconds)")
    ax.set_xlabel("Rows drawn from the full cleaned file")
    ax.set_title("Association-rule counting time by sample size")
    ax.legend(frameon=True, edgecolor="#222222", facecolor="white")
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5,
        0.04,
        "Random subsets, not a file prefix. Time covers the count only. Machine: goodpc, Ryzen 5 5600X, 31.89 GB.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(FIG / "fig14_association_runtime_by_size.png", dpi=300, facecolor="white")
    fig.savefig(FIG / "fig14_association_runtime_by_size.pdf", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / 'fig14_association_runtime_by_size.png'}", flush=True)


def main() -> None:
    if not RAW.is_file():
        raise SystemExit(f"Missing {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_raw()
    n_all = len(data)
    rng = np.random.RandomState(SEED)
    cand = np.array(candidates(), dtype=np.uint16)
    timing_rows = []
    full_rules: pd.DataFrame | None = None
    for size in SIZES:
        if size is None or size >= n_all:
            idx = np.arange(n_all)
            tag = "all"
            drawn = n_all
        else:
            idx = rng.choice(n_all, size=size, replace=False)
            tag = f"n{size}"
            drawn = size
        sample = data[idx]
        print(f"Encoding {tag} ({drawn:,} drawn rows) ...", flush=True)
        masks = encode(sample[:, 0], sample[:, 3], sample[:, 4], sample[:, 5])
        n_tx = int(len(masks))
        print(f"  baskets excluding Other: {n_tx:,}", flush=True)
        counts_1, sec_1 = count_supports(masks, cand, 1)
        counts_p, sec_p = count_supports(masks, cand, WORKERS)
        if not np.array_equal(counts_1, counts_p):
            raise SystemExit(f"Parallel count disagreed with the one-process count for {tag}")
        rules = rules_from_counts(cand, counts_p, n_tx)
        rules.to_csv(OUT / f"{tag}_rules.csv", index=False)
        if tag == "all":
            full_rules = rules
            rules.to_csv(ROOT / "data" / "processed" / "association_rules.csv", index=False)
        timing_rows.append(
            {
                "tag": tag,
                "rows_drawn": drawn,
                "baskets_excl_other": n_tx,
                "seconds_1worker": round(sec_1, 3),
                "seconds_parallel": round(sec_p, 3),
                "parallel_workers": WORKERS,
                "n_rules": int(len(rules)),
                "top_lift": None if rules.empty else float(rules.iloc[0]["lift"]),
            }
        )
        print(
            f"  {tag} 1-process {sec_1:.2f}s  {WORKERS}-process {sec_p:.2f}s  rules {len(rules)}",
            flush=True,
        )
    timing = pd.DataFrame(timing_rows)
    timing_path = OUT / "timing.csv"
    timing.to_csv(timing_path, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    timing.to_csv(REPORT, index=False)
    machine = dict(machine_info.machine_rows())
    (OUT / "machine.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in machine.items()) + "\n",
        encoding="utf-8",
    )
    plot_runtime(timing)
    if full_rules is not None and not full_rules.empty:
        sys.path.insert(0, str(ROOT / "local_python"))
        import make_figures4papers as figs

        figs.setup_paper_style()
        print(figs.fig_association_rules_top_lift())
    print(f"Wrote {timing_path}")


if __name__ == "__main__":
    main()
