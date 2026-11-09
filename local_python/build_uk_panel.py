"""
BASIC RUN — Task 1: housing trend.

Builds the tables used to compare, for the same county and the same
property type, median sale price and the number of registered sales
across months and years. This file is single-machine pandas. It does
not run Spark or MapReduce.

Build analysis panels from UK Price Paid (read-only on raw CSVs).

Input (never modified):
  E:\\cs4480\\group_project_uk\\pp-complete.csv

Outputs (new files only):
  data/processed/...
  data/parallel/...
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_COMPLETE = ROOT / "pp-complete.csv"
OUT = ROOT / "data" / "processed"
PAR = ROOT / "data" / "parallel"

COLS = [
    "price",
    "date",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town",
    "district",
    "county",
    "ppd_cat",
    "record_status",
]

USECOLS = [
    "price",
    "date",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "town",
    "district",
    "county",
    "ppd_cat",
    "record_status",
]

TYPE_MAP = {"D": 0, "S": 1, "T": 2, "F": 3, "O": 4}
CHUNK = 250_000
# All registered sales. The old 2018-only file was a subset, then copied to fake a larger n.
PARALLEL_YEAR_MIN = 1995


def clean_label(s: pd.Series) -> pd.Series:
    """Fix Kaggle artifacts like TORQUAY.1 / TORBAY.1."""
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.\d+$", "", regex=True)
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("postcode", "property_type", "old_new", "duration", "town", "district", "county", "ppd_cat", "record_status"):
        if c in df.columns:
            df[c] = clean_label(df[c])
    df["property_type"] = df["property_type"].str.upper()
    df["old_new"] = df["old_new"].str.upper()
    df["duration"] = df["duration"].str.upper()
    # Standard PP filter: positive price, known type, valid date
    df = df.dropna(subset=["price", "date", "property_type"])
    df = df[df["price"] > 0]
    df = df[df["property_type"].isin(TYPE_MAP)]
    # Category B is kept. Drop only rows marked deleted.
    if "record_status" in df.columns:
        df = df[df["record_status"].fillna("A").str.upper() != "D"]
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["ym"] = df["date"].dt.to_period("M").astype(str)
    df["log_price"] = np.log1p(df["price"].astype("float64"))
    df["is_new"] = (df["old_new"] == "Y").astype("int8")
    df["is_freehold"] = (df["duration"] == "F").astype("int8")
    df["type_code"] = df["property_type"].map(TYPE_MAP).astype("int8")
    return df


def main() -> None:
    if not RAW_COMPLETE.is_file():
        raise SystemExit(f"Missing raw file (read-only expected): {RAW_COMPLETE}")

    OUT.mkdir(parents=True, exist_ok=True)
    PAR.mkdir(parents=True, exist_ok=True)

    # Accumulators
    n_raw = 0
    n_keep = 0
    price_sum = 0.0
    price_min = float("inf")
    price_max = float("-inf")
    year_min = 10**9
    year_max = -1
    type_counts: dict[str, int] = defaultdict(int)

    monthly_prices: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    county_prices: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    county_new: dict[tuple[int, str], float] = defaultdict(float)
    county_fh: dict[tuple[int, str], float] = defaultdict(float)

    # All years. Do not overwrite the older 2018-only feature files.
    feat_path = PAR / "uk_features_raw_all.txt"
    if feat_path.exists():
        feat_path.unlink()

    # Online stats for z-score (6 features), memory-safe
    feat_dim = 6
    feat_n = 0
    feat_sum = np.zeros(feat_dim, dtype=np.float64)
    feat_sumsq = np.zeros(feat_dim, dtype=np.float64)
    # Reservoir for k=4 init centroids (scaled later)
    reservoir: list[np.ndarray] = []
    reservoir_k = 4
    reservoir_seen = 0
    rng = np.random.RandomState(42)

    reader = pd.read_csv(
        RAW_COMPLETE,
        header=None,
        names=COLS,
        usecols=USECOLS,
        dtype=str,
        chunksize=CHUNK,
        low_memory=False,
    )

    print(f"Reading {RAW_COMPLETE} in chunks of {CHUNK} ...")
    with feat_path.open("w", encoding="utf-8") as feat_f:
        for i, chunk in enumerate(reader, start=1):
            n_raw += len(chunk)
            df = prepare_chunk(chunk)
            n_keep += len(df)
            if df.empty:
                continue

            price_sum += float(df["price"].sum())
            price_min = min(price_min, float(df["price"].min()))
            price_max = max(price_max, float(df["price"].max()))
            year_min = min(year_min, int(df["year"].min()))
            year_max = max(year_max, int(df["year"].max()))
            for t, c in df["property_type"].value_counts().items():
                type_counts[str(t)] += int(c)

            for (ym, ptype), sub in df.groupby(["ym", "property_type"], sort=False):
                monthly_prices[(str(ym), str(ptype))].append(
                    sub["price"].to_numpy(dtype=np.float64)
                )
            located = df.dropna(subset=["county"])
            for (year, county_name), sub in located.groupby(["year", "county"], sort=False):
                key = (int(year), str(county_name))
                county_prices[key].append(sub["price"].to_numpy(dtype=np.float64))
                county_new[key] += float(sub["is_new"].sum())
                county_fh[key] += float(sub["is_freehold"].sum())

            recent = df[df["year"] >= PARALLEL_YEAR_MIN]
            if not recent.empty:
                arr = np.column_stack(
                    [
                        recent["log_price"].to_numpy(dtype=np.float64),
                        recent["year"].to_numpy(dtype=np.float64),
                        recent["month"].to_numpy(dtype=np.float64),
                        recent["type_code"].to_numpy(dtype=np.float64),
                        recent["is_new"].to_numpy(dtype=np.float64),
                        recent["is_freehold"].to_numpy(dtype=np.float64),
                    ]
                )
                np.savetxt(feat_f, arr, fmt="%.8f")
                feat_n += arr.shape[0]
                feat_sum += arr.sum(axis=0)
                feat_sumsq += np.square(arr).sum(axis=0)
                # reservoir sample rows for shared init centroids
                for row in arr:
                    reservoir_seen += 1
                    if len(reservoir) < reservoir_k:
                        reservoir.append(row.copy())
                    else:
                        j = rng.randint(0, reservoir_seen)
                        if j < reservoir_k:
                            reservoir[j] = row.copy()

            if i % 10 == 0:
                print(f"  chunk {i}: raw_rows={n_raw:,} kept={n_keep:,} feat_rows={feat_n:,}")
    # BASIC RUN — Task 1. median_price = median sale price.
    # n = number of registered sales (transaction volume).
    print("Combining monthly / county aggregates (exact medians) ...")
    monthly_rows = []
    for (ym, ptype), parts in monthly_prices.items():
        prices = np.concatenate(parts)
        monthly_rows.append(
            {
                "ym": ym,
                "property_type": ptype,
                "n": int(prices.size),
                "median_price": float(np.median(prices)),
                "mean_price": float(prices.mean()),
                "mean_log_price": float(np.log1p(prices).mean()),
            }
        )
    monthly = pd.DataFrame(monthly_rows).sort_values(["ym", "property_type"])
    monthly_path = OUT / "uk_monthly_by_type.csv"
    monthly.to_csv(monthly_path, index=False)

    county_rows = []
    for key, parts in county_prices.items():
        year, county_name = key
        prices = np.concatenate(parts)
        n = int(prices.size)
        county_rows.append(
            {
                "year": year,
                "county": county_name,
                "n": n,
                "median_price": float(np.median(prices)),
                "mean_log_price": float(np.log1p(prices).mean()),
                "new_share": county_new[key] / n,
                "freehold_share": county_fh[key] / n,
            }
        )
    county_year = pd.DataFrame(county_rows).sort_values(["year", "county"])
    county_path = OUT / "uk_county_year_stats.csv"
    county_year.to_csv(county_path, index=False)

    # Stream-scale parallel features (second pass over feat_path only)
    print("Scaling parallel feature file (streaming) ...")
    mean = feat_sum / max(feat_n, 1)
    var = feat_sumsq / max(feat_n, 1) - np.square(mean)
    var = np.maximum(var, 1e-12)
    scale = np.sqrt(var)

    scaled_path = PAR / "uk_features_scaled_all.txt"
    sample_path = PAR / "uk_features_scaled_all_sample10pct.txt"
    if scaled_path.exists():
        scaled_path.unlink()
    if sample_path.exists():
        sample_path.unlink()

    target_sample = feat_n // 10 if feat_n else 0
    chosen: set[int] = set()
    if feat_n and target_sample:
        chosen = set(int(i) for i in rng.choice(feat_n, size=target_sample, replace=False))
    sample_n = 0
    valid_i = 0
    with feat_path.open("r", encoding="utf-8") as src, scaled_path.open(
        "w", encoding="utf-8"
    ) as dst, sample_path.open("w", encoding="utf-8") as smp:
        for line in src:
            vals = np.fromstring(line, sep=" ")
            if vals.size != feat_dim:
                continue
            z = (vals - mean) / scale
            dst.write(" ".join(f"{x:.8f}" for x in z) + "\n")
            if valid_i in chosen:
                smp.write(" ".join(f"{x:.8f}" for x in z) + "\n")
                sample_n += 1
            valid_i += 1

    # Shared init centroids from reservoir, then scale
    cents_raw = np.vstack(reservoir) if reservoir else np.zeros((4, feat_dim))
    cents = (cents_raw - mean) / scale
    np.savetxt(PAR / "init_centroids_k4_all.txt", cents, fmt="%.8f")

    feat_names = ["log_price", "year", "month", "type_code", "is_new", "is_freehold"]
    meta = pd.DataFrame({"feature": feat_names, "mean": mean, "scale": scale})
    meta.to_csv(OUT / "scaler_params_all.csv", index=False)

    inventory = {
        "raw_file": str(RAW_COMPLETE),
        "raw_bytes": RAW_COMPLETE.stat().st_size,
        "rows_raw_seen": n_raw,
        "rows_kept": n_keep,
        "year_min": year_min,
        "year_max": year_max,
        "price_min": price_min,
        "price_max": price_max,
        "price_mean": price_sum / max(n_keep, 1),
        "property_type_counts": dict(type_counts),
        "parallel_year_min": PARALLEL_YEAR_MIN,
        "parallel_rows": feat_n,
        "parallel_sample_rows": sample_n,
        "features": feat_names,
        "outputs": {
            "monthly": str(monthly_path),
            "county_year": str(county_path),
            "features_raw": str(feat_path),
            "features_scaled": str(scaled_path),
            "features_sample": str(sample_path),
            "init_centroids": str(PAR / "init_centroids_k4_all.txt"),
        },
        "note": "Original CSVs under group_project_uk root were not modified.",
    }
    inv_path = OUT / "data_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (OUT / "data_inventory.txt").write_text(
        "\n".join(
            [
                f"rows_raw_seen={n_raw:,}",
                f"rows_kept={n_keep:,}",
                f"years={year_min}..{year_max}",
                f"price_mean={inventory['price_mean']:.2f}",
                f"types={dict(type_counts)}",
                f"parallel_all_rows={feat_n:,}",
                f"sample_rows={sample_n:,}",
                f"monthly={monthly_path.name}",
                f"county_year={county_path.name}",
                f"scaled={scaled_path.name}",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(inventory, indent=2))
    print(f"Wrote {inv_path}")

if __name__ == "__main__":
    main()
