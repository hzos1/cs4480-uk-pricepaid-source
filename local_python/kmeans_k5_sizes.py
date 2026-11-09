"""
k-means at k=5 on 200k, 1M, 5M, 10M, 20M, and every row of uk_features_scaled_all.txt.

Rows are a random subset of all Price Paid sales (1995 to the last date in the file),
not a copied file and not the 2018-only extract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "parallel" / "uk_features_scaled_all.txt"
OUT = ROOT / "data" / "processed" / "kmeans_k5"
FEAT_NAMES = ["log_price", "year", "month", "type_code", "is_new", "is_freehold"]
SIZES = [200_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, None]
K = 5
SEED = 42


def load_all() -> np.ndarray:
    print(f"Loading {DATA} ...", flush=True)
    X = np.loadtxt(DATA, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = X[:, : len(FEAT_NAMES)]
    print(f"Loaded {len(X):,} rows", flush=True)
    return X


def fit_one(X: np.ndarray, n: int | None, rng: np.random.RandomState) -> None:
    if n is None or n >= len(X):
        sample = X
        tag = "all"
        n_used = len(X)
    else:
        idx = rng.choice(len(X), size=n, replace=False)
        sample = X[idx]
        tag = f"n{n}"
        n_used = n
    n_init = 10 if n_used <= 1_000_000 else 3
    print(f"Fitting k={K} n_init={n_init} on {n_used:,} rows ...", flush=True)
    km = KMeans(n_clusters=K, n_init=n_init, max_iter=300, random_state=SEED)
    labels = km.fit_predict(sample)
    centers = pd.DataFrame(km.cluster_centers_, columns=FEAT_NAMES)
    centers.index.name = "cluster"
    counts = pd.Series(labels).value_counts().sort_index()
    sil_n = min(20_000, n_used)
    sil_idx = rng.choice(n_used, size=sil_n, replace=False)
    sil = float(silhouette_score(sample[sil_idx], labels[sil_idx]))
    OUT.mkdir(parents=True, exist_ok=True)
    centers.to_csv(OUT / f"{tag}_centers.csv")
    summary = [
        f"source={DATA.name}",
        f"rows_in_source={len(X)}",
        f"rows_used={n_used}",
        f"k={K}",
        f"n_init={n_init}",
        f"inertia={km.inertia_:.4f}",
        f"n_iter={int(km.n_iter_)}",
        f"silhouette_on_{sil_n}={sil:.4f}",
        "",
        "cluster_sizes:",
        counts.to_string(),
        "",
        "centers:",
        centers.round(4).to_string(),
    ]
    (OUT / f"{tag}_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"  wrote {tag}: inertia={km.inertia_:.0f} iters={km.n_iter_} silhouette={sil:.4f}", flush=True)


def main() -> None:
    if not DATA.is_file():
        raise SystemExit(f"Missing {DATA}. Run local_python/build_uk_panel.py first.")
    rng = np.random.RandomState(SEED)
    X = load_all()
    lines = [f"source_rows={len(X)}", f"k={K}", ""]
    for n in SIZES:
        fit_one(X, n, rng)
        tag = "all" if n is None or n >= len(X) else f"n{n}"
        lines.append(tag)
    (OUT / "index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Results in {OUT}")


if __name__ == "__main__":
    main()
