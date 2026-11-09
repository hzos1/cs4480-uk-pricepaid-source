"""
BASIC RUN — Task 2, larger samples. Single-machine scikit-learn, not Spark or MapReduce.

Choose k again on 200k, 1M, 5M, 10M, 20M, and every row.

For each size, fit k-means for k=2..8 with n_init=10 and keep the k
with the highest silhouette score. Silhouette is computed on 30,000
rows of that sample, not on every row.

Source is uk_features_scaled_all.txt (all Price Paid rows, 1995 to the
last date in the file). Run this on the other computer, not the Mac.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "parallel" / "uk_features_scaled_all.txt"
OUT = ROOT / "data" / "processed" / "choose_k_by_size"
FEAT_NAMES = ["log_price", "year", "month", "type_code", "is_new", "is_freehold"]
SIZES = [200_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, None]
K_MIN = 2
K_MAX = 8
N_INIT = 10
SIL_N = 30_000
SEED = 42


def load_all() -> np.ndarray:
    print(f"Loading {DATA} ...", flush=True)
    X = np.loadtxt(DATA, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = np.ascontiguousarray(X[:, : len(FEAT_NAMES)])
    print(f"Loaded {len(X):,} rows", flush=True)
    return X


def choose_one(sample: np.ndarray, tag: str, rng: np.random.RandomState) -> int:
    n_used = len(sample)
    sil_n = min(SIL_N, n_used)
    sil_idx = rng.choice(n_used, size=sil_n, replace=False)
    X_sil = sample[sil_idx]
    rows = []
    print(f"Choosing k on {n_used:,} rows (n_init={N_INIT}) ...", flush=True)
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=N_INIT, max_iter=300, random_state=SEED)
        labels = km.fit_predict(sample)
        sil = float(silhouette_score(X_sil, km.predict(X_sil)))
        rows.append(
            {
                "rows_used": n_used,
                "k": k,
                "inertia": km.inertia_,
                "n_iter": int(km.n_iter_),
                "silhouette": sil,
                "n_init": N_INIT,
                "silhouette_rows": sil_n,
            }
        )
        print(
            f"  {tag} k={k} inertia={km.inertia_:.0f} "
            f"iters={km.n_iter_} silhouette={sil:.4f}",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    best = int(metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / f"{tag}_k_metrics.csv", index=False)
    (OUT / f"{tag}_summary.txt").write_text(
        "\n".join(
            [
                f"source={DATA.name}",
                f"rows_used={n_used}",
                f"k_tested={K_MIN}..{K_MAX}",
                f"n_init={N_INIT}",
                f"silhouette_rows={sil_n}",
                f"best_k={best}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  {tag} best_k={best}", flush=True)
    return best


def main() -> None:
    if not DATA.is_file():
        raise SystemExit(f"Missing {DATA}. Run local_python/build_uk_panel.py first.")
    rng = np.random.RandomState(SEED)
    X = load_all()
    chosen = []
    for n in SIZES:
        if n is None or n >= len(X):
            sample = X
            tag = "all"
        else:
            idx = rng.choice(len(X), size=n, replace=False)
            sample = X[idx]
            tag = f"n{n}"
        best = choose_one(sample, tag, rng)
        chosen.append({"tag": tag, "rows_used": len(sample), "best_k": best, "n_init": N_INIT})
    pd.DataFrame(chosen).to_csv(OUT / "best_k_by_size.csv", index=False)
    print(f"Done. Results in {OUT}")


if __name__ == "__main__":
    main()
