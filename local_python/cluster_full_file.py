"""Fit the full-file silhouette winner and draw the k-selection figures.

choose_k_by_size.py must already have finished. The k used here is the
winner on tag "all", not the winner on a smaller sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
SCALED = ROOT / "data" / "parallel" / "uk_features_scaled_all.txt"
KDIR = ROOT / "data" / "processed" / "choose_k_by_size"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures4papers"
FEAT_NAMES = ["log_price", "year", "month", "type_code", "is_new", "is_freehold"]
SEED = 42


def best_k_all() -> int:
    summary = KDIR / "all_summary.txt"
    if not summary.is_file():
        raise SystemExit(f"Missing {summary}. Finish choose_k_by_size.py first.")
    for line in summary.read_text(encoding="utf-8").splitlines():
        if line.startswith("best_k="):
            return int(line.split("=", 1)[1])
    raise SystemExit(f"No best_k in {summary}")


def plot_full_file_choice(k_used: int) -> None:
    metrics = pd.read_csv(KDIR / "all_k_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    axes[0].plot(metrics["k"], metrics["inertia"], marker="o", color="#1f4e79")
    axes[0].set_title("Inertia, full file")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#b85c38")
    axes[1].axvline(k_used, color="#222222", lw=1, ls="--")
    axes[1].set_title("Silhouette, full file")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    fig.suptitle(f"k chosen on all 31,702,685 rows: k={k_used}", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig5_kmeans_elbow_silhouette.png", dpi=300, facecolor="white")
    fig.savefig(FIG / "fig5_kmeans_elbow_silhouette.pdf", facecolor="white")
    plt.close(fig)

    chosen = pd.read_csv(KDIR / "best_k_by_size.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labels = []
    for tag, rows in zip(chosen["tag"], chosen["rows_used"]):
        labels.append("all 31.7M" if tag == "all" else f"{int(rows)/1e6:.1f}M")
    ax.plot(range(len(chosen)), chosen["best_k"], marker="o", color="#1f4e79")
    ax.set_xticks(range(len(chosen)))
    ax.set_xticklabels(labels)
    ax.set_ylim(1.5, 8.5)
    ax.set_ylabel("k with the highest silhouette")
    ax.set_xlabel("Rows used for the fit")
    ax.set_title("Chosen k by sample size")
    fig.tight_layout()
    fig.savefig(FIG / "fig5b_chosen_k_by_size.png", dpi=300, facecolor="white")
    fig.savefig(FIG / "fig5b_chosen_k_by_size.pdf", facecolor="white")
    plt.close(fig)
    print("wrote fig5 and fig5b", flush=True)


def plot_centers(centers: pd.DataFrame) -> None:
    z = centers.copy()
    z_norm = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sns.heatmap(z_norm, annot=centers.round(2), fmt="", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Full-file cluster centres (numbers are z-scored means)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_kmeans_cluster_profiles.png", dpi=300, facecolor="white")
    fig.savefig(FIG / "fig6_kmeans_cluster_profiles.pdf", facecolor="white")
    plt.close(fig)
    print("wrote fig6", flush=True)


def main() -> None:
    k_used = best_k_all()
    print(f"Full-file k = {k_used}", flush=True)
    plot_full_file_choice(k_used)
    print(f"Loading {SCALED} ...", flush=True)
    X = np.loadtxt(SCALED, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = np.ascontiguousarray(X[:, : len(FEAT_NAMES)])
    print(f"Fitting k={k_used} n_init=10 on {len(X):,} rows ...", flush=True)
    km = KMeans(n_clusters=k_used, n_init=10, max_iter=300, random_state=SEED)
    labels = km.fit_predict(X)
    centers = pd.DataFrame(km.cluster_centers_, columns=FEAT_NAMES)
    centers.index.name = "cluster"
    PROC.mkdir(parents=True, exist_ok=True)
    centers.to_csv(PROC / "kmeans_cluster_centers_full.csv")
    centers.to_csv(PROC / "kmeans_cluster_centers.csv")
    sizes = pd.Series(labels).value_counts().sort_index()
    (PROC / "kmeans_full_summary.txt").write_text(
        "\n".join(
            [
                f"rows={len(X)}",
                f"k={k_used}",
                "n_init=10",
                "max_iter=300",
                f"random_state={SEED}",
                f"iterations={int(km.n_iter_)}",
                f"inertia={float(km.inertia_):.4f}",
                "cluster_sizes:",
                sizes.to_string(),
                "",
                "centers:",
                centers.round(4).to_string(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plot_centers(centers)
    print(f"Done. k={k_used} iterations={int(km.n_iter_)}", flush=True)


if __name__ == "__main__":
    main()
