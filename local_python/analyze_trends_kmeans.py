"""
BASIC RUN — two tasks, both single-machine. No Spark or MapReduce.

Task 1, housing trend: plot_median_price_by_type and plot_volume_by_type.
  Price is the median sale price. Volume is the number of registered sales.
Task 2, clustering: choose_k and run_kmeans.
  choose_k tests k = 2..8 and keeps the highest silhouette score.
  The centres are then read against new-build, leasehold/flat, and freehold.

Reads only derived files under data/; never touches raw CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PAR = ROOT / "data" / "parallel"
FIG = ROOT / "figures"

FEAT_NAMES = ["log_price", "year", "month", "type_code", "is_new", "is_freehold"]
TYPE_LABEL = {0: "D", 1: "S", 2: "T", 3: "F", 4: "O"}
# Use 10% sample file for local k-means (still ~855k rows); subsample further if needed.
KMEANS_MAX_ROWS = 200_000
REPORT_K = 4


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f4",
            "axes.edgecolor": "#333333",
            "grid.color": "#d0d0d0",
            "font.size": 11,
        }
    )


def load_monthly() -> pd.DataFrame:
    df = pd.read_csv(PROC / "uk_monthly_by_type.csv")
    df["ym"] = pd.to_datetime(df["ym"] + "-01", errors="coerce")
    return df.dropna(subset=["ym"])


def load_county_year() -> pd.DataFrame:
    return pd.read_csv(PROC / "uk_county_year_stats.csv")


# BASIC RUN — Task 1. Price = median sale price, by month and property type.
def plot_median_price_by_type(monthly: pd.DataFrame) -> Path:
    """BASIC RUN — Task 1. Median sale price by property type across months."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for t, g in monthly.groupby("property_type"):
        if t == "O":
            continue  # sparse / extreme outliers dominate scale
        ax.plot(g["ym"], g["median_price"], label=t, linewidth=1.6)
    ax.set_title("UK Price Paid — median price by property type")
    ax.set_xlabel("Month")
    ax.set_ylabel("Median price (£)")
    ax.legend(title="Type", frameon=False, ncol=4)
    fig.tight_layout()
    out = FIG / "trend_median_price_by_type.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# BASIC RUN — Task 1. Volume = number of registered sales, by month and property type.
def plot_volume_by_type(monthly: pd.DataFrame) -> Path:
    """BASIC RUN — Task 1. Transaction volume = number of registered sales."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for t, g in monthly.groupby("property_type"):
        if t == "O":
            continue
        ax.plot(g["ym"], g["n"], label=t, linewidth=1.3, alpha=0.9)
    ax.set_title("UK Price Paid — monthly transaction count by type")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transactions")
    ax.legend(title="Type", frameon=False, ncol=4)
    fig.tight_layout()
    out = FIG / "trend_volume_by_type.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_national_median(monthly: pd.DataFrame) -> Path:
    # weighted median approx via mean of type medians weighted by n (excluding O)
    sub = monthly[monthly["property_type"] != "O"].copy()
    sub["w"] = sub["median_price"] * sub["n"]
    nat = sub.groupby("ym", as_index=False).agg(n=("n", "sum"), w=("w", "sum"))
    nat["median_approx"] = nat["w"] / nat["n"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(nat["ym"], nat["median_approx"], color="#1f4e79", linewidth=2)
    ax.set_title("UK Price Paid — national median price (approx., excl. Other)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Median price (£)")
    fig.tight_layout()
    out = FIG / "trend_national_median_price.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# BASIC RUN — Task 1. Same county, one complete year: median sale price.
def plot_top_counties(county: pd.DataFrame, year: int = 2024) -> Path:
    y = county[county["year"] == year].nlargest(15, "median_price")
    if y.empty:
        year = int(county["year"].max())
        y = county[county["year"] == year].nlargest(15, "median_price")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y["county"][::-1], y["median_price"][::-1], color="#4c6a80")
    ax.set_title(f"Top 15 counties by median price ({year})")
    ax.set_xlabel("Median price (£)")
    fig.tight_layout()
    out = FIG / "trend_top_counties_median.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def load_features_for_kmeans() -> np.ndarray:
    """Reservoir-sample from the full 2018+ scaled file (not the head-only 10% file)."""
    path = PAR / "uk_features_scaled_2018plus.txt"
    if not path.exists():
        path = PAR / "uk_features_scaled_2018plus_sample10pct.txt"
    print(f"Reservoir-sampling {KMEANS_MAX_ROWS:,} rows from {path.name} ...")
    rng = np.random.RandomState(42)
    reservoir: list[np.ndarray] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = np.fromstring(line, sep=" ")
            if row.size != len(FEAT_NAMES):
                continue
            seen += 1
            if len(reservoir) < KMEANS_MAX_ROWS:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen)
                if j < KMEANS_MAX_ROWS:
                    reservoir[j] = row
            if seen % 1_000_000 == 0:
                print(f"  scanned {seen:,} ...")
    X = np.vstack(reservoir)
    print(f"Sampled {len(X):,} / {seen:,} rows")
    return X


# BASIC RUN — Task 2. Try k=2..8 and keep the k with the highest silhouette score.
def choose_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> tuple[int, pd.DataFrame]:
    """BASIC RUN — Task 2. Fit k = 2..8 and return the k with the highest silhouette."""
    rows = []
    # silhouette on a further subsample for speed
    rng = np.random.RandomState(0)
    sil_n = min(30000, len(X))
    sil_idx = rng.choice(len(X), size=sil_n, replace=False)
    X_sil = X[sil_idx]
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X_sil, km.predict(X_sil))
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k} inertia={km.inertia_:.0f} silhouette={sil:.4f}")
    metrics = pd.DataFrame(rows)
    best = metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best["k"]), metrics


def plot_elbow_silhouette(metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(metrics["k"], metrics["inertia"], marker="o")
    axes[0].set_title("Elbow (inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#b85c38")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    fig.tight_layout()
    out = FIG / "kmeans_elbow_silhouette.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_cluster_profiles(centers: pd.DataFrame) -> Path:
    z = centers.copy()
    z_norm = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(z_norm, annot=centers.round(2), fmt="", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Cluster centers (annotation = z-scored feature mean; color = row z)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    out = FIG / "kmeans_cluster_profiles.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_cluster_by_year(df: pd.DataFrame) -> Path:
    # year is column index 1 in scaled space — recover approx year band via scaler
    # Better: use raw year from unscaled companion. Reconstruct from scaler params.
    params = pd.read_csv(PROC / "scaler_params_2018plus.csv")
    mean = params.set_index("feature").loc[FEAT_NAMES, "mean"].to_numpy()
    scale = params.set_index("feature").loc[FEAT_NAMES, "scale"].to_numpy()
    raw = df[FEAT_NAMES].to_numpy() * scale + mean
    years = np.rint(raw[:, 1]).astype(int)
    years = np.clip(years, 2018, 2026)
    tmp = pd.DataFrame({"year": years, "cluster": df["cluster"].to_numpy()})
    yearly = tmp.groupby(["year", "cluster"]).size().rename("n").reset_index()
    totals = yearly.groupby("year")["n"].transform("sum")
    yearly["share"] = yearly["n"] / totals * 100
    wide = yearly.pivot(index="year", columns="cluster", values="share").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    wide.plot.area(ax=ax, stacked=True, alpha=0.9, colormap="Set2")
    ax.set_title("Cluster share over year (market regimes, 2018+ sample)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (%)")
    ax.set_xlim(2018, 2026)
    ax.set_ylim(0, 100)
    ax.legend(title="Cluster", frameon=False, ncol=4)
    fig.tight_layout()
    out = FIG / "kmeans_cluster_share_over_year.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_logprice_vs_type(df: pd.DataFrame) -> Path:
    params = pd.read_csv(PROC / "scaler_params_2018plus.csv")
    mean = params.set_index("feature").loc[FEAT_NAMES, "mean"].to_numpy()
    scale = params.set_index("feature").loc[FEAT_NAMES, "scale"].to_numpy()
    raw = df[FEAT_NAMES].to_numpy() * scale + mean
    plot_df = pd.DataFrame(
        {
            "log_price": raw[:, 0],
            "type_code": np.rint(raw[:, 3]).astype(int),
            "cluster": df["cluster"].to_numpy(),
        }
    )
    plot_df["type"] = plot_df["type_code"].map(TYPE_LABEL).fillna("?")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # subsample for scatter readability
    rng = np.random.RandomState(1)
    take = plot_df.sample(n=min(20000, len(plot_df)), random_state=rng)
    for c, g in take.groupby("cluster"):
        ax.scatter(g["type_code"], g["log_price"], s=8, alpha=0.25, label=f"C{c}")
    ax.set_xticks(list(TYPE_LABEL.keys()))
    ax.set_xticklabels(list(TYPE_LABEL.values()))
    ax.set_title("Clusters in log-price vs property-type space")
    ax.set_xlabel("Property type")
    ax.set_ylabel("log(price)")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    out = FIG / "kmeans_scatter_logprice_vs_type.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# BASIC RUN — Task 2. Fit k-means at the chosen k. Centres map to new-build, leasehold/flat, freehold.
def run_kmeans(X: np.ndarray) -> tuple[list[Path], dict]:
    """BASIC RUN — Task 2. Cluster with the chosen k and save centres for new-build, leasehold flat, and freehold."""
    print("Selecting k ...")
    best_k, metrics = choose_k(X)
    metrics.to_csv(PROC / "kmeans_k_metrics.csv", index=False)

    def fit(k: int, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        km = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        centers = pd.DataFrame(km.cluster_centers_, columns=FEAT_NAMES)
        centers.index.name = "cluster"
        labeled = pd.DataFrame(X, columns=FEAT_NAMES)
        labeled["cluster"] = labels
        labeled.to_csv(PROC / f"uk_sample_with_clusters_{tag}.csv", index=False)
        centers.to_csv(PROC / f"kmeans_cluster_centers_{tag}.csv")
        return labeled, centers

    print(f"Fitting silhouette-best k={best_k} ...")
    df_rep, centers_rep = fit(best_k, f"k{best_k}")
    df_rep.to_csv(PROC / "uk_sample_with_clusters.csv", index=False)
    centers_rep.to_csv(PROC / "kmeans_cluster_centers.csv")

    summary = [
        f"rows_used={len(X)}",
        f"features={FEAT_NAMES}",
        f"silhouette_best_k={best_k}",
        f"report_k={best_k}",
        "",
        f"cluster_sizes k={best_k}:",
        df_rep["cluster"].value_counts().sort_index().to_string(),
        "",
        f"centers k={best_k}:",
        centers_rep.round(3).to_string(),
    ]
    summary_path = PROC / "kmeans_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    figs = [
        plot_elbow_silhouette(metrics),
        plot_cluster_profiles(centers_rep),
        plot_cluster_by_year(df_rep),
        plot_logprice_vs_type(df_rep),
    ]
    return figs, {"best_k": best_k, "report_k": best_k, "n": len(X), "summary": summary_path}


def main() -> None:
    setup_style()
    FIG.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly()
    county = load_county_year()
    trend_figs = [
        plot_median_price_by_type(monthly),
        plot_volume_by_type(monthly),
        plot_national_median(monthly),
        plot_top_counties(county),
    ]

    X = load_features_for_kmeans()
    kmeans_figs, info = run_kmeans(X)

    print("Trend figures:")
    for p in trend_figs:
        print(f"  {p}")
    print("K-means figures:")
    for p in kmeans_figs:
        print(f"  {p}")
    print(
        f"silhouette_best_k={info['best_k']} report_k={info['report_k']} rows={info['n']}"
    )
    print(f"summary: {info['summary']}")


if __name__ == "__main__":
    main()
