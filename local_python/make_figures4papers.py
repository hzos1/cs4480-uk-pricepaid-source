"""
Publication-ready figures for the CS4480 report.
Writes ONLY to figures4papers/ (does not modify raw CSVs or figures/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "figures4papers"

# HM Land Registry Price Paid property type codes
TYPE_FULL = {
    "D": "Detached",
    "S": "Semi-detached",
    "T": "Terraced",
    "F": "Flat / maisonette",
    "O": "Other",
}
TYPE_CODE_FULL = {
    0: "Detached",
    1: "Semi-detached",
    2: "Terraced",
    3: "Flat / maisonette",
    4: "Other",
}
FEAT_DISPLAY = {
    "log_price": "log(price)",
    "year": "Year",
    "month": "Month",
    "type_code": "Property type",
    "is_new": "New build",
    "is_freehold": "Freehold",
}
FEAT_NAMES = list(FEAT_DISPLAY.keys())
DPI = 300
TYPE_ORDER = ["D", "S", "T", "F"]  # exclude Other from main trends
# Short labels from k=4 centres (z-scored): freehold seasonality, new-build, leasehold/flats
def cluster_label(i: int, center: pd.Series | None = None) -> str:
    """Name a cluster from its centre. Month, not year, is the seasonal split."""
    if center is None:
        return f"Cluster {i}"
    if float(center.get("is_new", 0)) > 1.5:
        return f"{i}: New-build"
    if float(center.get("is_freehold", 0)) < -1.0 or float(center.get("type_code", 0)) > 0.8:
        return f"{i}: Leasehold / flats"
    month = float(center.get("month", 0))
    if month <= -0.4:
        return f"{i}: Freehold, early months"
    if month >= 0.4:
        return f"{i}: Freehold, late months"
    return f"{i}: Freehold, mid-year"


def load_cluster_names() -> dict[int, str]:
    path = PROC / "kmeans_cluster_centers.csv"
    if not path.exists():
        return {}
    centers = pd.read_csv(path, index_col=0)
    return {int(i): cluster_label(int(i), centers.loc[i]) for i in centers.index}


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "font.family": "DejaVu Sans",
            # No default grey grid — it cuts through heatmaps / log plots and looks noisy in print
            "axes.grid": False,
            "lines.linewidth": 1.8,
            "savefig.dpi": DPI,
            # Do NOT use bbox_inches='tight' — it pulls footnotes back onto axis labels
            "savefig.bbox": None,
            "savefig.pad_inches": 0.05,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches=None, facecolor="white", edgecolor="none")
    fig.savefig(pdf, bbox_inches=None, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  wrote {png.name} / {pdf.name}")
    return png


def paper_legend(ax: plt.Axes, **kwargs):
    """Legend with solid white fill and black edge — no grey frame cutting the plot."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fancybox", False)
    kwargs.setdefault("edgecolor", "#222222")
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("framealpha", 1.0)
    return ax.legend(**kwargs)


def paper_footnotes(fig: plt.Figure, lines: list[str], *, bottom: float | None = None) -> None:
    """Black footnotes in figure coords, with reserved bottom margin (no grey/axis overlap)."""
    n = len(lines)
    if bottom is None:
        bottom = 0.16 + 0.08 * max(n, 1)
    for i, line in enumerate(lines):
        # Keep notes inside the reserved band, well below the x-axis label
        y = 0.035 + (n - 1 - i) * 0.07
        fig.text(0.5, y, line, ha="center", va="center", fontsize=8, color="#000000")
    fig.subplots_adjust(bottom=bottom)


def type_legend_label(code: str) -> str:
    return f"{TYPE_FULL.get(code, code)} ({code})"


def load_monthly() -> pd.DataFrame:
    df = pd.read_csv(PROC / "uk_monthly_by_type.csv")
    df["ym"] = pd.to_datetime(df["ym"] + "-01", errors="coerce")
    return df.dropna(subset=["ym"])


def fig_median_price_by_type(monthly: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for code in TYPE_ORDER:
        g = monthly[monthly["property_type"] == code]
        if g.empty:
            continue
        ax.plot(g["ym"], g["median_price"] / 1000.0, label=type_legend_label(code))
    ax.set_title("Median sale price by property type (England & Wales)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Median price (£ thousand)")
    paper_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Property type", fontsize=8)
    fig.subplots_adjust(right=0.72)
    paper_footnotes(
        fig,
        [
            "Source: HM Land Registry Price Paid Data. Codes: D=Detached, S=Semi-detached, T=Terraced, F=Flat/maisonette."
        ],
        bottom=0.18,
    )
    return save(fig, "fig1_median_price_by_type")


def fig_volume_by_type(monthly: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for code in TYPE_ORDER:
        g = monthly[monthly["property_type"] == code]
        if g.empty:
            continue
        ax.plot(g["ym"], g["n"] / 1000.0, label=type_legend_label(code))
    ax.set_title("Monthly transaction volume by property type")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transactions (thousand)")
    paper_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Property type", fontsize=8)
    fig.subplots_adjust(right=0.72)
    paper_footnotes(
        fig,
        ["Source: HM Land Registry Price Paid Data. Other (O) omitted (sparse / extreme values)."],
        bottom=0.18,
    )
    return save(fig, "fig2_volume_by_type")


def fig_national_median(monthly: pd.DataFrame) -> Path:
    sub = monthly[monthly["property_type"].isin(TYPE_ORDER)].copy()
    sub["w"] = sub["median_price"] * sub["n"]
    nat = sub.groupby("ym", as_index=False).agg(n=("n", "sum"), w=("w", "sum"))
    nat["median_approx"] = nat["w"] / nat["n"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(nat["ym"], nat["median_approx"] / 1000.0, color="#1f4e79")
    ax.set_title("Transaction-weighted average of type medians (excl. Other)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (£ thousand)")
    paper_footnotes(
        fig,
        ["Not the median of all sales. Each point weights the D/S/T/F monthly medians by count."],
        bottom=0.18,
    )
    return save(fig, "fig3_national_median_price")


def fig_top_counties(year: int | None = None) -> Path:
    county = pd.read_csv(PROC / "uk_county_year_stats.csv")
    if year is None or county[county["year"] == year].empty:
        year = int(county["year"].max())
        # prefer a complete-looking recent year if available
        for y in (2024, 2023, 2022):
            if not county[county["year"] == y].empty:
                year = y
                break
    ydf = county[(county["year"] == year) & (county["n"] >= 200)].nlargest(15, "median_price").copy()
    # Title-case so long ALL-CAPS names fit; keep full string (no clipping)
    labels = [str(c).strip().title() for c in ydf["county"][::-1]]
    values = ydf["median_price"][::-1] / 1000.0
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(labels, values, color="#4c6a80")
    ax.set_title(f"Top 15 counties / districts by median price ({year})")
    ax.set_xlabel("Median price (£ thousand)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.92, bottom=0.10)
    return save(fig, "fig4_top_counties_median")


def fig_elbow_silhouette() -> Path | None:
    path = PROC / "kmeans_k_metrics.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    # Extra horizontal space so y-labels do not collide between panels
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    axes[0].plot(m["k"], m["inertia"], marker="o", color="#1f4e79")
    axes[0].set_title("(a) Elbow method")
    axes[0].set_xlabel("Number of clusters k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(m["k"].tolist())
    axes[1].plot(m["k"], m["silhouette"], marker="o", color="#b85c38")
    axes[1].set_title("(b) Silhouette score")
    axes[1].set_xlabel("Number of clusters k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_xticks(m["k"].tolist())
    fig.suptitle("Choosing k for k-means (2018+ random sample)", fontsize=12)
    return save(fig, "fig5_kmeans_elbow_silhouette")


def fig_cluster_profiles() -> Path | None:
    path = PROC / "kmeans_cluster_centers.csv"
    if not path.exists():
        path = PROC / "kmeans_cluster_centers_k4.csv"
    if not path.exists():
        return None
    centers = pd.read_csv(path, index_col=0)
    centers = centers[[c for c in FEAT_NAMES if c in centers.columns]]
    row_labels = [cluster_label(int(i), centers.loc[i]) for i in centers.index]
    centers = centers.rename(columns=FEAT_DISPLAY)
    centers.index = row_labels
    z = (centers - centers.mean()) / centers.std(ddof=0).replace(0, 1)
    # Temporarily disable global grid so it cannot cut through the heatmap
    with plt.rc_context({"axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        sns.heatmap(
            z,
            annot=centers.round(2),
            fmt="",
            cmap="coolwarm",
            center=0,
            ax=ax,
            linewidths=0.0,
            linecolor="face",
            square=False,
            cbar_kws={"label": "Row-wise z-score of centre"},
            annot_kws={"size": 10, "color": "#000000"},
        )
        # Kill any residual cell borders seaborn may still draw
        if ax.collections:
            ax.collections[0].set_edgecolor("face")
            ax.collections[0].set_linewidth(0.0)
        ax.grid(False)
        ax.tick_params(axis="both", length=0, colors="#000000", labelsize=8)
        ax.set_title(f"K-means cluster centres (k={len(centers)}), standardised features", pad=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel("Cluster meaning", color="#000000")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#000000")
        # Colorbar axis: kill any grid / grey tick bleed into the strip
        for other in fig.axes:
            if other is ax:
                continue
            other.grid(False)
            # Tick labels outside only — no marks drawn into the colour strip
            other.tick_params(
                colors="#000000",
                length=4,
                width=0.8,
                direction="out",
                which="both",
            )
            for spine in other.spines.values():
                spine.set_color("#222222")
                spine.set_linewidth(0.6)

        ax.set_xlabel("Feature", color="#000000", labelpad=10)
        fig.subplots_adjust(left=0.32, right=0.90, top=0.90)
        paper_footnotes(
            fig,
            [
                "Cell numbers = cluster-centre values in z-scored feature space.",
                "Property type encoding: 0=Detached (D), 1=Semi-detached (S), "
                "2=Terraced (T), 3=Flat/maisonette (F), 4=Other (O).",
            ],
            bottom=0.36,
        )
        return save(fig, "fig6_kmeans_cluster_profiles")

def fig_cluster_share_over_year() -> Path | None:
    path = PROC / "uk_sample_with_clusters.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    params = pd.read_csv(PROC / "scaler_params_2018plus.csv")
    mean = params.set_index("feature").loc[FEAT_NAMES, "mean"].to_numpy()
    scale = params.set_index("feature").loc[FEAT_NAMES, "scale"].to_numpy()
    raw = df[FEAT_NAMES].to_numpy() * scale + mean
    years = np.clip(np.rint(raw[:, 1]).astype(int), 2018, 2026)
    tmp = pd.DataFrame({"year": years, "cluster": df["cluster"].to_numpy()})
    yearly = tmp.groupby(["year", "cluster"]).size().rename("n").reset_index()
    totals = yearly.groupby("year")["n"].transform("sum")
    yearly["share"] = yearly["n"] / totals * 100
    wide = yearly.pivot(index="year", columns="cluster", values="share").fillna(0)
    names = load_cluster_names()
    wide = wide.rename(columns={c: names.get(int(c), f"Cluster {c}") for c in wide.columns})
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    wide.plot.area(ax=ax, stacked=True, alpha=0.9, colormap="Set2")
    ax.set_title("Cluster membership share by year (2018–2026 sample)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (%)")
    ax.set_xlim(2018, 2026)
    ax.set_ylim(0, 100)
    paper_legend(ax, title="Cluster meaning", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.68, top=0.90)
    paper_footnotes(
        fig,
        [
            "Based on a random sample of standardised transactions (2018+). Recent years may be incomplete."
        ],
        bottom=0.18,
    )
    return save(fig, "fig7_kmeans_cluster_share_over_year")


def fig_logprice_vs_type() -> Path | None:
    path = PROC / "uk_sample_with_clusters.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    params = pd.read_csv(PROC / "scaler_params_2018plus.csv")
    mean = params.set_index("feature").loc[FEAT_NAMES, "mean"].to_numpy()
    scale = params.set_index("feature").loc[FEAT_NAMES, "scale"].to_numpy()
    raw = df[FEAT_NAMES].to_numpy() * scale + mean
    plot_df = pd.DataFrame(
        {
            "log_price": raw[:, 0],
            "type_code": np.clip(np.rint(raw[:, 3]).astype(int), 0, 4),
            "cluster": df["cluster"].to_numpy(),
        }
    )
    rng = np.random.RandomState(1)
    take = plot_df.sample(n=min(15000, len(plot_df)), random_state=rng)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for c in sorted(take["cluster"].unique()):
        g = take[take["cluster"] == c]
        ax.scatter(
            g["type_code"] + rng.normal(0, 0.06, size=len(g)),
            g["log_price"],
            s=6,
            alpha=0.22,
            label=load_cluster_names().get(int(c), f"Cluster {c}"),
        )
    ax.set_xticks(list(TYPE_CODE_FULL.keys()))
    ax.set_xticklabels(
        [f"{TYPE_CODE_FULL[i]}\n({['D','S','T','F','O'][i]})" for i in TYPE_CODE_FULL],
        fontsize=8,
    )
    ax.set_title("Clusters in log-price vs property-type space")
    ax.set_xlabel("Property type (Land Registry code)")
    ax.set_ylabel("log(price)")
    ymin = float(take["log_price"].min())
    ymax = float(take["log_price"].max())
    pad = max(0.4, (ymax - ymin) * 0.06)
    ax.set_ylim(ymin - pad, ymax + pad)
    paper_legend(
        ax,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7,
        title="Cluster meaning",
    )
    fig.subplots_adjust(left=0.10, right=0.66, top=0.90)
    paper_footnotes(
        fig,
        [
            "Labels come from the centres. k is the silhouette maximum.",
            "D=Detached, S=Semi-detached, T=Terraced, F=Flat/maisonette, O=Other. Points jittered for visibility.",
        ],
        bottom=0.28,
    )
    return save(fig, "fig8_kmeans_logprice_vs_type")


# Parallel benchmark (Bitnami Hadoop VM): 10% sample, k=4, 20 iters
PARALLEL_POINTS = 855_193
PARALLEL_ITERS = 20
# Same data/init/k/iters on the Hadoop VM (4 vCPU). Python = ProcessPool + NumPy.
PARALLEL_WALL = {
    "Python": 22.832,
    "Spark": 455.576,
    "MapReduce": 1259.789,
}

# Max centroid shift per iteration (identical trajectory on shared init)
PARALLEL_SHIFT = [
    1.74974965,
    0.28433087,
    0.18292520,
    0.15151351,
    0.20470021,
    0.22116557,
    0.28267981,
    0.39399538,
    0.29738460,
    0.12234053,
    0.06205262,
    0.04464782,
    0.04043436,
    0.03511388,
    0.03245817,
    0.02855381,
    0.02570583,
    0.02219888,
    0.01695673,
    0.01584315,
]


def ensure_parallel_timing_csv() -> Path:
    path = PROC / "parallel_timing_engines.csv"
    rows = []
    for eng, sec in PARALLEL_WALL.items():
        rows.append(
            {
                "engine": eng,
                "points": PARALLEL_POINTS,
                "iters": PARALLEL_ITERS,
                "wall_seconds": sec,
                "wall_minutes": sec / 60.0,
                "points_per_sec": PARALLEL_POINTS * PARALLEL_ITERS / sec,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    shift_path = PROC / "parallel_centroid_shift.csv"
    pd.DataFrame(
        {"iter": np.arange(1, len(PARALLEL_SHIFT) + 1), "max_shift": PARALLEL_SHIFT}
    ).to_csv(shift_path, index=False)
    return path


def fig_parallel_runtime() -> Path:
    ensure_parallel_timing_csv()
    engines = list(PARALLEL_WALL.keys())
    seconds = [PARALLEL_WALL[e] for e in engines]
    minutes = [s / 60.0 for s in seconds]
    throughput = [PARALLEL_POINTS * PARALLEL_ITERS / s for s in seconds]
    colors = ["#2f6b4f", "#1f4e79", "#c47b2b"]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.9))

    ax = axes[0]
    ax.grid(False)
    bars = ax.bar(engines, minutes, color=colors, width=0.65, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title("End-to-end k-means wall time")
    ymax_m = max(minutes) * 1.28
    ax.set_ylim(0, ymax_m)
    for bar, sec, m in zip(bars, seconds, minutes):
        if sec < 60:
            label = f"{sec:.1f} s\n({m:.2f} min)"
        else:
            label = f"{m:.1f} min\n({sec:.0f} s)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + ymax_m * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#000000",
        )

    ax = axes[1]
    ax.grid(False)
    thr_m = [t / 1e6 for t in throughput]
    bars = ax.bar(engines, thr_m, color=colors, width=0.65, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Throughput (million point-iters / s)")
    ax.set_title("Effective throughput")
    ymax = max(thr_m) * 1.35
    ax.set_ylim(0, ymax)
    for bar, t in zip(bars, thr_m):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            t + ymax * 0.03,
            f"{t:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#000000",
        )

    fig.suptitle("Fixed-size comparison (10% sample): framework overhead", fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, wspace=0.32)
    paper_footnotes(
        fig,
        [
            f"Same data/init on VM (n={PARALLEL_POINTS:,}, k=4, {PARALLEL_ITERS} iters, 4 vCPU / 6 GB).",
            "855,193 points are a random 10% of 2018-2026, not the first rows of the file. Engine times are the earlier same-machine VM run.",
        ],
        bottom=0.24,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.24, wspace=0.32)
    return save(fig, "fig9_parallel_runtime")


# Large-n bar chart (same VM); 25.7M = 3x full 2018+ scaled file; 5 iters (scaling sweep)
LARGE_POINTS = 25_655_817
LARGE_ITERS = 5
LARGE_SPARK_SEC = 415.868699131  # Spark ok
LARGE_MR_SEC = 1460.166  # MapReduce Streaming ok (same data/init/k/iters)
# Python ProcessPool+NumPy: OOM on fork at this size


def fig_parallel_runtime_25m() -> Path:
    """fig9-style bars at ~25M points: Python OOM; Spark vs MapReduce measured."""
    spark_min = LARGE_SPARK_SEC / 60.0
    mr_min = LARGE_MR_SEC / 60.0
    spark_thr = LARGE_POINTS * LARGE_ITERS / LARGE_SPARK_SEC / 1e6
    mr_thr = LARGE_POINTS * LARGE_ITERS / LARGE_MR_SEC / 1e6
    engines = ["Python", "Spark", "MapReduce"]
    colors = ["#2f6b4f", "#1f4e79", "#c47b2b"]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.0))

    # --- wall time ---
    ax = axes[0]
    ax.grid(False)
    heights = [0.0, spark_min, mr_min]
    bars = ax.bar(engines, heights, color=colors, width=0.65, edgecolor="#222222", linewidth=0.6)
    # Hatched stub for Python OOM (no measured wall time)
    bars[0].set_hatch("///")
    bars[0].set_facecolor("white")
    bars[0].set_edgecolor("#2f6b4f")
    stub = max(mr_min * 0.06, 0.45)
    bars[0].set_height(stub)
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title("End-to-end k-means wall time")
    ymax_m = mr_min * 1.28
    ax.set_ylim(0, ymax_m)
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        stub + ymax_m * 0.03,
        "OOM\n(cannot fork)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#2f6b4f",
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        spark_min + ymax_m * 0.015,
        f"{spark_min:.1f} min\n({LARGE_SPARK_SEC:.0f} s)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#000000",
    )
    ax.text(
        bars[2].get_x() + bars[2].get_width() / 2,
        mr_min + ymax_m * 0.015,
        f"{mr_min:.1f} min\n({LARGE_MR_SEC:.0f} s)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#000000",
    )

    # --- throughput ---
    ax = axes[1]
    ax.grid(False)
    thr_heights = [0.0, spark_thr, mr_thr]
    bars = ax.bar(engines, thr_heights, color=colors, width=0.65, edgecolor="#222222", linewidth=0.6)
    bars[0].set_hatch("///")
    bars[0].set_facecolor("white")
    bars[0].set_edgecolor("#2f6b4f")
    stub_t = max(spark_thr * 0.06, 0.015)
    bars[0].set_height(stub_t)
    ax.set_ylabel("Throughput (million point-iters / s)")
    ax.set_title("Effective throughput")
    ymax = spark_thr * 1.32
    ax.set_ylim(0, ymax)
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        stub_t + ymax * 0.03,
        "OOM",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#2f6b4f",
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        spark_thr + ymax * 0.02,
        f"{spark_thr:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#000000",
    )
    ax.text(
        bars[2].get_x() + bars[2].get_width() / 2,
        mr_thr + ymax * 0.02,
        f"{mr_thr:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#000000",
    )

    fig.suptitle("Large-scale comparison (~25.7M points) on Hadoop VM", fontsize=12)
    paper_footnotes(
        fig,
        [
            f"n={LARGE_POINTS:,} (3× full 2018+ scaled), k=4, {LARGE_ITERS} iters, 4 vCPU / 6 GB.",
            "Python ProcessPool+NumPy OOMs on fork; Spark 1.2.2 (~416 s) beats MapReduce Streaming (~1460 s).",
        ],
        bottom=0.26,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.26, wspace=0.32)
    return save(fig, "fig12_parallel_runtime_25m")


def fig_parallel_convergence() -> Path:
    ensure_parallel_timing_csv()
    iters = np.arange(1, len(PARALLEL_SHIFT) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.semilogy(
        iters,
        PARALLEL_SHIFT,
        color="#1f4e79",
        marker="o",
        markersize=4,
        label="Max centroid shift",
    )
    ax.grid(False, which="both")
    ax.minorticks_off()
    # No grey ε guideline — note goes in the black footnote instead
    ax.set_xlabel("Iteration", color="#000000", labelpad=8)
    ax.set_ylabel("Max centroid shift (L2)", color="#000000")
    ax.set_title("k-means centroid movement per iteration (shared init)")
    ax.set_xticks(iters[::2])
    ax.tick_params(colors="#000000")
    # Single series — no legend box (avoids grey frame / line through the key)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90)
    paper_footnotes(
        fig,
        [
            "Spark and MapReduce used the same initial centroids and matching shift each iteration.",
            r"Neither reached convergence threshold $\varepsilon=10^{-4}$ within 20 iterations.",
        ],
        bottom=0.32,
    )
    return save(fig, "fig10_parallel_convergence")


def fig_scaling_python_vs_spark() -> Path | None:
    """Scaling sweep: absolute wall time (a) and absolute throughput (b) for all engines."""
    path = PROC / "scaling_python_vs_spark.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    py = df[df["engine"] == "python"].copy()
    sp = df[df["engine"] == "spark"].copy()
    mr = df[df["engine"].isin(["mapreduce", "mr"])].copy()
    py_ok = py[py["status"] == "ok"].sort_values("points").copy()
    sp_ok = sp[sp["status"] == "ok"].sort_values("points").copy()
    mr_ok = mr[mr["status"] == "ok"].sort_values("points").copy()
    py_oom = py[py["status"].astype(str).str.contains("oom", case=False, na=False)].copy()

    def _throughput(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        iters = out["iters"].fillna(5).astype(float)
        out["throughput"] = out["points"].astype(float) * iters / out["wall_seconds"].astype(float) / 1e6
        return out

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))

    # --- (a) wall time vs n ---
    ax = axes[0]
    ax.grid(False)
    if not py_ok.empty:
        ax.plot(
            py_ok["points"] / 1e6,
            py_ok["wall_seconds"],
            marker="o",
            color="#2f6b4f",
            label="Python (4 workers)",
        )
    if not sp_ok.empty:
        ax.plot(
            sp_ok["points"] / 1e6,
            sp_ok["wall_seconds"],
            marker="s",
            color="#1f4e79",
            label="Spark",
        )
    if not mr_ok.empty:
        ax.plot(
            mr_ok["points"] / 1e6,
            mr_ok["wall_seconds"],
            marker="^",
            color="#c47b2b",
            label="MapReduce",
        )
    # Place OOM × at Spark's wall time for that size (no free-floating labels)
    if not py_oom.empty and not sp_ok.empty:
        oom_x, oom_y = [], []
        for _, row in py_oom.iterrows():
            match = sp_ok[np.isclose(sp_ok["points"], row["points"])]
            if match.empty:
                continue
            oom_x.append(float(row["points"]) / 1e6)
            oom_y.append(float(match["wall_seconds"].iloc[0]))
        if oom_x:
            ax.scatter(
                oom_x,
                oom_y,
                marker="x",
                s=90,
                color="#2f6b4f",
                linewidths=2.0,
                zorder=5,
                label="Python OOM",
            )
    y_candidates = []
    for part in (sp_ok, mr_ok, py_ok):
        if not part.empty:
            y_candidates.append(float(np.nanmax(part["wall_seconds"].to_numpy())))
    if y_candidates:
        ax.set_ylim(0, max(y_candidates) * 1.12)
    x_candidates = []
    for part in (sp_ok, py_ok, py_oom, mr_ok):
        if not part.empty:
            x_candidates.append(float(part["points"].max()) / 1e6)
    if x_candidates:
        ax.set_xlim(0, max(x_candidates) * 1.05)
    ax.set_xlabel("Data size (million points)")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("(a) Runtime vs data size")

    # --- (b) absolute throughput (each engine on its own measured sizes) ---
    ax = axes[1]
    ax.grid(False)
    thr_parts = []
    for part, color, marker, label in (
        (py_ok, "#2f6b4f", "o", "Python (4 workers)"),
        (sp_ok, "#1f4e79", "s", "Spark"),
        (mr_ok, "#c47b2b", "^", "MapReduce"),
    ):
        if part.empty:
            continue
        thr = _throughput(part)
        thr_parts.append(thr)
        ax.plot(
            thr["points"] / 1e6,
            thr["throughput"],
            marker=marker,
            color=color,
            label=label,
        )
    # Mark Python OOM sizes on the x-axis of throughput (no fake y value)
    if not py_oom.empty:
        for _, row in py_oom.iterrows():
            ax.axvline(
                float(row["points"]) / 1e6,
                color="#2f6b4f",
                linestyle=":",
                linewidth=1.0,
                alpha=0.55,
            )
        ax.plot([], [], color="#2f6b4f", linestyle=":", linewidth=1.0, label="Python OOM size")

    y_thr = []
    for thr in thr_parts:
        y_thr.append(float(np.nanmax(thr["throughput"].to_numpy())))
    if y_thr:
        ax.set_ylim(0, max(y_thr) * 1.12)
    if x_candidates:
        ax.set_xlim(0, max(x_candidates) * 1.05)
    ax.set_xlabel("Data size (million points)")
    ax.set_ylabel("Throughput (million point-iters / s)")
    ax.set_title("(b) Absolute throughput vs scale")
    handles, labels = axes[0].get_legend_handles_labels()
    extra_h, extra_l = axes[1].get_legend_handles_labels()
    for h, lab in zip(extra_h, extra_l):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4,
        frameon=True, fancybox=False, edgecolor="#222222", facecolor="white", fontsize=7,
    )
    fig.suptitle("Why Spark suits larger data on the Hadoop VM", fontsize=12, y=0.98)
    n_mr = int(mr_ok["points"].nunique()) if not mr_ok.empty else 0
    paper_footnotes(
        fig,
        [
            "Same Lloyd k-means (k=4, 5 iters, shared init) on the Hadoop VM (4 vCPU / 6 GB).",
            f"Both panels use absolute metrics; MapReduce timed at {n_mr} scale(s). Python OOMs beyond ~17M.",
        ],
        bottom=0.28,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.22, wspace=0.32)
    return save(fig, "fig11_scaling_python_vs_spark")


def _rule_pretty_item(item: str) -> str:
    if item.startswith("type="):
        code = item.split("=", 1)[1]
        # Short labels keep the bar panel centered (full names make left margin huge)
        short = {
            "D": "Detached",
            "S": "Semi-detached",
            "T": "Terraced",
            "F": "Flat",
            "O": "Other",
        }
        return short.get(code, TYPE_FULL.get(code, code))
    mapping = {
        "new": "New-build",
        "resale": "Resale",
        "freehold": "Freehold",
        "leasehold": "Leasehold",
        "price=Low": "Low price",
        "price=Mid": "Mid price",
        "price=High": "High price",
    }
    return mapping.get(item, item)


def _rule_pretty_side(side: str) -> str:
    return " + ".join(_rule_pretty_item(x.strip()) for x in side.split("&"))


def _format_rule_label(items: frozenset[str], has_both: bool) -> str:
    """Type on the left; price (and other attrs) on the right."""
    types = sorted(x for x in items if x.startswith("type="))
    prices = sorted(x for x in items if x.startswith("price="))
    other = sorted(
        x for x in items if not x.startswith("type=") and not x.startswith("price=")
    )
    arrow = " ↔ " if has_both else " → "
    if types and prices:
        left = _rule_pretty_side(" & ".join(types))
        right = _rule_pretty_side(" & ".join(other + prices))
        return f"{left}{arrow}{right}"
    if prices and other:
        left = _rule_pretty_side(" & ".join(other))
        right = _rule_pretty_side(" & ".join(prices))
        return f"{left}{arrow}{right}"
    # Fallback: no price / no type
    parts = sorted(items)
    mid = max(1, len(parts) // 2)
    left = _rule_pretty_side(" & ".join(parts[:mid]))
    right = _rule_pretty_side(" & ".join(parts[mid:]))
    return f"{left}{arrow}{right}"


def _association_display_rows(rules: pd.DataFrame, top_n: int = 10) -> list[tuple[str, float]]:
    """Collapse reverse pairs to ↔; Flat↔Leasehold+High price; price kept on the right."""
    rows: list[dict] = []
    for _, row in rules.iterrows():
        items = frozenset(
            x.strip()
            for side in (row["antecedent"], row["consequent"])
            for x in side.split("&")
        )
        if any(x.startswith("period=") for x in items):
            continue
        if "type=F" in items and "leasehold" in items:
            continue
        rows.append(
            {
                "items": items,
                "ant": row["antecedent"],
                "cons": row["consequent"],
                "lift": float(row["lift"]),
            }
        )
    if not rows:
        return []

    display: list[tuple[str, float]] = []

    # One highlighted Flat ↔ Leasehold + High price
    flat_hi = [
        r
        for r in rows
        if r["items"] == frozenset({"type=F", "leasehold", "price=High"})
    ]
    if flat_hi:
        best = max(flat_hi, key=lambda r: r["lift"])
        display.append(("Flat ↔ Leasehold + High price", best["lift"]))

    # Remaining: skip all Flat↔Leasehold variants; collapse A↔B by undirected itemset
    seen_itemsets: set[frozenset[str]] = set()
    if flat_hi:
        seen_itemsets.add(frozenset({"type=F", "leasehold", "price=High"}))

    by_items: dict[frozenset[str], list[dict]] = {}
    for r in rows:
        if "type=F" in r["items"] and "leasehold" in r["items"]:
            continue  # already covered / redundant
        by_items.setdefault(r["items"], []).append(r)

    ranked = sorted(
        by_items.items(),
        key=lambda kv: max(x["lift"] for x in kv[1]),
        reverse=True,
    )
    for items, group in ranked:
        if items in seen_itemsets:
            continue
        seen_itemsets.add(items)
        best = max(group, key=lambda r: r["lift"])
        dirs = {(g["ant"], g["cons"]) for g in group}
        has_both = any((c, a) in dirs for a, c in dirs)
        display.append((_format_rule_label(items, has_both), best["lift"]))
        if len(display) >= top_n:
            break
    return display


def fig_association_rules_top_lift() -> Path | None:
    """Top association rules by lift (from analyze_rule_mining.py)."""
    path = PROC / "association_rules.csv"
    if not path.exists():
        print(f"  skip fig13 (missing {path.name}; run scripts/analyze_rule_mining.py)")
        return None
    rules = pd.read_csv(path)
    if rules.empty:
        print("  skip fig13 (no rules)")
        return None
    display = _association_display_rows(rules, top_n=10)
    if not display:
        print("  skip fig13 (no display rows)")
        return None
    # Plot lowest lift at bottom
    display = list(reversed(display))
    labels = [lab for lab, _ in display]
    lifts = [lift for _, lift in display]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.barh(range(len(lifts)), lifts, color="#1f4e79", height=0.72)
    ax.set_yticks(range(len(lifts)))
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.set_xlabel("Lift", labelpad=8)
    ax.set_title("Association", pad=10)
    xmax = max(lifts)
    pad = 0.08
    for i, lift in enumerate(lifts):
        ax.text(
            float(lift) + pad,
            i,
            f"{lift:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#222222",
            clip_on=False,
        )
    ax.set_xlim(0, xmax + pad + 0.55)
    # Balanced side margins so the bar panel does not sit too far right
    fig.subplots_adjust(left=0.36, right=0.92, top=0.90, bottom=0.24)
    ax.xaxis.set_label_coords(0.5, -0.10)
    fig.text(
        0.5,
        0.095,
        "Apriori on ~500k sales from 2018 onward (excl. Other).",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#000000",
    )
    fig.text(
        0.5,
        0.040,
        "Items: property type, freehold/leasehold, new/resale, price band.  "
        "support≥0.05, conf≥0.45, lift≥1.15.  ↔ = both directions.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#000000",
    )
    return save(fig, "fig13_association_rules_top_lift")


def write_caption_sheet() -> Path:
    text = """# Figure captions (for report)

Property type codes (HM Land Registry Price Paid Data):
- D = Detached
- S = Semi-detached
- T = Terraced
- F = Flat / maisonette
- O = Other

fig1_median_price_by_type — Median sale price (£ thousand) by property type over time.
fig2_volume_by_type — Monthly transaction counts by property type.
fig3_national_median_price — Approximate national median from D/S/T/F type medians.
fig4_top_counties_median — Highest-median counties/districts in a recent year.
fig5_kmeans_elbow_silhouette — Inertia and silhouette for choosing k.
fig6_kmeans_cluster_profiles — k=4 cluster centres in standardised feature space; row labels name each cluster.
fig7_kmeans_cluster_share_over_year — Cluster share by year (2018–2026 sample); legend names each cluster.
fig8_kmeans_logprice_vs_type — Sample points by cluster in log-price vs type space; legend names each cluster (freehold season, new-build, leasehold/flats).
fig9_parallel_runtime — Fixed-size (10% sample) Python vs Spark vs MapReduce: NumPy wins here due to framework overhead; see fig11–12 for scale-up.
fig10_parallel_convergence — Max centroid L2 shift vs iteration (shared init; Spark and MR match).
fig11_scaling_python_vs_spark — Scaling sweep: absolute wall time (a) and absolute throughput (b) for Python / Spark / MapReduce.
fig12_parallel_runtime_25m — fig9-style bars at ~25.7M points: Python OOM; Spark (~416 s) vs MapReduce (~1460 s).
fig13_association_rules_top_lift — Association: top Apriori patterns by lift (↔ = both directions; Flat+High price↔Leasehold collapsed from reverse pairs).

All PNGs are 300 dpi; PDFs are vector copies for Word/LaTeX.
"""
    path = OUT / "CAPTIONS.md"
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.name}")
    return path


def main() -> None:
    setup_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing paper figures to {OUT}")
    monthly = load_monthly()
    fig_median_price_by_type(monthly)
    fig_volume_by_type(monthly)
    fig_national_median(monthly)
    fig_top_counties()
    fig_elbow_silhouette()
    fig_cluster_profiles()
    fig_cluster_share_over_year()
    fig_logprice_vs_type()
    fig_parallel_runtime()
    fig_parallel_convergence()
    fig_scaling_python_vs_spark()
    fig_parallel_runtime_25m()
    fig_association_rules_top_lift()
    write_caption_sheet()
    print("Done.")


if __name__ == "__main__":
    main()
