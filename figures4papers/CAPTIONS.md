# Figure captions (for report)

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
