# Figure captions (for report)

Property type codes (HM Land Registry Price Paid Data):
- D = Detached
- S = Semi-detached
- T = Terraced
- F = Flat / maisonette
- O = Other

fig1_median_price_by_type — Median sale price (£ thousand) by property type over time.
fig2_volume_by_type — Monthly transaction counts by property type.
fig3_national_median_price — Transaction-weighted average of the D/S/T/F monthly medians. Not the median of all sales. Other is excluded.
fig4_top_counties_median — Highest-median counties/districts in a recent year.
fig5_kmeans_elbow_silhouette — Inertia and silhouette on all 31,702,685 kept sales. The chosen k is 2.
fig6_kmeans_cluster_profiles — Full-file k=2 cluster centres in standardised feature space.
fig7_kmeans_cluster_share_over_year — Cluster share by year (2018–2026 sample); legend names each cluster.
fig8_kmeans_logprice_vs_type — Sample points by cluster in log-price vs type space; legend names each cluster (freehold season, new-build, leasehold/flats).
fig9–fig12 — Removed. Those bars were the old k=4 scaling run and are not this experiment.
fig13_association_rules_top_lift — Association rules on all kept sales, 1995–2026, England and Wales (31,702,685 sales; Other excluded). Top patterns by lift. ↔ means both directions.
fig15_apriori_engines_by_size — Apriori Count Distribution wall-clock seconds on goodpc. Python uses 6 processes. Spark is local[*] and is not YARN. MapReduce is YARN Streaming. Bars exclude Spark JVM startup and exclude the HDFS put.
fig16_kmeans_engines_by_size — Lloyd k-means at k=2 on goodpc, same initial centres, stop when the largest centre movement is at most 1e-4. Python uses 6 workers. Spark is local[*], 12 workers, and is not YARN. MapReduce is YARN Streaming. Wall clock starts after the matrix is loaded. HDFS put is not included.

All PNGs are 300 dpi; PDFs are vector copies for Word/LaTeX.
