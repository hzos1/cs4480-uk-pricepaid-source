"""Build a reference Final Report PDF (structure like past CS4480 samples).

Includes: cover, intro + impact, related work, pipeline flowchart,
dataset/method, selected figures, parallel results, discussion, conclusion,
ICS placeholder.

Outputs:
  figures4papers/fig0_pipeline_flowchart.png
  reports/CS4480_UK_PricePaid_Report_DRAFT_reference.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures4papers"
OUT_DIR = ROOT / "reports"
FLOW = FIG / "fig0_pipeline_flowchart.png"
PDF = OUT_DIR / "CS4480_UK_PricePaid_Report_DRAFT_reference.pdf"


def make_flowchart() -> Path:
    """Delegate to dedicated report-ready flowchart script."""
    from make_pipeline_flowchart import make_flowchart as _make

    return _make()


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_pdf() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    cover_sub = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
        textColor=colors.HexColor("#1f4e79"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        spaceBefore=9,
        spaceAfter=5,
        textColor=colors.HexColor("#2f6b4f"),
    )
    body = ParagraphStyle(
        "BodyJust",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=8,
        spaceAfter=2,
    )
    caption = ParagraphStyle(
        "Cap",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceBefore=3,
        spaceAfter=10,
    )
    note = ParagraphStyle(
        "Note",
        parent=body,
        fontSize=8.5,
        textColor=colors.HexColor("#666666"),
        alignment=TA_LEFT,
    )

    story = []

    # --- Cover ---
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph(esc("CITY UNIVERSITY OF HONG KONG"), cover_sub))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            esc("UK House Price Paid Data:\nTrends, Interpretable Clustering, and Parallel k-means"),
            cover_title,
        )
    )
    story.append(
        Paragraph(
            esc("CS4480 Data-Intensive Computing — Final Report (DRAFT / REFERENCE)"),
            cover_sub,
        )
    )
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        Paragraph(
            esc(
                "This PDF is a structural draft for team editing. "
                "Replace placeholders (names, IDs, contribution %), tighten wording, "
                "and keep or drop figures as needed for the final submission."
            ),
            note,
        )
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(esc("Group members (fill in):"), cover_sub))
    story.append(Paragraph(esc("Name A (SID) · Name B (SID) · Name C (SID) · Name D (SID)"), cover_sub))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(esc("November 2026"), cover_sub))
    story.append(PageBreak())

    # --- TOC-ish ---
    story.append(Paragraph(esc("Contents (suggested)"), h1))
    for line in [
        "1. Introduction",
        "2. Related Work",
        "3. Dataset and Preprocessing",
        "4. System / Analysis Pipeline",
        "5. Market Trends and k-means Analysis",
        "6. Parallel Computing Design and Evaluation",
        "7. Discussion",
        "8. Conclusion and Limitations",
        "9. Individual Contribution Statement (placeholder)",
        "References",
    ]:
        story.append(Paragraph(esc(line), bullet))
    story.append(PageBreak())

    # --- 1 Intro ---
    story.append(Paragraph(esc("1. Introduction"), h1))
    story.append(Paragraph(esc("1.1 Background and problem"), h2))
    story.append(
        Paragraph(
            esc(
                "Completed residential property sales in England and Wales are recorded in the "
                "HM Land Registry Price Paid Data. The full extract is large by student-project "
                "standards (on the order of ~3 GB for the complete CSV) and mixes price, date, "
                "location, property type, new/old flag, and tenure. Turning this register into "
                "interpretable market structure—and doing so with tools that still work when the "
                "feature matrix no longer fits comfortably in one process—is a natural "
                "data-intensive computing exercise."
            ),
            body,
        )
    )
    story.append(Paragraph(esc("1.2 Objectives"), h2))
    story.append(
        ListFlowable(
            [
                ListItem(
                    Paragraph(
                        esc(
                            "Describe national trends in median price and transaction volume by property type "
                            "(D/S/T/F/O) and highlight high-median areas."
                        ),
                        bullet,
                    )
                ),
                ListItem(
                    Paragraph(
                        esc(
                            "Cluster post-2018 transactions with Lloyd k-means (k=4) on standardised features "
                            "and interpret clusters in domain language."
                        ),
                        bullet,
                    )
                ),
                ListItem(
                    Paragraph(
                        esc(
                            "Compare Python (ProcessPool+NumPy), Apache Spark, and Hadoop Streaming MapReduce "
                            "for the same iterative k-means under shared initial centroids, including a scale-up "
                            "sweep to tens of millions of points."
                        ),
                        bullet,
                    )
                ),
            ],
            bulletType="bullet",
            start="•",
        )
    )
    story.append(Paragraph(esc("1.3 Real-world impact and creativity"), h2))
    story.append(
        Paragraph(
            esc(
                "Stakeholders such as buyers/sellers, mortgage and valuation desks, and local observers "
                "often need relative market context rather than a single national median. Type-level "
                "price/volume trends support monitoring of structural shifts; interpretable clusters "
                "(e.g., new-build vs leasehold/flats vs seasonal freehold segments) provide a compact "
                "segmentation of completed sales. We do not claim automated valuation: Price Paid "
                "lacks floor area and many listing attributes. Creativity in this project sits in (i) "
                "pairing official national register analytics with explainable clustering, and (ii) "
                "telling a complete parallel story—framework overhead at small n, Python OOM at large n, "
                "and Spark vs Streaming MapReduce wall time/throughput on a student VM."
            ),
            body,
        )
    )

    # --- 2 Related work ---
    story.append(Paragraph(esc("2. Related Work"), h1))
    story.append(
        Paragraph(
            esc(
                "This section is a short literature/tooling backdrop suitable for a CS4480 report. "
                "Expand citations with formal bibliography entries before final submission."
            ),
            note,
        )
    )
    story.append(Paragraph(esc("2.1 Open property registers and housing analytics"), h2))
    story.append(
        Paragraph(
            esc(
                "Public sale registers are widely used for descriptive housing research and policy "
                "dashboards. Official statistics bodies and academic papers typically report medians "
                "or indices by region and dwelling type, and study volume shocks around credit or "
                "tax regime changes. Our trend figures follow that descriptive tradition, but stay "
                "close to the raw Price Paid fields rather than constructing a full repeat-sales index."
            ),
            body,
        )
    )
    story.append(Paragraph(esc("2.2 Clustering for market segmentation"), h2))
    story.append(
        Paragraph(
            esc(
                "Unsupervised clustering (k-means and mixtures) is a standard way to segment "
                "heterogeneous transaction or listing data when labels are absent. Prior work often "
                "combines price with location and physical attributes; our feature set is constrained "
                "to Price Paid columns (log price, calendar fields, type code, new flag, freehold), "
                "which makes cluster interpretation dependent on those axes only. Elbow and silhouette "
                "diagnostics remain the usual practical guides for choosing k in course-scale studies."
            ),
            body,
        )
    )
    story.append(Paragraph(esc("2.3 Parallel iterative algorithms: MapReduce and Spark"), h2))
    story.append(
        Paragraph(
            esc(
                "MapReduce popularised scalable batch analytics on commodity clusters, but iterative "
                "algorithms such as Lloyd k-means pay repeated job-launch and materialisation costs "
                "when each iteration is a separate MapReduce job (as in Hadoop Streaming). Spark’s "
                "RDD/DataFrame model and in-memory reuse were motivated partly by iterative ML and "
                "graph workloads. Empirical comparisons commonly show Spark faster than classic "
                "Hadoop MapReduce for multi-iteration jobs when RAM suffices, while single-node "
                "NumPy/BLAS remains hard to beat on small matrices. Our experiments reproduce that "
                "qualitative pattern on one Bitnami Hadoop VM: Python wins at ~0.85M points; beyond "
                "~17M points ProcessPool fork OOMs; Spark finishes and is substantially faster than "
                "Streaming MapReduce at ~25.7M points."
            ),
            body,
        )
    )
    story.append(Paragraph(esc("2.4 Positioning"), h2))
    story.append(
        Paragraph(
            esc(
                "Relative to past CS4480 samples (image mosaic MapReduce/Spark pipelines; TripAdvisor "
                "TF–IDF MapReduce), this project emphasises (a) a national structured register, "
                "(b) end-to-end analytics figures for the report, and (c) a three-engine k-means "
                "timing narrative including scale-up and failure modes—not only a single MapReduce job."
            ),
            body,
        )
    )

    # --- 3 Dataset ---
    story.append(Paragraph(esc("3. Dataset and Preprocessing"), h1))
    story.append(
        Paragraph(
            esc(
                "Source: HM Land Registry Price Paid (England & Wales). Raw complete extract "
                "pp-complete.csv is about 3.1 GB. We keep originals untouched and write cleaned "
                "panels and feature files under data/. Property type codes: D=Detached, "
                "S=Semi-detached, T=Terraced, F=Flat/maisonette, O=Other. Parallel experiments "
                "use six z-scored features (log_price, year, month, type_code, is_new, is_freehold) "
                "on 2018+ rows (~8.55M), with optional replication to ~17.1M and ~25.7M points for "
                "capacity/scaling tests (same schema; larger n is for parallel evaluation)."
            ),
            body,
        )
    )

    # --- 4 Pipeline + flowchart ---
    story.append(Paragraph(esc("4. System / Analysis Pipeline"), h1))
    story.append(
        Paragraph(
            esc(
                "Figure 0 summarises the dual path: local trend and k-means for the market figures, and "
                "the same Lloyd and Apriori steps timed on goodpc, including the 12-piece stitch."
            ),
            body,
        )
    )
    if FLOW.exists():
        img = Image(str(FLOW), width=6.8 * inch, height=3.5 * inch)
        story.append(img)
        story.append(
            Paragraph(
                esc(
                    "Figure 0. End-to-end pipeline: Price Paid ingest, cleaned panel, scaled features, then "
                    "market figures on the left and, on the right, full-file Lloyd, the 12-piece stitch, and Apriori."
                ),
                caption,
            )
        )

    story.append(
        Paragraph(
            esc(
                "Implementation sketch: scripts/build_uk_panel.py builds the analysis panel; "
                "scripts/analyze_trends_kmeans.py produces trend tables and cluster artefacts; "
                "scripts/make_figures4papers.py emits report figures; parallel/ holds Spark, "
                "Hadoop Streaming, and Python ProcessPool k-means with shared init centroids."
            ),
            body,
        )
    )
    story.append(PageBreak())

    # --- 5 Analysis figures ---
    story.append(Paragraph(esc("5. Market Trends and k-means Analysis (selected figures)"), h1))
    story.append(
        Paragraph(
            esc(
                "Below are representative figures already produced for the report. Captions are "
                "abbreviated; full wording lives in figures4papers/CAPTIONS.md."
            ),
            note,
        )
    )

    def add_fig(name: str, caption_text: str, width=6.3 * inch, height=3.4 * inch):
        path = FIG / name
        if not path.exists():
            story.append(Paragraph(esc(f"[missing {name}]"), note))
            return
        story.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(esc(caption_text), caption)]))

    add_fig(
        "fig1_median_price_by_type.png",
        "Figure 1. Median sale price by property type over time (D/S/T/F).",
        6.2 * inch,
        3.2 * inch,
    )
    add_fig(
        "fig6_kmeans_cluster_profiles.png",
        "Figure 6. k=4 cluster centres in standardised feature space (interpretable row labels).",
        6.2 * inch,
        3.3 * inch,
    )
    add_fig(
        "fig8_kmeans_logprice_vs_type.png",
        "Figure 8. Sample points by cluster in log-price vs type space.",
        6.2 * inch,
        3.3 * inch,
    )
    story.append(PageBreak())

    # --- 6 Parallel ---
    story.append(Paragraph(esc("6. Parallel Computing Design and Evaluation"), h1))
    story.append(Paragraph(esc("6.1 Design"), h2))
    story.append(
        Paragraph(
            esc(
                "All engines run Lloyd k-means with k=4 and identical initial centroids. "
                "Python uses ProcessPool + NumPy; Spark 1.2.2 uses MEMORY_AND_DISK caching with "
                "enlarged driver memory on the VM; MapReduce uses Hadoop Streaming with one job "
                "per iteration. Mid-size bars use a 10% sample (~0.85M) and 20 iterations; the "
                "scaling sweep and large-n bars use 5 iterations."
            ),
            body,
        )
    )
    story.append(Paragraph(esc("6.2 Key quantitative results"), h2))
    data = [
        [Paragraph(esc("<b>Setting</b>"), bullet), Paragraph(esc("<b>Python</b>"), bullet), Paragraph(esc("<b>Spark</b>"), bullet), Paragraph(esc("<b>MapReduce</b>"), bullet)],
        [
            Paragraph(esc("~0.85M, 20 iters"), bullet),
            Paragraph(esc("~23 s"), bullet),
            Paragraph(esc("~456 s"), bullet),
            Paragraph(esc("~1260 s"), bullet),
        ],
        [
            Paragraph(esc("~25.7M, 5 iters"), bullet),
            Paragraph(esc("OOM"), bullet),
            Paragraph(esc("~416 s"), bullet),
            Paragraph(esc("~1460 s"), bullet),
        ],
    ]
    t = Table(data, colWidths=[1.7 * inch, 1.3 * inch, 1.3 * inch, 1.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 8))
    add_fig(
        "fig9_parallel_runtime.png",
        "Figure 9. Fixed mid-size comparison: NumPy wins when framework overhead dominates.",
        6.3 * inch,
        3.1 * inch,
    )
    add_fig(
        "fig11_scaling_python_vs_spark.png",
        "Figure 11. Scaling sweep: absolute wall time and throughput (Python / Spark / MapReduce).",
        6.4 * inch,
        3.3 * inch,
    )
    add_fig(
        "fig12_parallel_runtime_25m.png",
        "Figure 12. Large-n bars (~25.7M): Python OOM; Spark ~3.5× faster than Streaming MapReduce.",
        6.3 * inch,
        3.1 * inch,
    )

    # --- 7 Discussion ---
    story.append(Paragraph(esc("7. Discussion"), h1))
    story.append(
        Paragraph(
            esc(
                "Small-n results caution against judging Spark solely on toy samples: startup and "
                "scheduling dominate. As n grows, Spark’s relative cost falls and its throughput "
                "rises, while MapReduce Streaming remains penalised by per-iteration jobs and HDFS "
                "materialisation. At ≥~17M points the decisive issue is capacity—Python cannot fork "
                "the working set on our 6 GB VM—whereas Spark still completes. Correctness checks "
                "(shared init; matching centroid-shift trajectories for Spark and MapReduce) support "
                "comparing engines rather than divergent algorithms."
            ),
            body,
        )
    )

    # --- 8 Conclusion ---
    story.append(Paragraph(esc("8. Conclusion and Limitations"), h1))
    story.append(
        Paragraph(
            esc(
                "We analysed UK Price Paid sales via trends and interpretable k=4 clustering, and "
                "evaluated parallel Lloyd k-means across Python, Spark, and Hadoop Streaming. "
                "Limitations: Price Paid omits size/amenities; large-n matrices partly use replication "
                "for scaling stress tests; experiments run on a single student VM, not a multi-rack "
                "cluster. Future work could add geography-rich features, compare Spark MLlib "
                "k-means to our hand-rolled loop, or multi-node deployments."
            ),
            body,
        )
    )

    # --- 9 ICS ---
    story.append(Paragraph(esc("9. Individual Contribution Statement (placeholder)"), h1))
    story.append(
        Paragraph(
            esc(
                "Fill before submission. Example table format from past samples:"
            ),
            note,
        )
    )
    ics = [
        [esc("Student Name"), esc("Student ID"), esc("Main contributions"), esc("%")],
        [esc("Member A"), esc("________"), esc("Data cleaning / panel / trends figures"), esc("__")],
        [esc("Member B"), esc("________"), esc("k-means analysis / cluster interpretation"), esc("__")],
        [esc("Member C"), esc("________"), esc("Spark / MapReduce / Python parallel runs"), esc("__")],
        [esc("Member D"), esc("________"), esc("Report writing / figure QA / integration"), esc("__")],
    ]
    t2 = Table(ics, colWidths=[1.3 * inch, 1.0 * inch, 3.2 * inch, 0.5 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf5ea")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t2)

    # --- References ---
    story.append(Paragraph(esc("References (starter list — expand formally)"), h1))
    refs = [
        "HM Land Registry. Price Paid Data. https://www.gov.uk/government/collections/price-paid-data",
        "Dean, J. & Ghemawat, S. MapReduce: Simplified Data Processing on Large Clusters. OSDI / CACM.",
        "Zaharia, M. et al. Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing. NSDI.",
        "Lloyd, S. Least squares quantization in PCM. IEEE Trans. Information Theory (k-means).",
        "Apache Hadoop / Spark documentation (version notes matching the Bitnami VM stack).",
    ]
    for r in refs:
        story.append(Paragraph(esc(f"• {r}"), bullet))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(0.85 * inch, 0.5 * inch, "CS4480 UK Price Paid — DRAFT reference")
        canvas.drawRightString(A4[0] - 0.85 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="CS4480 UK Price Paid Final Report DRAFT",
        author="CS4480 group (draft)",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes)")
    return PDF


def main() -> None:
    make_flowchart()
    build_pdf()


if __name__ == "__main__":
    main()
