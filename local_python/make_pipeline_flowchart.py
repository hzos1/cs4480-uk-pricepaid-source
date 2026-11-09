"""Report-ready pipeline flowchart (PNG 300 dpi + PDF).

Track 1 / Track 2 use vertical stacks: 5→6→output and 7→8→output.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "figures4papers" / "fig0_pipeline_flowchart.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")

INK = "#222222"
MUTED = "#666666"
GREEN = "#2f6b4f"
BLUE = "#1f4e79"
COLORS = {
    "data": ("#f5f5f3", "#444444"),
    "scale": ("#fff6ea", "#a65c1a"),
    "insight": ("#eaf5ec", GREEN),
    "par": ("#eaf0f7", BLUE),
}
OFF = 0.06


def make_flowchart() -> Path:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def lane(x, y, w, h, fc, ec):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                lw=1.0,
                fc=fc,
                ec=ec,
                zorder=1,
            )
        )

    def box(x, y, w, h, title, lines, kind):
        fc, ec = COLORS[kind]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                lw=1.4,
                fc=fc,
                ec=ec,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.16,
            title,
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + 0.34,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=6.8,
            color=MUTED,
            linespacing=1.22,
            zorder=3,
        )
        return {
            "L": (x - OFF, y + h / 2),
            "R": (x + w + OFF, y + h / 2),
            "T": (x + w / 2, y + h + OFF),
            "B": (x + w / 2, y - OFF),
            "cx": x + w / 2,
        }

    def arrow(a, b, color=INK, head_len=0.17, head_half_w=0.085, lw=1.55):
        """Shaft + filled triangle — complete arrow even on short spans."""
        x1, y1 = a
        x2, y2 = b
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        hl = min(head_len, max(0.11, length * 0.42))
        hw = head_half_w * (hl / head_len)
        tip = (x2, y2)
        base = (x2 - hl * ux, y2 - hl * uy)
        px, py = -uy, ux
        left = (base[0] + hw * px, base[1] + hw * py)
        right = (base[0] - hw * px, base[1] - hw * py)
        ax.plot([x1, base[0]], [y1, base[1]], color=color, lw=lw, zorder=4, solid_capstyle="butt")
        ax.add_patch(
            Polygon([tip, left, right], closed=True, facecolor=color, edgecolor=color, lw=0, zorder=5)
        )

    def feed(src_bottom, dst_top, color, rail_y):
        """Orthogonal feed: down to a shared rail ABOVE the lanes, across, then arrow down."""
        x0, y0 = src_bottom
        x1, y1 = dst_top
        ax.plot(
            [x0, x0, x1],
            [y0, rail_y, rail_y],
            color=color,
            lw=1.55,
            zorder=4,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        arrow((x1, rail_y), (x1, y1), color, head_len=0.18, head_half_w=0.09)

    # ---- Title ----
    ax.text(
        5.75,
        7.38,
        "UK Price Paid — end-to-end project pipeline",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(0.35, 7.12, "Stage A — Data preparation", fontsize=8.4, fontweight="bold", color="#444444")

    # ---- Stage A ----
    bw, bh = 2.20, 0.88
    ys = 6.05
    xs = [0.45, 3.15, 5.85, 8.55]
    b1 = box(xs[0], ys, bw, bh, "1  Raw register", ["HM Land Registry Price Paid", "~3.1 GB CSV (unchanged)"], "data")
    b2 = box(xs[1], ys, bw, bh, "2  Clean panel", ["monthly tables · type codes"], "data")
    b3 = box(xs[2], ys, bw, bh, "3  Feature matrix", ["6D z-scored · 2018+", "~8.55M points"], "data")
    b4 = box(xs[3], ys, bw, bh, "4  Scale-up", ["replicate features", "~17.1M / ~25.7M"], "data")

    arrow(b1["R"], b2["L"])
    arrow(b2["R"], b3["L"])
    arrow(b3["R"], b4["L"])

    # ---- Track lanes (tighter under Stage A; short rail in the remaining gap) ----
    lane_top = 5.52
    lane(0.30, 0.28, 5.30, lane_top - 0.28, "#f4faf6", "#c5d9cc")
    lane(5.90, 0.28, 5.30, lane_top - 0.28, "#f3f6fb", "#c5d0e0")
    ax.text(0.45, 5.28, "Track 1 — Market insight", ha="left", fontsize=9.0, fontweight="bold", color=GREEN, zorder=2)
    ax.text(0.45, 5.04, "Local analytics for report figures", ha="left", fontsize=7.0, color=MUTED, zorder=2)
    ax.text(6.05, 5.28, "Track 2 — Parallel evaluation", ha="left", fontsize=9.0, fontweight="bold", color=BLUE, zorder=2)
    ax.text(6.05, 5.04, "Shared-init Lloyd k-means on Hadoop VM", ha="left", fontsize=7.0, color=MUTED, zorder=2)

    # Vertical stacks: sit under titles, equal gaps, room for arrow shafts
    tw, th = 3.60, 0.80
    gap = 0.38
    y5 = 3.95
    y6 = y5 - gap - th
    y_out = y6 - gap - th

    x1 = 0.30 + (5.30 - tw) / 2
    b5 = box(x1, y5, tw, th, "5  Trends", ["median price & volume", "by type / region"], "insight")
    b6 = box(x1, y6, tw, th, "6  k-means (k=4)", ["elbow + silhouette", "named cluster profiles"], "insight")
    o1 = box(x1, y_out, tw, th, "Output — insight figures", ["trends + interpretable segments"], "insight")

    x2 = 5.90 + (5.30 - tw) / 2
    b7 = box(x2, y5, tw, th, "7  Engines", ["Python ProcessPool · Spark 1.2.2", "MapReduce Streaming"], "par")
    b8 = box(x2, y6, tw, th, "8  Scale sweep", ["0.5M → 25.7M", "wall time & throughput"], "par")
    o2 = box(x2, y_out, tw, th, "Output — parallel evidence", ["python vs spark vs MR graph"], "par")

    # Shared rail in the short white gap between Stage A and lanes
    rail_y = 5.72
    feed(b3["B"], b5["T"], GREEN, rail_y)
    feed(b4["B"], b7["T"], BLUE, rail_y)

    # Vertical chains — attach just outside boxes; gap is large enough for full shafts
    arrow(b5["B"], b6["T"], GREEN, head_len=0.16, head_half_w=0.08)
    arrow(b6["B"], o1["T"], GREEN, head_len=0.16, head_half_w=0.08)
    arrow(b7["B"], b8["T"], BLUE, head_len=0.16, head_half_w=0.08)
    arrow(b8["B"], o2["T"], BLUE, head_len=0.16, head_half_w=0.08)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    return OUT_PNG


if __name__ == "__main__":
    make_flowchart()
