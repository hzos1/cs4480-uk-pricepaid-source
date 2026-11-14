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
        "England and Wales Price Paid — project pipeline",
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
    b1 = box(xs[0], ys, bw, bh, "1  Raw register", ["Price Paid CSV, unchanged", "31,702,685 sales"], "data")
    b2 = box(xs[1], ys, bw, bh, "2  Clean panel", ["positive price, valid date", "types D / S / T / F / O"], "data")
    b3 = box(xs[2], ys, bw, bh, "3  Feature matrix", ["6 z-scored fields", "1995–2026, all rows"], "data")
    b4 = box(xs[3], ys, bw, bh, "4  Choose k", ["Python only, k = 2..8", "full file chooses k = 2"], "scale")

    arrow(b1["R"], b2["L"])
    arrow(b2["R"], b3["L"])
    arrow(b3["R"], b4["L"])

    # ---- Track lanes (tighter under Stage A; short rail in the remaining gap) ----
    lane_top = 5.52
    lane(0.30, 0.28, 5.30, lane_top - 0.28, "#f4faf6", "#c5d9cc")
    lane(5.90, 0.28, 5.30, lane_top - 0.28, "#f3f6fb", "#c5d0e0")
    ax.text(0.45, 5.28, "Track 1 — Market insight", ha="left", fontsize=9.0, fontweight="bold", color=GREEN, zorder=2)
    ax.text(0.45, 5.04, "Purpose: what the sales look like", ha="left", fontsize=7.0, color=MUTED, zorder=2)
    ax.text(6.05, 5.28, "Track 2 — Parallel timing", ha="left", fontsize=9.0, fontweight="bold", color=BLUE, zorder=2)
    ax.text(6.05, 5.04, "Same steps, timed because the file is large", ha="left", fontsize=7.0, color=MUTED, zorder=2)

    # Track 1 has four boxes. Track 2 times the cluster step and the rule step.
    tw, th = 3.60, 0.72
    gap = 0.20
    y5 = 4.12
    y6 = y5 - gap - th
    y7 = y6 - gap - th
    y_out = y7 - gap - th

    x1 = 0.30 + (5.30 - tw) / 2
    b5 = box(x1, y5, tw, th, "5  Trends", ["median price and volume", "by type and county"], "insight")
    b6 = box(x1, y6, tw, th, "6  k-means groups", ["two groups of sales", "tenure and property type"], "insight")
    b7 = box(x1, y7, tw, th, "7  Association rules", ["type, tenure, price band", "Apriori on all cleaned sales"], "insight")
    o1 = box(x1, y_out, tw, th, "Output — market figures", ["prices, groups, and rules"], "insight")

    x2 = 5.90 + (5.30 - tw) / 2
    b8 = box(
        x2,
        y5,
        tw,
        th,
        "8  Full-file Lloyd",
        ["Python, Spark, YARN, CUDA, GPU Spark", "86 rounds, same centres, stop at 1e-4"],
        "par",
    )
    b10 = box(
        x2,
        y6,
        tw,
        th,
        "9  12-piece stitch",
        ["Spark, CUDA, and MapReduce", "each piece clusters, then one stitch"],
        "par",
    )
    b9 = box(
        x2,
        y7,
        tw,
        th,
        "10  Time the same Apriori",
        ["Python, Spark, MapReduce", "0.2M rows through the full file"],
        "par",
    )
    o2 = box(
        x2,
        y_out,
        tw,
        th,
        "Output — wall-clock seconds",
        ["sizes, core counts, and the stitch", "one machine, goodpc"],
        "par",
    )

    # Shared rail in the short white gap between Stage A and lanes
    rail_y = 5.72
    feed(b2["B"], b5["T"], GREEN, rail_y)
    # Choose k enters the top-right of the groups box, clear of the trend arrow.
    gutter_x = 5.15
    entry_x = x1 + tw - 0.45
    entry_y = y6 + th
    approach_y = entry_y + 0.16
    x0, y0 = b4["B"]
    ax.plot(
        [x0, x0, gutter_x, gutter_x, entry_x],
        [y0, rail_y, rail_y, approach_y, approach_y],
        color=GREEN,
        lw=1.55,
        zorder=4,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    arrow((entry_x, approach_y), (entry_x, entry_y), GREEN, head_len=0.12, head_half_w=0.07)

    arrow(b5["B"], b6["T"], GREEN, head_len=0.12, head_half_w=0.07)
    arrow(b6["B"], b7["T"], GREEN, head_len=0.12, head_half_w=0.07)
    arrow(b7["B"], o1["T"], GREEN, head_len=0.12, head_half_w=0.07)
    # k-means groups feed the full-file timing, which sits one row higher.
    gutter_x2 = 5.62
    ax.plot(
        [b6["R"][0], gutter_x2, gutter_x2],
        [b6["R"][1], b6["R"][1], b8["L"][1]],
        color=BLUE,
        lw=1.55,
        zorder=4,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    arrow((gutter_x2, b8["L"][1]), b8["L"], BLUE, head_len=0.16, head_half_w=0.08)
    arrow(b8["B"], b10["T"], BLUE, head_len=0.12, head_half_w=0.07)
    arrow(b7["R"], b9["L"], BLUE, head_len=0.16, head_half_w=0.08)
    arrow(b10["B"], b9["T"], BLUE, head_len=0.12, head_half_w=0.07)
    arrow(b9["B"], o2["T"], BLUE, head_len=0.12, head_half_w=0.07)
    ax.text(
        8.55,
        0.55,
        "Memory check: the 2.22 GB matrix does not fit in a 2 GB limit",
        ha="center",
        va="center",
        fontsize=6.6,
        color=MUTED,
        zorder=2,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    return OUT_PNG


if __name__ == "__main__":
    make_flowchart()
