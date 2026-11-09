"""Convert figures4papers/CAPTIONS.md -> CAPTIONS.pdf."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures4papers" / "CAPTIONS.md"
OUT = ROOT / "figures4papers" / "CAPTIONS.pdf"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCap",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        spaceAfter=14,
        leading=20,
    )
    body = ParagraphStyle(
        "BodyCap",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "BulletCap",
        parent=body,
        leftIndent=12,
        spaceAfter=3,
    )
    fig = ParagraphStyle(
        "FigCap",
        parent=body,
        fontSize=9.5,
        leading=13,
        spaceAfter=5,
    )
    note = ParagraphStyle(
        "NoteCap",
        parent=body,
        fontSize=9,
        textColor="#333333",
        spaceBefore=12,
    )

    story = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:].strip()), title))
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(
                    ListItem(
                        Paragraph(esc(lines[i][2:].strip()), bullet),
                        leftIndent=18,
                        bulletColor="#222222",
                    )
                )
                i += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="•",
                    leftIndent=12,
                    bulletFontSize=9,
                )
            )
            continue
        elif " — " in line:
            left, right = line.split(" — ", 1)
            story.append(Paragraph(f"<b>{esc(left)}</b> — {esc(right)}", fig))
        else:
            style = note if line.startswith("All PNGs") else body
            story.append(Paragraph(esc(line), style))
        i += 1

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Figure captions (for report)",
        author="CS4480 UK Price Paid project",
    )
    doc.build(story)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
