"""Generate self-contained sample documents for the demo knowledge base.

These are ORIGINAL, fictional annual reports (no real or borrowed content), so
anyone who clones the repo can reproduce the demo with zero external downloads.
Each report mixes narrative prose (for retrieval) with financial tables (for the
code-execution agent to chart).

Run:  python scripts/generate_sample_docs.py    # writes PDFs into data/
Requires: reportlab  (pip install reportlab)
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (company, year, narrative paragraphs, quarterly revenue $M, segment breakdown)
REPORTS = [
    (
        "Northwind Robotics",
        2024,
        [
            "Northwind Robotics designs autonomous warehouse robots for mid-market "
            "logistics operators. In fiscal 2024 the company expanded into two new "
            "regions and shipped its third-generation picking arm, Vega-3.",
            "Management's strategy centres on recurring software revenue: every robot "
            "now ships with a subscription fleet-management platform, lifting gross "
            "margin from 41% to 48% year over year.",
            "Key risks include supplier concentration for actuators and intensifying "
            "competition from incumbent material-handling vendors.",
        ],
        [("Q1", 82), ("Q2", 95), ("Q3", 110), ("Q4", 138)],
        [("Hardware", 289), ("Software subscriptions", 96), ("Services", 40)],
    ),
    (
        "Helios Energy",
        2024,
        [
            "Helios Energy operates utility-scale solar and battery-storage assets "
            "across the southwestern grid. Fiscal 2024 saw 1.2 GW of new capacity "
            "reach commercial operation.",
            "The company's storage segment grew fastest as arbitrage revenue from "
            "time-shifting cheap midday solar into evening peaks more than doubled.",
            "Helios funds growth through a mix of project debt and tax-equity "
            "partnerships; rising interest rates remain the principal headwind.",
        ],
        [("Q1", 140), ("Q2", 158), ("Q3", 205), ("Q4", 176)],
        [("Solar generation", 512), ("Battery storage", 121), ("Grid services", 46)],
    ),
    (
        "Meridian Foods",
        2024,
        [
            "Meridian Foods is a packaged-goods maker specialising in plant-based "
            "protein. In fiscal 2024 it reformulated its flagship line to cut sodium "
            "by 20% and launched in 3,000 additional retail doors.",
            "Volume growth was strong but input-cost inflation on pea protein "
            "compressed operating margin to 9.4% from 11.1% a year earlier.",
            "The company is investing in a second manufacturing facility to relieve "
            "capacity constraints expected by fiscal 2026.",
        ],
        [("Q1", 61), ("Q2", 64), ("Q3", 70), ("Q4", 79)],
        [("Retail", 208), ("Food service", 52), ("International", 14)],
    ),
]


def _build(company, year, paragraphs, quarterly, segments):
    styles = getSampleStyleSheet()
    out = DATA_DIR / f"{company.lower().replace(' ', '_')}_{year}_annual_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=LETTER, title=f"{company} {year}")
    flow = [
        Paragraph(f"{company}", styles["Title"]),
        Paragraph(f"Annual Report — Fiscal {year}", styles["Heading2"]),
        Spacer(1, 12),
    ]
    for p in paragraphs:
        flow += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]

    def _table(title, rows, unit_header):
        data = [[title, unit_header]] + [[k, f"{v}"] for k, v in rows]
        t = Table(data, hAlign="LEFT", colWidths=[220, 120])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a67")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef1f7")]),
        ]))
        return t

    flow += [Spacer(1, 16), Paragraph("Quarterly Revenue", styles["Heading3"]),
             _table("Quarter", quarterly, "Revenue ($M)"), Spacer(1, 16),
             Paragraph("Revenue by Segment", styles["Heading3"]),
             _table("Segment", segments, "Revenue ($M)")]
    doc.build(flow)
    return out


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for r in REPORTS:
        path = _build(*r)
        print(f"wrote {path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
