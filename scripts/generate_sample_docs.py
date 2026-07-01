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
# Figures are stated inline in the prose so both the retrieval agent and the
# code-execution agent can read clean numbers. Quarterly totals equal the sum of
# the segment breakdown for each company (internally consistent).
REPORTS = [
    (
        "Northwind Robotics",
        2024,
        [
            "Northwind Robotics designs and manufactures autonomous mobile robots for "
            "mid-market warehouse and logistics operators. In fiscal 2024 the company "
            "generated total revenue of $425 million, up 37% from $310 million in 2023, "
            "and expanded operations into two new geographic regions.",
            "Quarterly revenue grew steadily through the year: $82 million in Q1, "
            "$95 million in Q2, $110 million in Q3, and $138 million in Q4, reflecting "
            "strong holiday-season fulfillment demand and the launch of the "
            "third-generation picking arm, Vega-3.",
            "By segment, Hardware contributed $289 million, Software subscriptions "
            "$96 million, and Services $40 million. Management's strategy centres on "
            "recurring software revenue: every robot now ships with a subscription "
            "fleet-management platform, lifting gross margin from 41% to 48% year over year.",
            "Key risks include supplier concentration for precision actuators, where "
            "two vendors account for over 70% of supply, and intensifying competition "
            "from incumbent material-handling manufacturers.",
            "For fiscal 2025, Northwind guides to revenue between $520 million and "
            "$560 million, driven by international expansion and higher software "
            "attach rates across its installed fleet.",
        ],
        [("Q1", 82), ("Q2", 95), ("Q3", 110), ("Q4", 138)],
        [("Hardware", 289), ("Software subscriptions", 96), ("Services", 40)],
    ),
    (
        "Helios Energy",
        2024,
        [
            "Helios Energy develops and operates utility-scale solar generation and "
            "battery-storage assets across the southwestern grid. In fiscal 2024 the "
            "company brought 1.2 gigawatts of new capacity into commercial operation "
            "and reported total revenue of $679 million.",
            "Revenue by quarter was $140 million in Q1, $158 million in Q2, "
            "$205 million in Q3, and $176 million in Q4; the third-quarter peak "
            "reflects maximum summer solar output and elevated grid-balancing payments.",
            "Solar generation accounted for $512 million of revenue, battery storage "
            "$121 million, and grid services $46 million. The storage segment grew "
            "fastest as arbitrage revenue from time-shifting midday solar into evening "
            "peaks more than doubled year over year.",
            "Helios funds growth through a mix of long-term project debt and "
            "tax-equity partnerships; rising interest rates remain the principal "
            "headwind to new project economics.",
            "Management targets 1.5 gigawatts of additional capacity in fiscal 2025 "
            "and expects storage to reach 25% of total revenue by 2027.",
        ],
        [("Q1", 140), ("Q2", 158), ("Q3", 205), ("Q4", 176)],
        [("Solar generation", 512), ("Battery storage", 121), ("Grid services", 46)],
    ),
    (
        "Meridian Foods",
        2024,
        [
            "Meridian Foods is a packaged-goods manufacturer specialising in "
            "plant-based protein products. In fiscal 2024 total revenue was "
            "$274 million as the company launched in 3,000 additional retail doors and "
            "reformulated its flagship line to reduce sodium by 20%.",
            "Quarterly revenue was $61 million in Q1, $64 million in Q2, $70 million "
            "in Q3, and $79 million in Q4, with sequential growth driven by expanded "
            "distribution and new product launches.",
            "The Retail segment generated $208 million, Food service $52 million, and "
            "International $14 million. Volume growth was strong, but input-cost "
            "inflation on pea protein compressed operating margin to 9.4% from 11.1% "
            "a year earlier.",
            "The company is investing $60 million in a second manufacturing facility "
            "to relieve capacity constraints expected by fiscal 2026.",
            "Meridian guides fiscal 2025 revenue to approximately $320 million and "
            "expects margin recovery as commodity costs normalise.",
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
