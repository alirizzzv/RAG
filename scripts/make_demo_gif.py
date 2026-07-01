"""Generate a demo GIF from real system outputs.

Runs live queries through the pipeline, captures actual answers, and renders
them as styled chat frames. Content is 100% real system output — just
presented visually for the README.

Run:  python scripts/make_demo_gif.py
Requires: pillow  (already in requirements.txt via sentence-transformers dep)
"""
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.build import get_graph

# ── colours (dark Chainlit palette) ──────────────────────────────────────────
BG      = (17,  17,  27)   # page background
BUBBLE_USER  = (55, 65, 110)
BUBBLE_BOT   = (30, 35,  55)
CITE_BG      = (24, 30,  50)
TEXT_WHITE   = (240, 240, 250)
TEXT_GREY    = (160, 165, 190)
ACCENT       = (100, 149, 237)   # cornflower blue
GREEN        = ( 72, 199, 142)
ORANGE       = (255, 165,  80)

W, H = 900, 560
FONT_SIZE_NORMAL = 18
FONT_SIZE_SMALL  = 14
FONT_SIZE_LABEL  = 12
PAD = 28
RADIUS = 12

# ── font (use default PIL bitmap font as fallback) ────────────────────────────
def _font(size=FONT_SIZE_NORMAL, bold=False):
    try:
        name = "Arial Bold" if bold else "Arial"
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _rounded_rect(draw, xy, radius, fill):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)


def _wrap(text, width=68):
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def make_frame(title, exchanges, highlight=None, chart_path=None):
    """Render one frame: title bar + list of (role, text) exchanges."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # title bar
    draw.rectangle([0, 0, W, 48], fill=(25, 25, 40))
    draw.text((PAD, 14), "🤖  Agentic Knowledge Retrieval System", font=_font(16, bold=True), fill=ACCENT)
    if title:
        tw = draw.textlength(title, font=_font(FONT_SIZE_SMALL))
        draw.text((W - tw - PAD, 16), title, font=_font(FONT_SIZE_SMALL), fill=TEXT_GREY)

    y = 64
    for role, text in exchanges:
        is_user = role == "user"
        bubble_fill = BUBBLE_USER if is_user else BUBBLE_BOT
        prefix = "You" if is_user else "Assistant"
        prefix_color = ACCENT if is_user else GREEN

        # prefix label
        draw.text((PAD, y), prefix, font=_font(FONT_SIZE_LABEL, bold=True), fill=prefix_color)
        y += 20

        # word-wrap and measure
        wrapped = _wrap(text, width=72 if is_user else 75)
        lines = wrapped.split("\n")
        line_h = FONT_SIZE_NORMAL + 6
        box_h = line_h * len(lines) + 20
        box_w = W - 2 * PAD

        _rounded_rect(draw, (PAD, y, PAD + box_w, y + box_h), RADIUS, bubble_fill)
        for i, line in enumerate(lines):
            draw.text((PAD + 12, y + 10 + i * line_h), line,
                      font=_font(FONT_SIZE_NORMAL), fill=TEXT_WHITE)
        y += box_h + 10

    # chart thumbnail if provided
    if chart_path and Path(chart_path).exists():
        chart = Image.open(chart_path).convert("RGB")
        chart.thumbnail((320, 180))
        img.paste(chart, (W - 320 - PAD, H - 180 - PAD))
        draw.text((W - 320 - PAD, H - 180 - PAD - 22),
                  "📊 Generated chart", font=_font(FONT_SIZE_SMALL), fill=ORANGE)

    # highlight badge
    if highlight:
        bw = draw.textlength(highlight, font=_font(FONT_SIZE_SMALL)) + 20
        _rounded_rect(draw, (W - bw - PAD, H - 40, W - PAD, H - 14), 8, ACCENT)
        draw.text((W - bw - PAD + 10, H - 36), highlight,
                  font=_font(FONT_SIZE_SMALL, bold=True), fill=BG)

    return img


def run():
    graph = get_graph()
    frames = []
    durations = []

    # ── Frame 1: welcome ─────────────────────────────────────────────────────
    frames.append(make_frame(
        "Live demo",
        [("bot", "Ask anything about the knowledge base.\n"
                 "I answer with cited sources or generate & run Python for computations.")],
        highlight="alirizzv-agentic-rag.hf.space",
    ))
    durations.append(2000)

    # ── Frame 2: retrieval question ──────────────────────────────────────────
    q1 = "What was Helios Energy total revenue in 2024 and which segment grew fastest?"
    print(f"Running Q1: {q1}")
    r1 = graph.invoke({"question": q1})
    ans1 = r1.get("answer", "").strip()[:220]
    src1 = r1["citations"][0].source if r1.get("citations") else ""
    frames.append(make_frame(
        "Retrieval + citations",
        [("user", q1),
         ("bot", ans1 + (f"\n\n📄 Source: {src1}" if src1 else ""))],
        highlight="`retrieval`",
    ))
    durations.append(3500)

    # ── Frame 3: follow-up (memory) ──────────────────────────────────────────
    q2 = "What were their key risks?"
    print(f"Running Q2 (follow-up): {q2}")
    r2 = graph.invoke({"question": q2, "history": f"Q: {q1}\nA: {ans1}"})
    ans2 = r2.get("answer", "").strip()[:220]
    frames.append(make_frame(
        "Multi-turn memory",
        [("user", q1),
         ("bot", ans1[:100] + "…"),
         ("user", q2 + "  ← follow-up, no context needed"),
         ("bot", ans2)],
        highlight="session memory",
    ))
    durations.append(3500)

    # ── Frame 4: code + chart ─────────────────────────────────────────────────
    q3 = "Plot quarterly revenue for all three companies as a grouped bar chart"
    print(f"Running Q3: {q3}")
    r3 = graph.invoke({"question": q3})
    ans3 = r3.get("answer", "").strip()
    art = r3.get("artifact_path")
    retries = r3.get("retries", 0)
    badge = f"`code` · {retries} retries" if retries else "`code` · 0 retries"
    frames.append(make_frame(
        "Code agent + sandbox",
        [("user", q3),
         ("bot", "Python generated → sandbox executed → chart produced")],
        highlight=badge,
        chart_path=art,
    ))
    durations.append(4000)

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path("docs/demo.gif")
    out.parent.mkdir(exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"\nSaved {out}  ({out.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    run()
