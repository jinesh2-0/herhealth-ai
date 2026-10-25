"""Render the GP summary as a printable PDF in an NHS patient-information style.

NHS Design System colours and plain layout are used for readability. No NHS logo or
branding is used: the document says clearly that it was prepared by the patient with a
concept prototype and is not an NHS document.
"""

from __future__ import annotations

import io
from datetime import date
from xml.sax.saxutils import escape

import matplotlib
import matplotlib.dates  # noqa: F401  (date axis formatting)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from . import tracker  # noqa: E402

BLUE, DARK_BLUE = colors.HexColor("#005EB8"), colors.HexColor("#003087")
TEXT, GREY = colors.HexColor("#212B32"), colors.HexColor("#4C6272")
PALE, BORDER = colors.HexColor("#F0F4F5"), colors.HexColor("#D8DDE0")
RED, GREEN, WHITE = colors.HexColor("#D5281B"), colors.HexColor("#007F3B"), colors.white
HEX = {"blue": "#005EB8", "red": "#D5281B", "green": "#007F3B", "grey": "#4C6272"}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
TITLE = "Health summary for my GP appointment"

_PUNCT = {"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
          "≤": "<=", "≥": ">=", "→": "->", " ": " "}


def _clean(text) -> str:
    """Escape for ReportLab and keep to characters the built-in fonts can draw."""
    text = "".join(_PUNCT.get(ch, ch) for ch in str(text))
    text = text.encode("cp1252", "ignore").decode("cp1252")
    return escape(text)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
S = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, textColor=TEXT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11.5,
                            textColor=GREY),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14, leading=18,
                         textColor=TEXT, spaceBefore=10, spaceAfter=5),
    "card_head": ParagraphStyle("card_head", fontName="Helvetica-Bold", fontSize=10.5,
                                leading=13, textColor=WHITE),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=TEXT),
    "cell_b": ParagraphStyle("cell_b", fontName="Helvetica-Bold", fontSize=9, leading=12,
                             textColor=TEXT),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=10, leading=14,
                             textColor=TEXT, leftIndent=12, bulletIndent=2, alignment=TA_LEFT),
}


def _p(text, style="body"):
    return Paragraph(text, S[style])


def _bullets(items):
    return [Paragraph(_clean(i), S["bullet"], bulletText="•") for i in items]


def _box(flowables, border=BLUE, background=WHITE):
    """NHS 'inset text' style: content with a thick coloured left border."""
    t = Table([[flowables]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LINEBEFORE", (0, 0), (0, -1), 4, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _care_card(alert: dict):
    """Coloured heading strip + white body, like an NHS care card."""
    colour = RED if alert["level"] == "urgent" else BLUE
    label = "Urgent advice" if alert["level"] == "urgent" else "See a GP"
    body = [_p(_clean(alert["detail"]), "cell")]
    if alert.get("tests"):
        body.append(Spacer(1, 3))
        body.append(_p("<b>Tests a GP might consider:</b> " + _clean("; ".join(alert["tests"])),
                       "cell"))
    body.append(Spacer(1, 2))
    body.append(_p("Source: " + _clean(alert["source"]), "small"))
    t = Table([[_p(f"{label}: {_clean(alert['title'])}", "card_head")], [body]],
              colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colour),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([t, Spacer(1, 6)])


def _table(rows, widths, header=True):
    data = [[_p(_clean(c), "cell_b" if header and i == 0 else "cell") for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, BORDER),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Charts (matplotlib -> PNG in memory)
# ---------------------------------------------------------------------------
def _fig_to_image(fig, width=CONTENT_W):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width, height=width * h / w)


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#AEB7BD")
    ax.spines["bottom"].set_color("#AEB7BD")
    ax.tick_params(colors=HEX["grey"], labelsize=8)
    ax.yaxis.label.set_color(HEX["grey"])


def cycle_chart(log: dict, today: date):
    st = tracker.stats(log, today)
    if not st["cycles"]:
        return None
    labels = [s.strftime("%d %b") for s, _ in st["cycles"]]
    values = [c for _, c in st["cycles"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axhspan(21, 35, color=HEX["green"], alpha=0.12, label="Usual range (21-35 days)")
    bar_colours = [HEX["blue"] if 21 <= v <= 35 else HEX["red"] for v in values]
    bars = ax.bar(labels, values, color=bar_colours, width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=8,
                color="#212B32")
    ax.set_ylabel("Days between periods")
    ax.set_title("Cycle length (red = outside the usual 21 to 35 days)", fontsize=10,
                 loc="left", color="#212B32")
    ax.set_ylim(0, max(values + [40]) + 6)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    _style_axes(ax)
    return _fig_to_image(fig)


def period_chart(log: dict, today: date):
    st = tracker.stats(log, today)
    rows = [(tracker._d(p["start"]),
             (tracker._d(p["end"]) - tracker._d(p["start"])).days + 1 if p.get("end") else None,
             p.get("pain")) for p in st["periods"]]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return None
    labels = [d.strftime("%d %b") for d, _, _ in rows]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.bar(labels, [n for _, n, _ in rows], color="#41B6E6", width=0.55, label="Days of bleeding")
    ax.axhline(7, color=HEX["red"], linestyle="--", linewidth=1, label="7 days (NHS)")
    ax.set_ylabel("Days")
    _style_axes(ax)
    pains = [(i, pn) for i, (_, _, pn) in enumerate(rows) if pn is not None]
    if pains:
        ax2 = ax.twinx()
        ax2.plot([i for i, _ in pains], [pn for _, pn in pains], color=HEX["blue"], marker="o",
                 linewidth=1.6, label="Worst pain (0-10)")
        ax2.set_ylim(0, 10.5)
        ax2.set_ylabel("Pain (0-10)", color=HEX["grey"])
        ax2.tick_params(colors=HEX["grey"], labelsize=8)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5, frameon=False, ncol=3)
    else:
        ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax.set_ylim(0, max(max(n for _, n, _ in rows), 8) + 3)
    ax.set_title("Period length and pain", fontsize=10, loc="left", color="#212B32")
    return _fig_to_image(fig)


def symptom_chart(counted: list[tuple[str, int]]):
    if not counted:
        return None
    counted = sorted(counted, key=lambda x: x[1])[-10:]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(counted) + 0.9))
    ax.barh([c[0] for c in counted], [c[1] for c in counted], color=HEX["blue"], height=0.55)
    for i, (_, n) in enumerate(counted):
        ax.text(n + 0.1, i, str(n), va="center", fontsize=8, color="#212B32")
    ax.set_xlabel("Days logged in the last 6 months", color=HEX["grey"], fontsize=8)
    ax.set_title("Symptoms logged", fontsize=10, loc="left", color="#212B32")
    _style_axes(ax)
    return _fig_to_image(fig)


def impact_chart(log: dict, today: date):
    rows = sorted((tracker._d(s["date"]), s["impact"]) for s in log["symptoms"]
                  if s.get("impact") is not None
                  and (today - tracker._d(s["date"])).days <= tracker.RECENT_DAYS)
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.axhspan(tracker.IMPACT_GP, 10.5, color=HEX["red"], alpha=0.08,
               label="Affecting daily life (7+)")
    ax.plot([d for d, _ in rows], [v for _, v in rows], color=HEX["blue"], marker="o")
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Effect on my day (0-10)")
    ax.set_title("How much symptoms affected my day", fontsize=10, loc="left", color="#212B32")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    _style_axes(ax)
    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def _first_page(canvas, doc, prepared):
    canvas.saveState()
    band = 30 * mm
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H - band, PAGE_W, band, stroke=0, fill=1)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 19)
    canvas.drawString(MARGIN, PAGE_H - 15 * mm, TITLE)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(MARGIN, PAGE_H - 22 * mm,
                      f"Prepared by the patient on {prepared} using HerHealth")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 22 * mm,
                           "Concept prototype - not an NHS document")
    _footer(canvas, doc)
    canvas.restoreState()


def _later_pages(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, PAGE_H - 12 * mm, PAGE_W - MARGIN, PAGE_H - 12 * mm)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, PAGE_H - 10 * mm, TITLE)
    _footer(canvas, doc)
    canvas.restoreState()


def _footer(canvas, doc):
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN, 9.5 * mm, "Information logged by the patient in HerHealth, a concept "
                      "prototype. Not affiliated with the NHS. Not a diagnosis.")
    canvas.drawRightString(PAGE_W - MARGIN, 9.5 * mm, f"Page {doc.page}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_pdf(log: dict, today: date, alert_list: list[dict], result: dict | None,
              extras: dict | None = None, accuracy: dict | None = None) -> bytes:
    d = tracker.summary_data(log, today, alert_list, result, extras)
    story = [Spacer(1, 16 * mm)]

    story.append(_box([_p("This summary was written by me from my own records, to help us use "
                          "the appointment well. The flags below follow published NHS and NICE "
                          "guidance; they are prompts to discuss, not a diagnosis.", "cell")],
                      border=GREY, background=PALE))

    if d["about"]:
        story += [_p("About me", "h2"),
                  _table([(k, v) for k, v in d["about"]], [45 * mm, CONTENT_W - 45 * mm],
                         header=False)]
    if d["concern"]:
        story += [_p("What I'd like help with", "h2"), _box([_p(_clean(d["concern"]))])]
    if d["impact"]:
        value, day = d["impact"]
        colour = HEX["red"] if value >= tracker.IMPACT_GP else HEX["blue"]
        story += [_p("How my symptoms affect me", "h2"),
                  _box([_p(f"Worst day in the last month: <font color='{colour}'><b>{value}/10"
                           f"</b></font> - {_clean(tracker.impact_label(value))} "
                           f"({day.strftime('%d %b')}).")],
                       border=RED if value >= tracker.IMPACT_GP else BLUE)]

    story.append(_p("Things flagged from my records", "h2"))
    if d["flagged"]:
        story += [_care_card(a) for a in d["flagged"]]
    else:
        story.append(_p("Nothing in my records matched NHS or NICE 'see a GP' guidance."))

    if d["tests"]:
        story += [_p("Tests I'd like to ask about", "h2"), *_bullets(d["tests"])]

    if d["medicines"] or d["tried"] or d["tried_notes"]:
        rows = []
        if d["medicines"]:
            rows.append(("Medicines I take now", d["medicines"]))
        if d["tried"]:
            rows.append(("Treatments I've tried", ", ".join(d["tried"])))
        if d["tried_notes"]:
            rows.append(("What helped or didn't", d["tried_notes"]))
        story += [_p("Medicines and treatments", "h2"),
                  _table(rows, [45 * mm, CONTENT_W - 45 * mm], header=False)]

    if d["answers"]:
        story += [_p("More about my symptoms", "h2"),
                  _table([("Question", "My answer")] + d["answers"],
                         [75 * mm, CONTENT_W - 75 * mm])]

    story += [_p("Questions I'd like to ask", "h2"), *_bullets(d["questions"])]

    charts = [c for c in (cycle_chart(log, today), period_chart(log, today),
                          symptom_chart(d["symptoms"]), impact_chart(log, today)) if c]
    if charts:
        story.append(KeepTogether([_p("My records at a glance", "h2"), charts[0]]))
        for c in charts[1:]:
            story += [Spacer(1, 6), c]

    if d["periods"]:
        cyc = (f"Typical cycle {d['typical_cycle']:.0f} days (range {min(d['cycles'])} to "
               f"{max(d['cycles'])})." if d["cycles"] else "Not enough periods for a cycle length.")
        story += [_p("Periods logged", "h2"), _p(_clean(cyc), "small"), Spacer(1, 3),
                  _table([("First day", "Days", "Heaviest flow", "Worst pain (0-10)", "Source")]
                         + d["periods"],
                         [30 * mm, 15 * mm, 32 * mm, 32 * mm, CONTENT_W - 109 * mm])]
    if d["weight_change"]:
        (w0, d0), (w1, d1) = d["weight_change"]
        story.append(_p(_clean(f"Weight logged: {w0} kg ({d0.strftime('%b %Y')}) to {w1} kg "
                               f"({d1.strftime('%b %Y')})."), "small"))

    if result:
        acc = ""
        if accuracy:
            acc = (f" How accurate is it? Tested on {accuracy['n']} people from a public research "
                   f"dataset: of 100 people with PMOS it flags about {accuracy['caught']}, and of "
                   f"100 without it about {accuracy['false_alarm']} are flagged too. In the "
                   f"general population only about {accuracy['ppv_in_10']} in 10 flags would be "
                   "right, so this is a prompt to discuss, not a diagnosis.")
        story += [_p("PMOS pattern check (formerly PCOS)", "h2"),
                  _box([_p(f"<b>Screening score {round(result['risk'] * 100)}/100.</b> "
                           + _clean(result["band_text"])),
                        Spacer(1, 3), _p(_clean(acc), "small")])]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=18 * mm, bottomMargin=20 * mm, title=TITLE,
                            author="Patient (HerHealth concept prototype)")
    doc.build(story, onFirstPage=lambda c, dd: _first_page(c, dd, d["prepared"]),
              onLaterPages=_later_pages)
    return buf.getvalue()
