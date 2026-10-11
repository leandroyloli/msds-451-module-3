"""Render the report chapters into one PDF.

The chapter notes live in report/. This script does not refit the model.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_IMAGE = re.compile(r"^!\[(.*)]\((.+)\)$")
_USABLE = 7.0 * inch


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ChapterBody",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChapterH1",
            parent=styles["Heading1"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=10,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChapterH2",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            spaceBefore=2,
            spaceAfter=10,
            textColor=colors.black,
        )
    )
    return styles


def _cell_styles():
    cell = ParagraphStyle(
        name="Cell",
        fontName="Times-Roman",
        fontSize=8,
        leading=10,
        textColor=colors.black,
    )
    header = ParagraphStyle(
        name="CellHead",
        fontName="Times-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.black,
    )
    return cell, header


def _auto_widths(rows: list[list[str]]) -> list[float]:
    weights = []
    for index in range(len(rows[0])):
        longest = max((len(row[index]) if index < len(row) else 0) for row in rows)
        weights.append(max(longest, 3) ** 0.5)
    total = sum(weights)
    return [_USABLE * weight / total for weight in weights]


def _image_flowable(md_path: Path, alt: str, raw_path: str, caption_style):
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (md_path.parent / candidate).resolve()
    if not candidate.exists():
        raise RuntimeError(f"report image not found: {candidate}")
    pixel_w, pixel_h = ImageReader(str(candidate)).getSize()
    width = min(6.2 * inch, _USABLE)
    height = width * pixel_h / pixel_w
    return [Image(str(candidate), width=width, height=height), Paragraph(_escape(alt), caption_style)]


def _blocks(markdown: str) -> list[tuple[str, object]]:
    lines = markdown.splitlines()
    blocks: list[tuple[str, object]] = []
    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = " ".join(part.strip() for part in paragraph).strip()
        if text:
            blocks.append(("p", text))
        paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            blocks.append(("table", table))
        table = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                table.append(cells)
            continue
        flush_table()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append(("h1", stripped[2:].strip()))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("h2", stripped[3:].strip()))
            continue
        matched = _IMAGE.fullmatch(stripped)
        if matched:
            flush_paragraph()
            blocks.append(("image", (matched.group(1), matched.group(2))))
            continue
        paragraph.append(stripped)
    flush_paragraph()
    flush_table()
    return blocks


def _flowables(md_path: Path, col_widths: list[float] | None) -> list:
    styles = _styles()
    cell, header = _cell_styles()
    story = []
    for kind, payload in _blocks(md_path.read_text(encoding="utf-8")):
        if kind == "h1":
            story.append(Paragraph(_escape(payload), styles["ChapterH1"]))
        elif kind == "h2":
            story.append(Paragraph(_escape(payload), styles["ChapterH2"]))
        elif kind == "p":
            story.append(Paragraph(_escape(payload), styles["ChapterBody"]))
        elif kind == "image":
            alt, raw_path = payload
            story.extend(_image_flowable(md_path, alt, raw_path, styles["Caption"]))
        else:
            header_row, *body = payload
            widths = col_widths if col_widths and len(col_widths) == len(header_row) else _auto_widths(payload)
            data = [[Paragraph(_escape(text), header) for text in header_row]]
            data += [[Paragraph(_escape(text), cell) for text in row] for row in body]
            table = Table(data, colWidths=widths, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.93, 0.93, 0.93)),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.Color(0.75, 0.75, 0.75)),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))
    return story


def _page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(letter[0] / 2, 0.45 * inch, str(doc.page))
    canvas.restoreState()


def _write_pdf(story: list, pdf_path: Path, title: str) -> Path:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=title,
        author="MSDS 451",
    )
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return pdf_path


def render_markdown(md_path: Path, pdf_path: Path, title: str, col_widths: list[float]) -> Path:
    return _write_pdf(_flowables(md_path, col_widths), pdf_path, title)


def build_features_chapter() -> Path:
    return render_markdown(
        config.FEATURES_CHAPTER_MD,
        config.FEATURES_CHAPTER_PDF,
        "MSDS 451 Programming Assignment 1 — Features",
        [1.15 * inch, 3.15 * inch, 2.5 * inch],
    )


def build_report() -> Path:
    chapters = [
        config.PROBLEM_MD,
        config.DATA_MD,
        config.FEATURES_CHAPTER_MD,
        config.SELECTION_CHAPTER_MD,
        config.MODEL_CHAPTER_MD,
        config.EVAL_CHAPTER_MD,
        config.CONCLUSION_MD,
    ]
    story = []
    for chapter in chapters:
        story.extend(_flowables(chapter, None))
    return _write_pdf(story, config.REPORT_PDF, "MSDS 451 Programming Assignment 1 — WEGE3.SA")


def main() -> None:
    path = build_report()
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
