"""Render the report chapters into one HTML file and print it to PDF.

The chapter notes live in report/. This script does not refit the model.
The PDF is printed from the HTML by a headless Edge or Chrome.
"""

from __future__ import annotations

import base64
import html
import importlib.util
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

_IMAGE = re.compile(r"^!\[(.*)]\((.+)\)$")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*%?$")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)
_ITALIC = re.compile(r"\*(.+?)\*")


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _prose(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _ITALIC.finditer(text):
        pieces.append(_escape(text[cursor : match.start()]))
        pieces.append(f"<em>{_escape(match.group(1))}</em>")
        cursor = match.end()
    pieces.append(_escape(text[cursor:]))
    return "".join(pieces)


def _rich(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _MATH.finditer(text):
        pieces.append(_prose(text[cursor : match.start()]))
        latex = match.group(1) if match.group(1) is not None else match.group(2)
        delimiter = ("\\[", "\\]") if match.group(1) is not None else ("\\(", "\\)")
        pieces.append(delimiter[0] + _escape(latex.strip()) + delimiter[1])
        cursor = match.end()
    pieces.append(_prose(text[cursor:]))
    return "".join(pieces)


def _paragraph(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$$") and stripped.endswith("$$") and stripped.count("$$") == 2:
        latex = _escape(stripped[2:-2].strip())
        return f'<div class="equation">\\[{latex}\\]</div>'
    return f"<p>{_rich(text)}</p>"


def _slug(text: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in text)
    return "-".join(part for part in cleaned.split("-") if part)


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


def _is_number(text: str) -> bool:
    return bool(_NUMBER.fullmatch(text.strip()))


def _matrix(table: list[list[str]]) -> str:
    header = table[0]
    if header[:3] != ["Rule", "Accuracy", "ROC AUC"]:
        return ""
    chosen = next((row for row in table[1:] if row[0] == "XGBoost"), None)
    if chosen is None or len(chosen) < 7:
        return ""
    down_hit, down_miss, up_miss, up_hit = chosen[3:7]
    cells = (
        ("hit", down_hit, "correct"),
        ("miss", down_miss, "miss"),
        ("miss", up_miss, "miss"),
        ("hit", up_hit, "correct"),
    )
    rendered = []
    for kind, count, label in cells:
        rendered.append(
            f'<div class="cell {kind}"><strong>{_rich(count)}</strong><span>{label}</span></div>'
        )
    top_left, top_right, bottom_left, bottom_right = rendered
    return f"""
<figure class="matrix-wrap">
  <div class="matrix">
    <div></div>
    <div class="axis">Predicted down or flat</div>
    <div class="axis">Predicted up</div>
    <div class="axis">Actual down or flat</div>
    {top_left}
    {top_right}
    <div class="axis">Actual up</div>
    {bottom_left}
    {bottom_right}
  </div>
  <figcaption>Hold-out calls for the fitted classifier. A green cell is a correct call.</figcaption>
</figure>
"""


def _table(table: list[list[str]]) -> str:
    header, *body = table
    numeric = []
    width = len(header)
    for index in range(width):
        numeric.append(all(_is_number(row[index]) for row in body if index < len(row)))
    head = "".join(
        f'<th class="{"num" if numeric[index] else ""}">{_rich(cell)}</th>'
        for index, cell in enumerate(header)
    )
    rows = []
    for row in body:
        cells = []
        for index in range(width):
            value = row[index] if index < len(row) else ""
            cells.append(f'<td class="{"num" if numeric[index] else ""}">{_rich(value)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + _matrix(table)
    )


def _embed(image: Path) -> str:
    """Inline the figure so the HTML and the PDF do not depend on relative paths."""
    data = base64.b64encode(image.resolve().read_bytes()).decode("ascii")
    kind = "svg+xml" if image.suffix == ".svg" else image.suffix.lstrip(".")
    return f"data:image/{kind};base64,{data}"


def _chapter(path: Path) -> tuple[list[tuple[str, str]], str]:
    nav: list[tuple[str, str]] = []
    parts: list[str] = []
    references = False
    for kind, payload in _blocks(path.read_text(encoding="utf-8")):
        if kind == "h1":
            references = False
            anchor = _slug(payload)
            nav.append((anchor, payload))
            parts.append(f'<h1 id="{anchor}">{_escape(payload)}</h1>')
        elif kind == "h2":
            references = payload == "References"
            parts.append(f"<h2>{_escape(payload)}</h2>")
        elif kind == "p":
            markup = _paragraph(payload)
            if references and markup.startswith("<p>"):
                markup = markup.replace("<p>", '<p class="ref">', 1)
            parts.append(markup)
        elif kind == "image":
            alt, raw_path = payload
            parts.append(
                "<figure>"
                f'<img src="{_embed(path.parent / raw_path)}" alt="{_escape(alt)}">'
                f"<figcaption>{_escape(alt)}</figcaption>"
                "</figure>"
            )
        else:
            parts.append(_table(payload))
    return nav, "\n".join(parts)


def build_report() -> Path:
    chapters = [
        config.PROBLEM_MD,
        config.DATA_MD,
        config.FEATURES_CHAPTER_MD,
        config.UNIVERSE_CHAPTER_MD,
        config.SELECTION_CHAPTER_MD,
        config.MODEL_CHAPTER_MD,
        config.EVAL_CHAPTER_MD,
        config.CONCLUSION_MD,
    ]
    nav: list[tuple[str, str]] = []
    body: list[str] = []
    for chapter in chapters:
        links, markup = _chapter(chapter)
        nav.extend(links)
        body.append(markup)
    links = "\n".join(f'<a href="#{anchor}">{_escape(label)}</a>' for anchor, label in nav)
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicting the next-day direction of WEGE3.SA</title>
<style>
  :root {{
    --paper: #f3f6f4;
    --sheet: #ffffff;
    --ink: #1c2420;
    --muted: #5c6b63;
    --line: #d5ded8;
    --up: #1f6b56;
    --down: #8f4038;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.65;
    hyphens: auto;
    -webkit-hyphens: auto;
  }}
  .layout {{
    display: grid;
    grid-template-columns: 200px minmax(0, 40rem);
    gap: 40px;
    max-width: 980px;
    margin: 0 auto;
    padding: 36px 24px 80px;
  }}
  nav {{
    position: sticky;
    top: 24px;
    align-self: start;
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  nav a {{
    display: block;
    color: var(--ink);
    text-decoration: none;
    padding: 7px 0 7px 12px;
    border-left: 2px solid var(--line);
  }}
  nav a:hover {{ border-left-color: var(--up); color: var(--up); }}
  h1, h2 {{ font-weight: 650; letter-spacing: -0.02em; line-height: 1.2; }}
  h1 {{ font-size: 1.7rem; margin: 2.2rem 0 0.8rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.6rem 0 0.6rem; }}
  .mast {{
    margin-bottom: 2rem;
    padding-bottom: 1.3rem;
    border-bottom: 1px solid var(--ink);
  }}
  .mast p, .mast dd {{ text-align: left; hyphens: manual; }}
  .facts {{
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.85rem 1.2rem;
    margin: 0 0 1.45rem;
  }}
  .facts div {{ grid-column: span 2; min-width: 0; }}
  .facts div.wide {{ grid-column: span 3; }}
  .facts dt {{
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .facts dd {{ margin: 0.12rem 0 0; }}
  .facts .who dd {{ font-size: 1.15rem; }}
  .mast h1 {{
    margin: 0;
    font-size: 2.05rem;
    line-height: 1.15;
  }}
  p, figcaption {{
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
    hyphenate-limit-chars: 6 3 3;
  }}
  p {{ margin: 0 0 0.85rem; }}
  p.ref {{
    padding-left: 1.5em;
    text-indent: -1.5em;
    margin-bottom: 0.45rem;
  }}
  .equation {{
    margin: 0.35rem 0 1rem;
    overflow-x: auto;
    text-align: center;
  }}
  .table-wrap {{ overflow-x: auto; margin: 0.4rem 0 1.4rem; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-variant-numeric: tabular-nums;
    font-size: 0.92rem;
  }}
  th {{
    text-align: left;
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--ink);
    padding: 8px 10px 8px 0;
  }}
  td {{
    border-bottom: 1px solid var(--line);
    padding: 8px 10px 8px 0;
    vertical-align: middle;
  }}
  th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  figure {{ margin: 1.2rem 0 1.6rem; }}
  img {{ width: 100%; height: auto; background: var(--sheet); }}
  figcaption {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.92rem;
  }}
  .matrix {{
    display: grid;
    grid-template-columns: 148px 1fr 1fr;
    gap: 8px;
    align-items: stretch;
  }}
  .axis {{
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    align-items: end;
    padding-bottom: 4px;
  }}
  .cell {{
    background: var(--sheet);
    border-radius: 10px;
    padding: 16px 10px;
    text-align: center;
  }}
  .cell strong {{
    display: block;
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 1.7rem;
    font-weight: 650;
    letter-spacing: -0.03em;
  }}
  .cell span {{
    color: var(--muted);
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .hit {{ background: #e7f3ee; color: var(--up); }}
  .miss {{ background: #f8eeec; color: var(--down); }}
  @media print {{
    @page {{ size: A4; margin: 18mm 16mm; }}
    body {{ background: #ffffff; font-size: 10.5pt; }}
    .layout {{ display: block; max-width: none; padding: 0; }}
    nav {{ display: none; }}
    h1 {{ break-after: avoid; }}
    h2 {{ break-after: avoid; }}
    table, figure, .equation {{ break-inside: avoid; }}
    .cell {{ border: 1px solid var(--line); }}
  }}
  @media (max-width: 800px) {{
    .layout {{ grid-template-columns: 1fr; }}
    nav {{ position: static; }}
    .facts {{ grid-template-columns: 1fr; }}
    .facts div, .facts div.wide {{ grid-column: auto; }}
  }}
</style>
</head>
<body>
<div class="layout">
<nav>
{links}
</nav>
<main>
<header class="mast">
  <dl class="facts">
    <div><dt>University</dt><dd>Northwestern University</dd></div>
    <div><dt>Program</dt><dd>Master of Science in Data Science</dd></div>
    <div class="who"><dt>Student</dt><dd>Leandro Loli</dd></div>
    <div><dt>Course</dt><dd>MSDS 451, Section 001</dd></div>
    <div><dt>Class</dt><dd>Financial Engineering</dd></div>
    <div><dt>Term</dt><dd>Fall 2026</dd></div>
    <div class="wide"><dt>Assignment</dt><dd>Programming Assignment 1</dd></div>
    <div class="wide"><dt>Date</dt><dd>October 11, 2026</dd></div>
  </dl>
  <h1>Predicting the next-day direction of WEGE3.SA</h1>
</header>
{"".join(body)}
</main>
</div>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</body>
</html>
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_HTML.write_text(document, encoding="utf-8")
    return config.REPORT_HTML


# Full install paths come first: a bare "msedge" on Windows can resolve to an app alias
# that hands the job to a running browser and exits before the PDF is written.
_BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "msedge",
    "chrome",
    "google-chrome",
    "chromium",
)


def _browser() -> str | None:
    for candidate in _BROWSERS:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    return None


def build_pdf(html_path: Path) -> Path | None:
    """Print the HTML with a headless browser. MathJax needs network access to render."""
    browser = _browser()
    if browser is None:
        print("no Edge or Chrome found; open the HTML and print it to PDF")
        return None
    profile = Path(tempfile.mkdtemp(prefix="pdf-profile-"))
    subprocess.run(
        [
            browser,
            "--headless=new",
            f"--user-data-dir={profile}",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={config.REPORT_PDF}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    shutil.rmtree(profile, ignore_errors=True)
    if not config.REPORT_PDF.exists():
        raise RuntimeError("the browser did not write the PDF")
    return config.REPORT_PDF


def main() -> None:
    path = build_report()
    print(f"wrote: {path}")
    pdf = build_pdf(path)
    if pdf is not None:
        print(f"wrote: {pdf}")


if __name__ == "__main__":
    main()
