"""Build a report PDF from a Markdown report.

Usage:
    python report/build_report.py                 # report/report.md
    python report/build_report.py report/report_10epoch_ablation.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "report.md"
HTML_PATH = ROOT / "report.html"
PDF_PATH = ROOT / "report.pdf"

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 8pt;
    color: #666;
  }
}
body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 9.6pt;
  line-height: 1.42;
  color: #1a1a1a;
}
h1, h2, h3, h4 {
  font-family: "DejaVu Sans", Helvetica, sans-serif;
  color: #12355b;
  line-height: 1.2;
}
h1 { font-size: 20pt; margin: 0 0 4mm 0; }
h2 { font-size: 13pt; margin: 6mm 0 2mm 0; border-bottom: 1px solid #d5dde8; padding-bottom: 1mm; }
h3 { font-size: 10.5pt; margin: 4mm 0 1.5mm 0; }
p { margin: 0 0 2.4mm 0; text-align: justify; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 2mm 0 3mm 0;
  font-family: "DejaVu Sans", Helvetica, sans-serif;
  font-size: 7.6pt;
}
th { background: #eef2f7; color: #12355b; text-align: left; }
th, td { border: 1px solid #cfd8e3; padding: 1.1mm 1.4mm; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfd; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f2f4f7; padding: 0 0.5mm; }
pre { background: #f5f7fa; border: 1px solid #dde3ea; padding: 2mm; font-size: 7.8pt; overflow-wrap: break-word; }
figure { margin: 3mm 0 4mm 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8pt; color: #444; margin-top: 1mm; }
.small { font-size: 8pt; color: #555; }
.note { background: #f7f9fc; border-left: 3px solid #4c78a8; padding: 2mm 3mm; margin: 2mm 0 3mm 0; }
.titlepage { text-align: center; margin-top: 30mm; }
.titlepage h1 { font-size: 26pt; border: none; }
.titlepage .subtitle { font-size: 12pt; color: #444; margin-bottom: 12mm; }
.titlepage .meta { font-size: 10pt; color: #333; }
.pagebreak { page-break-before: always; }
"""


def main() -> None:
    global MD_PATH, HTML_PATH, PDF_PATH
    if len(sys.argv) > 1:
        MD_PATH = (ROOT / sys.argv[1]).resolve() if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
        HTML_PATH = MD_PATH.with_suffix(".html")
        PDF_PATH = MD_PATH.with_suffix(".pdf")
    if not MD_PATH.exists():
        raise SystemExit(f"markdown source not found: {MD_PATH}")

    text = MD_PATH.read_text()
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""
    HTML_PATH.write_text(html)
    HTML(string=html, base_url=str(MD_PATH.parent)).write_pdf(PDF_PATH)
    print(f"[report] {PDF_PATH} ({PDF_PATH.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
