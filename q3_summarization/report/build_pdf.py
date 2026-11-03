"""
Convert report/Q3_report.md into Q3_report.html (figures embedded) and Q3_report.pdf.

Needs the `markdown` package (pip install markdown) and, for the PDF, a Chromium-based browser
(Chrome, Chromium, Brave or Edge), which prints the HTML page to PDF in headless mode.

Run from q3_summarization/:
    python report/build_pdf.py
"""
import base64
import re
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
BROWSERS = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave",
            "brave-browser", "microsoft-edge", "msedge"]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #111; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { text-align: left; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; width: 100%;
        break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; }
th { background: #eee; }
code { font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 8.8pt; background: #f3f3f3;
       padding: 0 2pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6pt; font-size: 8.3pt; line-height: 1.3;
      white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; break-inside: avoid; }
p:has(+ table), p:has(> img) { break-after: avoid; }   /* keep captions with their table / figure */
hr { border: none; border-top: 1px solid #bbb; margin: 12pt 0; }
"""


def embed_images(html):
    """Inline every figures/*.png as base64 so the HTML file is self-contained."""
    def inline(match):
        data = base64.b64encode((HERE / match.group(1)).read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'
    return re.sub(r'src="(figures/[^"]+\.png)"', inline, html)


def main():
    body = markdown.markdown((HERE / "Q3_report.md").read_text(), extensions=["tables", "fenced_code"])
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Q3 Report</title>"
            f"<style>{CSS}</style></head><body>{embed_images(body)}</body></html>")
    html_path = HERE / "Q3_report.html"
    html_path.write_text(html)
    print(f"Wrote {html_path}")

    browser = next((shutil.which(b) for b in BROWSERS if shutil.which(b)), None)
    if browser is None:
        print("No Chromium-based browser found: open the HTML file and print it to PDF by hand.")
        return
    pdf_path = HERE / "Q3_report.pdf"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
