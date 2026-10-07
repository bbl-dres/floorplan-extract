"""Convert the PDFs in papers-original/ to clean Markdown in papers-md/ with Docling.

    python research/convert_papers.py            # convert PDFs without a Markdown file yet
    python research/convert_papers.py --force    # reconvert everything

New, unsorted PDFs go to papers-incoming/ first; once named and added to papers.json they move to
papers-original/ and are converted here.

The papers are born-digital, so OCR is off; tables are reconstructed, images become placeholders.
Like the PDFs, the Markdown files are full-text copies for private reading and are gitignored.
"""
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

HERE = Path(__file__).resolve().parent
PAPERS, MARKDOWN = HERE / "papers-original", HERE / "papers-md"


def converter():
    options = PdfPipelineOptions()
    options.do_ocr = False                  # born-digital PDFs: use the text layer
    options.do_table_structure = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


if __name__ == "__main__":
    MARKDOWN.mkdir(exist_ok=True)
    force = "--force" in sys.argv
    todo = [p for p in sorted(PAPERS.glob("*.pdf")) if force or not (MARKDOWN / f"{p.stem}.md").exists()]
    print(f"{len(todo)} PDF(s) to convert", flush=True)
    conv, failed = converter(), []
    for i, pdf in enumerate(todo, 1):
        start = time.time()
        try:
            doc = conv.convert(pdf).document
            md = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
            (MARKDOWN / f"{pdf.stem}.md").write_text(md, encoding="utf-8")
            print(f"[{i}/{len(todo)}] ok    {pdf.name}  {doc.num_pages()} pages, {len(md) / 1000:.0f}k chars, {time.time() - start:.0f}s", flush=True)
        except Exception as e:                              # keep going; report at the end
            failed.append(pdf.name)
            print(f"[{i}/{len(todo)}] FAIL  {pdf.name}: {type(e).__name__}: {e}", flush=True)
    print(f"done: {len(todo) - len(failed)} converted, {len(failed)} failed {failed or ''}")
