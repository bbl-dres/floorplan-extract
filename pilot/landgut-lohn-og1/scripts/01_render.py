"""Render the 2005 vector PDF to a 400 dpi raster (text layer ignored), crop the main building,
and write 1500 px previews of all test sheets for whole-page VLM reading (mode A)."""
import pymupdf
from PIL import Image

from common import OUT, PDF_2005, SCAN_0056, SCAN_0066

Image.MAX_IMAGE_PIXELS = None

page = pymupdf.open(PDF_2005)[0]
page.get_pixmap(dpi=400, colorspace=pymupdf.csGRAY).save(OUT / "s1_pdf2005_400dpi.png")

# Main-building crop at native resolution (mode B); box found on the 1500 px preview
render = Image.open(OUT / "s1_pdf2005_400dpi.png")
render.crop((467, 1107, 1762, 2073)).save(OUT / "s1_main_crop_400dpi.png")

for name, path in [("s1_pdf2005", OUT / "s1_pdf2005_400dpi.png"), ("s2_scan0056", SCAN_0056), ("s3_scan0066", SCAN_0066)]:
    im = Image.open(path).convert("L")
    s = 1500 / max(im.size)
    im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS).save(OUT / f"{name}_preview.png")
    print(name, im.size)
