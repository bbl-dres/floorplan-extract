"""Swiss plans on Wikimedia Commons (public domain, CC0): images without a reference or a known scale."""
from fpeval.datasets import BENCH, Native
from fpeval.raster import rgb

FOLDER = BENCH / "commons-plans/images"


def ids():
    return [f"c{k:02d}" for k in range(1, 19)]


def image_path(cid):
    return next(FOLDER.glob(f"{cid}.*"))


def native(cid):
    return Native(rgb(image_path(cid)), None, None, None, None)
