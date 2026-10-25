"""Extract four vector panels from the existing validated distribution PDFs.

No samples, numerical searches, histogram bins, or frequencies are changed.
The common legend is copied from the original figures as vector content.
"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRECTORY = (
    ROOT / "Code/Code_kde/outputs/pdfs/continuous_ise_qhat_distributions_vs_n"
)
OUTPUT_DIRECTORY = ROOT / "output/pdf/continuous_minimizers_selected_n"
PANEL_WIDTH = 496.764145
PANEL_HEIGHT = 222.0
ROW_PITCH = 229.50266
RIGHT_COLUMN_X = 503.2
PAGE_WIDTH = 525.0
PAGE_HEIGHT = 302.0
pdfmetrics.registerFont(TTFont(
    "FigureSans", Path(reportlab.__file__).parent / "fonts/Vera.ttf"
))


def merge_region(destination, source, bounds, x, y):
    """Clip a source rectangle and translate it without rasterizing."""
    left, bottom, right, top = bounds
    clipped = deepcopy(source)
    clipped.cropbox = RectangleObject(bounds)
    clipped.trimbox = RectangleObject(bounds)
    destination.merge_transformed_page(
        clipped,
        Transformation().translate(tx=x - left, ty=y - bottom),
        over=True,
    )


def export_sample_size(n, source_prefix, row, column):
    source_path, = SOURCE_DIRECTORY.glob(f"{source_prefix}*.pdf")
    source = PdfReader(source_path).pages[0]

    header = BytesIO()
    canvas = Canvas(header, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    canvas.setTitle(f"Continuous bandwidth minimizers, n={n}, M=1000")
    canvas.setAuthor("")
    canvas.setFont("FigureSans", 13)
    canvas.drawCentredString(PAGE_WIDTH / 2, 285, "Continuous bandwidth minimizers")
    canvas.setFont("FigureSans", 9)
    canvas.drawCentredString(
        PAGE_WIDTH / 2, 270,
        "Gaussian samples; M = 1000; bandwidth interval [0.01, 2]",
    )
    canvas.showPage()
    canvas.save()
    page = PdfReader(header).pages[0]

    # The original four-colour legend, including its mathematical subscripts.
    merge_region(page, source, (350, 1196, 650, 1223), 112.5, 237)

    left = RIGHT_COLUMN_X if column else 0.0
    bottom = (4 - row) * ROW_PITCH
    merge_region(
        page, source,
        (left, bottom, left + PANEL_WIDTH, bottom + PANEL_HEIGHT),
        (PAGE_WIDTH - PANEL_WIDTH) / 2, 12,
    )

    path = OUTPUT_DIRECTORY / f"continuous_minimizers_n{n}.pdf"
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({
        "/Title": f"Continuous bandwidth minimizers, n={n}, M=1000",
        "/Subject": (
            "ISE, Q0, Q1, Q13/14; original interior distributions and "
            "upper-endpoint mass, frequencies divided by all 1000 repetitions"
        ),
    })
    writer.compress_identical_objects()
    writer.pages[0].compress_content_streams()
    with path.open("wb") as stream:
        writer.write(stream)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for configuration in (
        (100, "003", 0, 0),
        (1000, "003", 4, 1),
        (1500, "004", 2, 0),
        (2000, "004", 4, 1),
    ):
        export_sample_size(*configuration)
