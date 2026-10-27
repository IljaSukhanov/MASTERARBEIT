"""Extract four vector panels from the existing validated distribution PDFs.

No samples, numerical searches, histogram bins, or frequencies are changed.
The common legend is copied from the original figures as vector content.
Inset axes and their endpoint-percentage annotations are omitted. Main-axes
drawing groups are checked for exact equality before and after that omission.
"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, NameObject, RectangleObject
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
_CLEANED_SOURCES = {}
_SOURCE_AUDITS = {}
pdfmetrics.registerFont(TTFont(
    "FigureSans", Path(reportlab.__file__).parent / "fonts/Vera.ttf"
))


def graphics_groups(operations):
    """Map each saved graphics state to its matching restore operation."""
    stack, matches = [], {}
    for index, (_, operator) in enumerate(operations):
        if operator == b"q":
            stack.append(index)
        elif operator == b"Q":
            if not stack:
                raise ValueError("Unbalanced PDF graphics state.")
            matches[stack.pop()] = index
    if stack:
        raise ValueError("Unbalanced PDF graphics state.")
    return matches


def group_text(operations):
    pieces = []
    for operands, operator in operations:
        if operator == b"Tj":
            value = operands[0]
            pieces.append(value.decode("latin1") if isinstance(value, bytes) else str(value))
        elif operator == b"TJ":
            pieces.extend(value.decode("latin1") if isinstance(value, bytes) else value
                          for value in operands[0] if isinstance(value, (str, bytes)))
    return "".join("".join(pieces).split())


def main_axes_fingerprints(operations):
    """Hash every original main-axis clipping group, including histogram paths."""
    matches = graphics_groups(operations)
    signatures = []
    stack = []
    for index, (operands, operator) in enumerate(operations):
        if operator == b"q":
            stack.append(index)
        elif operator == b"Q":
            stack.pop()
        elif operator == b"re" and float(operands[2]) > 400 and float(operands[3]) > 150:
            start = stack[-1]
            group = operations[start:matches[start]+1]
            encoded = repr([(str(args), op) for args, op in group]).encode()
            signatures.append(hashlib.sha256(encoded).hexdigest())
    return signatures


def source_without_insets(path):
    """Remove complete Matplotlib inset artists from an in-memory source page.

    Keep every q/Q operation in removed regions so the original graphics-state
    nesting is preserved. Underlying main-axis paths and gridlines are retained;
    nothing is masked, rasterized, or numerically regenerated.
    """
    if path in _CLEANED_SOURCES:
        return _CLEANED_SOURCES[path]
    reader = PdfReader(path)
    page = deepcopy(reader.pages[0])
    stream = ContentStream(page.get_contents(), reader)
    operations = stream.operations
    before = main_axes_fingerprints(operations)
    groups = graphics_groups(operations)
    annotation_groups, inset_titles = [], []
    for start, end in groups.items():
        group = operations[start:end+1]
        if sum(operator == b"BT" for _, operator in group) != 1:
            continue
        text = group_text(group)
        if text.startswith("Ath=2:"):
            annotation_groups.append((start, end+1))
        elif text == "ath=2":
            inset_titles.append((start, end+1))
    inset_titles.sort()
    backgrounds = []
    background_color = [250/255, 251/255, 253/255]
    for index in range(len(operations)-12):
        if operations[index][1] != b"gs" or operations[index+1][1] != b"rg":
            continue
        if not all(abs(float(a)-b) < 1e-7 for a, b in zip(operations[index+1][0], background_color)):
            continue
        if [operator for _, operator in operations[index+6:index+12]] != [b"m", b"l", b"l", b"l", b"h", b"f"]:
            raise ValueError("Unexpected Matplotlib inset background structure.")
        backgrounds.append(index)
    if len(backgrounds) != 10 or len(annotation_groups) != 10 or len(inset_titles) != 10:
        raise ValueError("Expected ten complete inset artists and endpoint annotations.")
    removed_ranges = list(annotation_groups)
    for start in backgrounds:
        candidates = [pair for pair in inset_titles if pair[0] > start]
        title_start, end = min(candidates)
        if title_start-start > 400:
            raise ValueError("Could not associate the inset title with its graphics.")
        removed_ranges.append((start, end))
    omit = set()
    for start, end in removed_ranges:
        omit.update(range(start, end))
    cleaned = [(args, operator) for index, (args, operator) in enumerate(operations)
               if index not in omit or operator in (b"q", b"Q")]
    after = main_axes_fingerprints(cleaned)
    if before != after:
        raise ValueError("Removing inset artists altered a main-axis drawing group.")
    stream.operations = cleaned
    page[NameObject("/Contents")] = stream
    _CLEANED_SOURCES[path] = page
    _SOURCE_AUDITS[str(path.relative_to(ROOT))] = {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "removed_inset_axes": len(backgrounds),
        "removed_endpoint_annotation_groups": len(annotation_groups),
        "unchanged_main_axis_groups": len(before),
        "main_axis_group_fingerprints": before,
        "main_axis_drawing_groups_exactly_preserved": True,
        "graphics_state_balanced": True,
    }
    return page


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
    source = source_without_insets(source_path)

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
            "ISE, Q0, Q1, Q13/14; original interior distributions without "
            "insets; frequencies divided by all 1000 repetitions"
        ),
    })
    writer.compress_identical_objects()
    writer.pages[0].compress_content_streams()
    with path.open("wb") as stream:
        writer.write(stream)
    text = "".join(PdfReader(path).pages[0].extract_text().split())
    if "Ath=2:" in text or "ath=2" in text:
        raise ValueError("An inset annotation remains in the exported panel.")
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
    audit = {
        "sources": _SOURCE_AUDITS,
        "outputs": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUTPUT_DIRECTORY.glob("continuous_minimizers_n*.pdf"))
        },
        "numerical_samples_bins_and_frequencies_changed": False,
    }
    (Path(__file__).parent / "inset_removal_validation.json").write_text(
        json.dumps(audit, indent=2)+"\n"
    )
