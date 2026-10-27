"""Export Figure 3.9 with every selected interval minimizer, including h=2.

Copy panels from the original validated full-interval PDFs as vector content.
Samples, numerical searches, histogram bins, and counts remain unchanged.
Recover integer bin counts from the original filled step paths to verify that
all 1,000 repetitions are represented in each displayed distribution.
"""
from copy import deepcopy
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, RectangleObject
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRECTORY = ROOT/'Code/Code_kde/outputs/pdfs/continuous_ise_qhat_distributions_vs_n'
OUTPUT_DIRECTORY = ROOT/'output/pdf/continuous_minimizers_selected_n'
ANALYSIS_DIRECTORY = Path(__file__).parent
PANEL_WIDTH, PANEL_HEIGHT = 496.732874, 222.0
ROW_PITCH, RIGHT_COLUMN_X = 229.504, 503.2
PAGE_WIDTH, PAGE_HEIGHT = 525.0, 302.0
REPETITIONS, HISTOGRAM_POINTS, ORIGINAL_Y_MAX = 1000, 201, 1.06
METHOD_COLORS = {
    'ISE': (123/255, 74/255, 178/255),
    'Q0': (0.0, 114/255, 178/255),
    'Q1': (213/255, 94/255, 0.0),
    'Q13_14': (0.0, 158/255, 115/255),
}
CONFIGURATIONS = ((100, '001', 0, 0), (1000, '001', 4, 1),
                  (1500, '002', 2, 0), (2000, '002', 4, 1))
pdfmetrics.registerFont(TTFont('FigureSans', Path(reportlab.__file__).parent/'fonts/Vera.ttf'))


def filled_histograms(reader, page, row, column):
    """Validate integer counts recovered from all 201 original step segments.

    Matplotlib filled stairs retain every bin vertex. Source coordinates have
    six decimal places; recovery accepts at most 0.0001 of a count rounding
    error. The original full-interval plots have a common y maximum of 1.06.
    """
    operations = ContentStream(page.get_contents(), reader).operations
    stack, groups = [], []
    for index, (_, operator) in enumerate(operations):
        if operator == b'q':
            stack.append(index)
        elif operator == b'Q':
            if not stack:
                raise ValueError('Unbalanced source graphics state.')
            groups.append((stack.pop(), index))
    if stack:
        raise ValueError('Unbalanced source graphics state.')
    target_left = 53.621875 + column*RIGHT_COLUMN_X
    target_bottom = 36.590625 + (4-row)*ROW_PITCH
    result = {}
    for start, end in groups:
        group = operations[start:end+1]
        rectangles = [args for args, op in group if op == b're']
        if len(rectangles) != 1 or not any(op == b'f' for _, op in group):
            continue
        left, bottom, width, height = map(float, rectangles[0])
        if abs(left-target_left) > 1e-5 or abs(bottom-target_bottom) > 1e-5:
            continue
        colors = [tuple(map(float, args)) for args, op in group if op == b'rg']
        methods = [name for name, rgb in METHOD_COLORS.items()
                   if any(len(color) == 3 and max(abs(a-b) for a, b in zip(color, rgb)) < 1e-7
                          for color in colors)]
        if len(methods) != 1:
            raise ValueError('Cannot identify the original histogram color.')
        method, = methods
        vertices = [tuple(map(float, args)) for args, op in group if op in (b'm', b'l')]
        if len(vertices) != 2*HISTOGRAM_POINTS+2:
            raise ValueError('Unexpected original filled-stairs geometry.')
        counts, errors = [], []
        for index in range(HISTOGRAM_POINTS):
            first, second = vertices[2*index+1:2*index+3]
            if first[0] >= second[0] or abs(first[1]-second[1]) > 1e-8:
                raise ValueError('Expected one horizontal segment per histogram bin.')
            raw_count = (first[1]-bottom)/height*ORIGINAL_Y_MAX*REPETITIONS
            count = round(raw_count)
            error = abs(raw_count-count)
            if error > 1e-4 or not 0 <= count <= REPETITIONS:
                raise ValueError('A vector histogram ordinate is not an integer count.')
            counts.append(count)
            errors.append(error)
        if method in result or sum(counts) != REPETITIONS:
            raise ValueError('Each distribution must contain exactly 1,000 selections.')
        result[method] = {'counts': counts, 'total_count': sum(counts),
                          'last_bin_count': counts[-1],
                          'maximum_vector_count_rounding_error': max(errors)}
    if set(result) != set(METHOD_COLORS):
        raise ValueError('Expected all four full-interval histograms.')
    return result


def merge_region(destination, source, bounds, x, y):
    """Clip and translate vector content without altering its plotted paths."""
    left, bottom, _, _ = bounds
    clipped = deepcopy(source)
    clipped.cropbox = RectangleObject(bounds)
    clipped.trimbox = RectangleObject(bounds)
    destination.merge_transformed_page(
        clipped, Transformation().translate(tx=x-left, ty=y-bottom), over=True)


def export_sample_size(n, source_prefix, row, column):
    source_path, = SOURCE_DIRECTORY.glob(f'{source_prefix}*.pdf')
    reader = PdfReader(source_path)
    source = reader.pages[0]
    histograms = filled_histograms(reader, source, row, column)
    header = BytesIO()
    canvas = Canvas(header, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    canvas.setTitle(f'Global bandwidth minimizers, n={n}, M=1000, full interval')
    canvas.setAuthor('')
    canvas.setFont('FigureSans', 13)
    canvas.drawCentredString(PAGE_WIDTH/2, 285, 'Global bandwidth minimizers')
    canvas.setFont('FigureSans', 9)
    canvas.drawCentredString(PAGE_WIDTH/2, 270,
                            'Gaussian samples; M = 1000; full interval [0.01, 2]')
    canvas.showPage()
    canvas.save()
    page = PdfReader(header).pages[0]
    merge_region(page, source, (350, 1196, 650, 1223), 112.5, 237)
    left, bottom = column*RIGHT_COLUMN_X, (4-row)*ROW_PITCH
    bounds = (left, bottom, left+PANEL_WIDTH, bottom+PANEL_HEIGHT)
    merge_region(page, source, bounds, (PAGE_WIDTH-PANEL_WIDTH)/2, 12)
    path = OUTPUT_DIRECTORY/f'continuous_minimizers_n{n}.pdf'
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({
        '/Title': f'Global bandwidth minimizers, n={n}, M=1000, full interval',
        '/Subject': 'ISE, Q0, Q1, Q13/14; all 1000 numerical interval-minimizer selections per criterion; endpoints included in main histograms; no insets',
    })
    writer.compress_identical_objects()
    writer.pages[0].compress_content_streams()
    with path.open('wb') as stream:
        writer.write(stream)
    if len(PdfReader(path).pages) != 1:
        raise ValueError('Expected a single figure panel.')
    return {'n': n, 'source': str(source_path.relative_to(ROOT)),
            'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'source_panel_bounds': bounds, 'output': str(path.relative_to(ROOT)),
            'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'histograms': histograms}


if __name__ == '__main__':
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    panels = [export_sample_size(*configuration) for configuration in CONFIGURATIONS]
    records = []
    for panel in panels:
        for method, histogram in panel['histograms'].items():
            counts = histogram.pop('counts')
            for index, count in enumerate(counts):
                records.append({'n': panel['n'], 'method': method, 'bin_index': index,
                                'bandwidth_center': 0.01*200**(index/(HISTOGRAM_POINTS-1)),
                                'count': count, 'relative_frequency': count/REPETITIONS})
    with (ANALYSIS_DIRECTORY/'full_interval_histogram_counts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    audit = {'interval': [0.01, 2.0], 'repetitions_per_n_and_method': REPETITIONS,
             'all_four_methods_each_have_1000_selections_in_every_panel': True,
             'endpoint_selections_in_main_histograms': True,
             'histogram_counts_provenance': 'recovered from unmodified original full-interval vector paths; integer rounding checked to 0.0001 count',
             'samples_and_optimization_unchanged': True,
             'vector_content_copied_without_resampling': True, 'panels': panels}
    (ANALYSIS_DIRECTORY/'full_interval_validation.json').write_text(json.dumps(audit, indent=2)+'\n')
    for panel in panels:
        print(panel['output'])
