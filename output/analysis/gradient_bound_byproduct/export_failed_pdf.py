"""Assemble the verified numerical diagnostic images into a review PDF."""
import csv
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TARGET = ROOT / "output/pdf/ar3_sixteen_failed_bound_computations.pdf"


def fitted_image(pdf, path, x, y, width, height):
    reader = ImageReader(str(path))
    image_width, image_height = reader.getSize()
    scale = min(width / image_width, height / image_height)
    drawn_width, drawn_height = image_width * scale, image_height * scale
    pdf.drawImage(reader, x + (width-drawn_width)/2,
                  y + (height-drawn_height)/2, drawn_width, drawn_height)


def footer(pdf, number, width):
    pdf.setFillColorRGB(.38, .43, .48)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(32, 19, "Existing 50-run experiment | pilot embedding k=9 | true order=3 | no new gradient fits")
    pdf.drawRightString(width-32, 19, f"{number} / 17")


def main():
    rows = list(csv.DictReader((OUT / "sixteen_failed_calculations.csv").open()))
    assert len(rows) == 16
    TARGET.parent.mkdir(exist_ok=True)
    width, height = landscape(A4)
    pdf = canvas.Canvas(str(TARGET), pagesize=(width, height))
    pdf.setTitle("All 16 failed gradient-derived bound computations - AR(3)")
    pdf.setAuthor("AR experiment diagnostic")
    fitted_image(pdf, OUT / "sixteen_failed_overview.png", 30, 40, width-60, height-65)
    footer(pdf, 1, width)
    pdf.showPage()
    for index, row in enumerate(rows, 2):
        replication = int(row["replication"])
        pdf.setFillColorRGB(.10, .14, .19)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(32, height-30, f"Failed bound computation {index-1} of 16 | saved run {replication:02d}")
        fitted_image(pdf, OUT / "failed_case_figures" / f"run_{replication:02d}.png",
                     25, 156, width-50, height-205)
        pdf.setFont("Helvetica", 9)
        phi = json.loads(row["coefficients"])
        pdf.drawString(38, 134, "True AR coefficients: (" + ", ".join(f"{x:.7f}" for x in phi) + ")")
        pdf.drawString(38, 117,
                       f"Original width: {float(row['achieved_gradient_width']):.9f}; "
                       f"corrected projection width: {float(row['canonical_projection_width']):.9f}; "
                       f"maximum absolute residual at t={row['maximizing_observation_time']}.")
        pdf.setFont("Helvetica", 8)
        for y, label, column in [(100, "Original direction u", "original_direction"),
                                  (85, "Corrected direction v", "canonical_direction")]:
            values = json.loads(row[column])
            pdf.drawString(38, y, label + " = (" + ", ".join(f"{x:.5f}" for x in values) + ")")
        pdf.setFont("Helvetica", 7)
        pdf.setFillColorRGB(.38, .43, .48)
        pdf.drawString(38, 62,
                       f"Inputs: ar3_order_selection_warm_starts/processes/process_{replication:03d}.csv "
                       f"and replications/replication_{replication:03d}.csv (Padded warm start).")
        pdf.drawString(38, 49,
                       "Calculation data: output/analysis/gradient_bound_byproduct/sixteen_failed_calculations.csv. "
                       "Population reference: thesis Proposition prop:minimum_width.")
        footer(pdf, index, width)
        pdf.showPage()
    pdf.save()
    print(TARGET)


if __name__ == "__main__":
    main()
