"""Recover plotted values from the saved pre-refactor PDF snapshots.

Run with a Python environment containing pdfplumber and NumPy. Recover
marker centers in drawing order; calibrate the vertical axis from tick
labels, then verify bandwidth values against their original discrete grids.
"""

import json
import re
from pathlib import Path

import numpy as np
import pdfplumber


def main():
    directory = Path(__file__).resolve().parent
    specifications = {
        "figure_3_1": (4, 191, 3.5),
        "figure_3_2": (4, 199, 2.8),
        "figure_3_3": (1, 101, 3.5),
        "figure_3_4": (6, 30, 6.0),
    }
    result = {}
    for name, (series_count, point_count, marker_size) in specifications.items():
        with pdfplumber.open(directory / "before" / f"{name}.pdf") as document:
            page = document.pages[0]
            groups, colors, current = [], [], None
            for curve in page.curves:
                if (
                    curve["width"] > 300
                    and not curve["fill"]
                    and curve["stroke"]
                    and curve["linewidth"] >= 1.5
                ):
                    current = []
                    groups.append(current)
                    colors.append(curve["stroking_color"])
                elif (
                    current is not None
                    and len(current) < point_count
                    and curve["fill"]
                    and curve["stroke"]
                    and abs(curve["width"] - marker_size) < 1e-7
                    and abs(curve["height"] - marker_size) < 1e-7
                ):
                    current.append(
                        [
                            (curve["x0"] + curve["x1"]) / 2,
                            (curve["top"] + curve["bottom"]) / 2,
                        ]
                    )
            assert len(groups) == series_count, (name, len(groups))
            assert all(len(group) == point_count for group in groups)
            points = np.asarray(groups)
            assert np.all(np.diff(points[:, :, 0], axis=1) > 0)
            if name == "figure_3_3":
                span = points[0, -1, 0] - points[0, 0, 0]
                spines = [
                    line
                    for line in page.lines
                    if abs(line["width"] - span) < 1e-4
                    and line["height"] == 0
                    and line["stroking_color"] == 0
                    and abs(line["linewidth"] - 0.8) < 1e-8
                ]
                assert len(spines) == 2
                top, bottom = sorted(line["top"] for line in spines)
                values = 10 ** (
                    np.log10(0.01)
                    + (bottom - points[:, :, 1]) / (bottom - top) * np.log10(200)
                )
            else:
                ticks = [
                    word
                    for word in page.extract_words()
                    if re.fullmatch(r"0\.\d+", word["text"])
                    and word["x1"] < points[0, 0, 0] - 20
                ]
                positions = np.array(
                    [(word["top"] + word["bottom"]) / 2 for word in ticks]
                )
                labels = np.array([float(word["text"]) for word in ticks])
                slope, offset = np.polyfit(positions, labels, 1)
                assert np.max(np.abs(slope * positions + offset - labels)) < 1e-8
                if name == "figure_3_1":
                    # Each plotted Perron profile has sum one.
                    offset = 1 / 191 - slope * points[0, :, 1].mean()
                else:
                    # The first coarse grid resolves the text-baseline offset.
                    size = 11 if name == "figure_3_2" else 100
                    grid = np.geomspace(0.01, 2.0, size)
                    approximate = slope * points[0, 0, 1] + offset
                    anchor = grid[np.argmin(abs(grid - approximate))]
                    offset = anchor - slope * points[0, 0, 1]
                values = slope * points[:, :, 1] + offset
            item = {
                "marker_coordinates": points.tolist(),
                "recovered_values": values.tolist(),
                "line_colors": colors,
            }
            if name != "figure_3_1":
                indices = np.empty(values.shape, dtype=int)
                snapped = np.empty(values.shape)
                for position in range(point_count):
                    size = (
                        11 + 5 * position
                        if name == "figure_3_2"
                        else 100 * (position + 1) if name == "figure_3_4" else 201
                    )
                    grid = np.geomspace(0.01, 2.0, size)
                    for series in range(series_count):
                        indices[series, position] = np.argmin(
                            abs(grid - values[series, position])
                        )
                        snapped[series, position] = grid[indices[series, position]]
                error = float(np.max(abs(snapped - values)))
                assert error < 1e-7, (name, error)
                item.update(
                    selected_indices=indices.tolist(),
                    selected_bandwidths=snapped.tolist(),
                    grid_snap_max_error=error,
                )
            result[name] = item
            print(
                name,
                "recovered",
                values.shape,
                "grid error",
                item.get("grid_snap_max_error", "profile"),
            )
    (directory / "previous_figure_data.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
