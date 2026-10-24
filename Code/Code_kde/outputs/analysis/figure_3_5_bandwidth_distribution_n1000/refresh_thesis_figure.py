"""Render the canonical Figure 3.5 cell from its verified N=1000 selections."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

project = Path(__file__).resolve().parents[5]
code = project / "Code" / "Code_kde"
os.chdir(project)
sys.path.insert(0, str(code))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(project / ".matplotlib-cache"))

import matplotlib.pyplot as plt
from kde_tools.bandwidths import (
    bandwidth_grid_linear_in_psi_beta,
    gaussian_mise_oracle_bandwidth,
)
from kde_tools.pdf_export import enable_automatic_pdf_saving

notebook_path = code / "xx_notebooks_overview" / "spectral_scale_selection_kde.ipynb"
notebook = json.loads(notebook_path.read_text())
cells = {cell["id"]: cell for cell in notebook["cells"]}
chapter = project / "Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex"
chapter_before = hashlib.sha256(chapter.read_bytes()).hexdigest()
thesis_directory = code / "outputs/images/thesis"
figure_path = thesis_directory / "distribution_of_chosen_bandwidths.pdf"
old_figure_hash = hashlib.sha256(figure_path.read_bytes()).hexdigest()

enable_automatic_pdf_saving("spectral_scale_selection_kde")
namespace = {
    "np": np, "plt": plt, "seed": 20260818,
    "code_directory": code,
    "bandwidth_grid_linear_in_psi_beta": bandwidth_grid_linear_in_psi_beta,
    "gaussian_mise_oracle_bandwidth": gaussian_mise_oracle_bandwidth,
    "kde_thesis_figure_dir": thesis_directory,
}
exec("".join(cells["beta2-configurable-parameters"]["source"]), namespace)
run_directory = code / "outputs/data/figure_3_5_bandwidth_distribution_n1000"
with np.load(run_directory / "checkpoint.npz", allow_pickle=False) as checkpoint:
    settings = json.loads(str(checkpoint["settings_json"].item()))
    assert np.all(checkpoint["completed"])
    assert np.array_equal(checkpoint["h_grid"], namespace["beta2_h_grid"])
    selected = checkpoint["selected_indices"].copy()
for key, variable in (
    ("n", "beta2_n"), ("N", "beta2_N"), ("M", "beta2_M"),
    ("lambda", "beta2_lambda"), ("seed", "beta2_seed"),
    ("h_min", "beta2_h_min"), ("h_max", "beta2_h_max"),
    ("fft_half_width", "beta2_half_width"), ("fft_spatial_points", "beta2_spatial_points"),
):
    assert settings[key] == namespace[variable]
assert (namespace["beta2_N"], namespace["beta2_binning"]) == (1000, 5)
for column, variable in enumerate((
    "beta2_selection_counts", "beta2_ise_selection_counts", "beta2_qhat_selection_counts",
)):
    namespace[variable] = np.bincount(selected[:, column], minlength=namespace["beta2_N"])

exec("".join(cells["beta2-configurable-histograms"]["source"]), namespace)
assert len(namespace["beta2_bin_starts"]) == 200
assert np.all(namespace["beta2_binned_counts"].sum(axis=1) == 1000)
assert hashlib.sha256(chapter.read_bytes()).hexdigest() == chapter_before
new_figure_hash = hashlib.sha256(figure_path.read_bytes()).hexdigest()
assert new_figure_hash != old_figure_hash
assert r"{Code/Code_kde/outputs/images/thesis/distribution_of_chosen_bandwidths.pdf}" in chapter.read_text()
record = {
    "canonical_notebook": str(notebook_path),
    "thesis_figure": str(figure_path), "N": 1000, "binning": 5,
    "bins_per_distribution": 200, "M": 1000, "n": 200, "lambda": 1.0,
    "count_totals": namespace["beta2_binned_counts"].sum(axis=1).tolist(),
    "thesis_includegraphics_targets_updated_pdf": True,
    "thesis_source_and_caption_unchanged": True,
    "thesis_source_sha256": chapter_before,
    "old_pdf_sha256": old_figure_hash, "new_pdf_sha256": new_figure_hash,
}
(Path(__file__).parent / "thesis_update_validation.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(json.dumps(record, indent=2))
