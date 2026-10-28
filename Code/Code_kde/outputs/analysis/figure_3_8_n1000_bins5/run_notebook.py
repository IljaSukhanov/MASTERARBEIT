"""Regenerate only Figure 3.8 from the canonical notebook's saved cells."""

import ast
import json
import os
from pathlib import Path

project = Path(__file__).resolve().parents[5]
os.chdir(project)
path = project / "Code/Code_kde/xx_notebooks_overview/spectral_scale_selection_kde.ipynb"
notebook = json.loads(path.read_text())
namespace = {"__name__": "__main__"}
for cell_index in (1, 12, 16, 17):
    tree = ast.parse("".join(notebook["cells"][cell_index]["source"]))
    if cell_index == 1:
        # This Matplotlib-only figure does not need the notebook's optional
        # interactive Plotly imports or their plotting-function module.
        def interactive_import(node):
            if isinstance(node, ast.Import):
                return any(alias.name.startswith("plotly") or
                           alias.name == "kde_tools.plotting_functions"
                           for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                return (node.module or "").startswith("plotly") or node.module == "kde_tools.plotting_functions"
            return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "reload" and node.value.args
                    and isinstance(node.value.args[0], ast.Name)
                    and node.value.args[0].id == "kde_plotting_functions")

        tree.body = [node for node in tree.body if not interactive_import(node)]
    if cell_index == 16:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "beta2_n_sweep_values"
                for target in node.targets
            ):
                node.value = ast.parse("np.array([300, 400, 500, 1000], dtype=int)", mode="eval").body
        ast.fix_missing_locations(tree)
    if cell_index == 17:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "beta2_n_selected_overlay_values"
                for target in node.targets
            ):
                node.value = ast.parse("[300, 500, 1000]", mode="eval").body
        ast.fix_missing_locations(tree)
        # The two all-size overlays belong to the full notebook run.
        tree.body = tree.body[:next(
            i for i, node in enumerate(tree.body)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "beta2_n_overlay"
                for target in node.targets
            )
        )]
    exec(compile(tree, f"{path}:cell{cell_index}", "exec"), namespace)

summary = {
    "N": namespace["beta2_n_sweep_N"],
    "M": namespace["beta2_n_sweep_M"],
    "lambda": namespace["beta2_n_sweep_lambda"],
    "positions_per_bin": namespace["beta2_n_sweep_binning"],
    "sample_sizes": namespace["beta2_n_sweep_values"].tolist(),
    "seed": namespace["beta2_n_sweep_seed"],
    "run_directory": str(namespace["beta2_n_sweep_output_directory"]),
    "panels": {},
}
for n, result in namespace["beta2_n_sweep_spectral_results"].items():
    diagnostics = result["solver_diagnostics"]
    counts = namespace["beta2_n_sweep_results"][n]
    summary["panels"][str(n)] = {
        "samples_sha256": result["samples_sha256"],
        "max_eigenpair_relative_residual": float(diagnostics[:, :, 0].max()),
        "min_relative_spectral_gap": float(diagnostics[:, :, 1].min()),
        "selection_counts": {key: int(counts[key].sum())
                             for key in ("counts", "ise_counts", "qhat_counts")},
    }
Path(__file__).with_name("validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
