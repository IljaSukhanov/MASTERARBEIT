"""Run the localized histogram profile from the main spectral KDE notebook."""

import ast
import json
import os
from pathlib import Path

project = Path(__file__).resolve().parents[5]
os.chdir(project)
path = project / "Code/Code_kde/xx_notebooks_overview/spectral_scale_selection_kde.ipynb"
notebook = json.loads(path.read_text())
cells = {cell["id"]: cell for cell in notebook["cells"]}
namespace = {"__name__": "__main__"}

for cell_id in ("imports", "beta2-configurable-parameters", "beta2-configurable-selections", "beta2-configurable-histograms"):
    tree = ast.parse("".join(cells[cell_id]["source"]))
    if cell_id == "beta2-configurable-parameters":
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "beta2_profile"
                for target in node.targets
            ):
                node.value = ast.Constant("localized")
        ast.fix_missing_locations(tree)
    exec(compile(tree, f"{path}:{cell_id}", "exec"), namespace)
