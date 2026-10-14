"""Execute the dedicated Figure 3.5 notebook and retain its printed output."""

import contextlib
import io
import json
import os
from pathlib import Path

project = Path(__file__).resolve().parents[5]
os.chdir(project)
path = project / "Code/Code_kde/figure_3_5_bandwidth_distribution_n1000.ipynb"
notebook = json.loads(path.read_text())
namespace = {"__name__": "__main__"}
count = 0


class Tee(io.StringIO):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def write(self, value):
        self.original.write(value)
        self.original.flush()
        return super().write(value)

    def flush(self):
        self.original.flush()


for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    import sys
    output = Tee(sys.stdout)
    with contextlib.redirect_stdout(output):
        exec(compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec"), namespace)
    count += 1
    cell["execution_count"] = count
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(keepends=True)}]
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
