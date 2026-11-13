"""Check thesis scripts start when launched outside the Code directory."""

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_thesis_scripts_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            code = workspace / 'Code'
            shutil.copytree(ROOT/'kde_tools', code/'kde_tools',
                            ignore=shutil.ignore_patterns('__pycache__'))
            scripts = [
                'experiments/repeated_alpha_evolution/run_experiment.py',
                'experiments/repeated_alpha_evolution/plot_results.py',
                'experiments/outlook_four_lambdas.py',
            ]
            for name in scripts:
                destination = code/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, destination)
            checkpoint = Path('outputs/data/repeated_alpha_evolution/n400/paired_selections.npz')
            (code/checkpoint).parent.mkdir(parents=True)
            shutil.copy2(ROOT/checkpoint, code/checkpoint)
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(workspace/'matplotlib'),
                       PYTHONDONTWRITEBYTECODE='1')
            for name in (scripts[0], scripts[2]):
                # run_name prevents expensive full simulations while exercising
                # imports and initialization exactly as the script does.
                completed = subprocess.run(
                    [sys.executable, '-c', 'import runpy,sys; runpy.run_path(sys.argv[1])', str(code/name)],
                    cwd=workspace, env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            completed = subprocess.run([sys.executable, str(code/scripts[1])], cwd=workspace,
                                       env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in ('alpha_0.pdf', 'alpha_1_2.pdf', 'alpha_1.pdf', 'alpha_3_2.pdf'):
                self.assertTrue((code/'outputs/pdfs/repeated_alpha_evolution'/name).read_bytes().startswith(b'%PDF-'))


if __name__ == '__main__':
    unittest.main()
