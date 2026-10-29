"""Make the shared KDE functions available when running a notebook here."""

import os
import sys
from pathlib import Path


CODE_DIRECTORY = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from kde_tools.notebook_setup import setup_notebook

__all__ = ["setup_notebook"]
