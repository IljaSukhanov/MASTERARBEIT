"""Shared inputs for the retained 50-process AR(3) coefficient studies.

The samples belong to ar3_minimum_width_k4.ipynb's seed family. Store their
original floating-point values independently of any particular restart study
so existing fits can still be checked without rerunning a removed experiment.
The input file contains no fitted coefficients, trajectories, or method results.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


AR_DIR = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = AR_DIR / "xx_notebooks_estimation" / "ar3_minimum_width_k4.ipynb"
SOURCE_DATA = AR_DIR / "outputs/csv/ar3_minimum_width_k4"
INPUT_DATA = AR_DIR / "inputs/ar3_coefficient_recovery_50.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@lru_cache(maxsize=1)
def _inputs():
    with np.load(INPUT_DATA, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    assert arrays["series"].shape == (50, 2000)
    assert arrays["true_phi"].shape == (50, 3)
    assert arrays["initial_directions"].shape == (50, 12, 4)
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def source_settings():
    settings = json.loads(str(_inputs()["settings"]))
    # The archived dataset key named the old step-scale setting. Its numeric
    # value now supplies an absolute initial step; no cloud scaling is used.
    archived_key = "initial_step_scale"
    if archived_key in settings:
        settings["initial_step"] = settings.pop(archived_key)
    return settings


def load_process(replication):
    """Return independent copies of one series, true vector, and 12 starts."""
    if not isinstance(replication, (int, np.integer)) or not 1 <= replication <= 50:
        raise ValueError("replication must be an integer from 1 through 50")
    index = replication - 1
    data = _inputs()
    series = data["series"][index].copy()
    expected = str(data["series_sha256"][index])
    if hashlib.sha256(series.tobytes()).hexdigest() != expected:
        raise ValueError(f"Input sample checksum mismatch for process {replication}")
    return series, data["true_phi"][index].copy(), data["initial_directions"][index].copy()


def radii(directions):
    """AR(3) companion radii; undefined coefficient ratios give infinity.

    Uses the same companion construction as functions.is_stationary.
    """
    u = np.atleast_2d(np.asarray(directions, dtype=float))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        phi = -u[:, 1:] / u[:, :1]
    finite = np.all(np.isfinite(phi), axis=1)
    rho = np.full(len(u), np.inf)
    matrices = np.zeros((np.count_nonzero(finite), 3, 3))
    matrices[:, 0, :] = phi[finite]
    matrices[:, 1, 0] = 1
    matrices[:, 2, 1] = 1
    if len(matrices):
        rho[finite] = np.max(np.abs(np.linalg.eigvals(matrices)), axis=1)
    return rho, phi
