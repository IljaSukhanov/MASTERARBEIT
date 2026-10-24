"""Shared, resumable continuous selectors for paired sample-size experiments.

Q-hat and Q0 name the same objective. Their notebook views use one cache,
with signed endpoint flags internally and adapters at the notebook boundary.
Historical imports retain their producer and require verified source versions,
search settings, and regenerated sample hashes.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from .continuous_minimizers import CONTINUOUS_SELECTOR_VERSION, METHODS, R_VALUES, continuous_ise_qr_minima
from .qhat import QHAT_VERSION, continuous_qhat_minimum
from .spectral_sweeps import _atomic_checkpoint, _sample_hash


CONTINUOUS_RUNNER_VERSION = "shared_continuous_selectors_v1"
FIELDS = ("h", "objective", "log_derivative", "log_curvature", "boundary", "interior_candidates")
_QHAT_SOURCE = "1491254b17be17762329db34cdbe5f1b1b185df857d4a23dc4a3fc8fdae8bc56"
_QR_SOURCE = "700b1392d667cadb0532b276643a41377e78c6f5bd26ca647a992b2d5ccdd597"


def _validate(values, done, methods, settings):
    if values.shape != (*done.shape, len(FIELDS)) or done.dtype != np.dtype(bool):
        raise ValueError("Continuous checkpoint arrays have invalid shapes or flags.")
    rows = values[done]
    if (not np.all(np.isfinite(rows)) or np.any(rows[:, 0] < settings["h_min"])
            or np.any(rows[:, 0] > settings["h_max"])
            or np.any(~np.isin(rows[:, 4], [-1, 0, 1]))
            or np.any(rows[:, 5] < 0) or np.any(rows[:, 5] != np.floor(rows[:, 5]))):
        raise ValueError("Continuous checkpoint contains invalid completed values.")
    if len(methods) != done.shape[1]:
        raise ValueError("Continuous checkpoint method ordering differs.")


def _legacy(path, samples, settings):
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved["configuration_json"].item()))
        qhat = config.get("version") == QHAT_VERSION
        if qhat:
            if config.get("source_sha256") != _QHAT_SOURCE:
                raise ValueError("Unrecognized historical Q-hat implementation.")
            previous = config["sampling_configuration"]["settings"]
            bank = np.random.default_rng(previous["seed"]).normal(
                0, np.sqrt(previous["sigma2"]),
                size=(previous["repetitions"], previous["maximum_sample_size"]))
            n = int(saved["sample_size"])
            methods = ("Q0",)
            if tuple(saved["fields"]) != ("h", "qhat", *FIELDS[2:]):
                raise ValueError("Historical Q-hat fields differ.")
            values = saved["values"].copy()[:, None, :]
            # The old fast notebook reports a boolean endpoint flag.
            values[:, 0, 4] = np.where(values[:, 0, 4] == 0, 0,
                                      np.where(values[:, 0, 0] == config["h_min"], -1, 1))
        elif config.get("version") == CONTINUOUS_SELECTOR_VERSION:
            if config.get("source_sha256") != {"qhat.py": _QHAT_SOURCE,
                                               "continuous_minimizers.py": _QR_SOURCE}:
                raise ValueError("Unrecognized historical continuous-selector implementation.")
            if config.get("r_values") != R_VALUES or config.get("methods") != list(METHODS):
                raise ValueError("Historical objective definitions differ.")
            shape = config["sample_bank_shape"]
            if config.get("sample_bank_generation") == "fixed_1000_column_blocks_preserving_original_prefix_v1":
                block_size = config["sample_bank_block_size"]
                bank = np.concatenate([
                    np.random.default_rng(config["seed"] if block == 0 else
                                          np.random.SeedSequence([config["seed"], block])).normal(
                        0, np.sqrt(config["sigma2"]), size=(shape[0], block_size))
                    for block in range((shape[1] + block_size - 1) // block_size)
                ], axis=1)[:, :shape[1]]
            elif "sample_bank_generation" not in config:
                bank = np.random.default_rng(config["seed"]).normal(0, np.sqrt(config["sigma2"]), size=shape)
            else:
                raise ValueError("Unknown historical sample-bank generator.")
            n = int(saved["n"])
            methods = tuple(saved["methods"])
            if methods != METHODS or tuple(saved["fields"]) != FIELDS:
                raise ValueError("Historical continuous method or field ordering differs.")
            values = saved["values"].copy()
        else:
            raise ValueError("Unrecognized historical continuous checkpoint version.")
        if (_sample_hash(bank) != config["sample_bank_sha256"] or bank.shape[0] < len(samples)
                or n != samples.shape[1] or not np.array_equal(bank[:len(samples), :n], samples)):
            raise ValueError("Historical continuous checkpoint samples differ.")
        for key in ("h_min", "h_max", "scan_points", "spatial_points", "half_width"):
            if config[key] != settings[key]:
                raise ValueError("Historical continuous search settings differ.")
        if not qhat and config["sigma2"] != settings["sigma2"]:
            raise ValueError("Historical ISE truth variance differs.")
        completed = int(saved["completed"])
    if len(values) < len(samples) or not 0 <= completed <= len(values):
        raise ValueError("Invalid historical continuous completion count.")
    done = np.zeros(values.shape[:2], dtype=bool)
    done[:completed] = True
    _validate(values, done, methods, settings)
    return methods, values[:len(samples)], done[:len(samples)], config["version"]


def run_continuous_point(n, full_samples, settings, methods, output_directory, *,
                         cache_root=None, configuration=None, reuse=True, workers=4,
                         import_directories=()):
    """Return M x methods x fields, sharing Q0 across both notebook views.

    Search settings contain h_min/h_max, sigma2, scan_points, half_width and
    spatial_points. Cache identity uses the actual sample prefixes, so a larger
    paired sample bank can reuse smaller-n calculations without relabeling it.
    """
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Choose unique continuous selectors from METHODS.")
    bank = np.asarray(full_samples, dtype=float)
    if bank.ndim != 2 or not len(bank) or not np.all(np.isfinite(bank)) or not 1 <= n <= bank.shape[1]:
        raise ValueError("Continuous samples must be a finite bank containing the requested sample size.")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    numerical = {key: settings[key] for key in
                 ("h_min", "h_max", "sigma2", "scan_points", "half_width", "spatial_points")}
    if (not 0 < numerical["h_min"] < numerical["h_max"] < np.inf
            or not 0 < numerical["sigma2"] < np.inf
            or numerical["scan_points"] < 5 or numerical["half_width"] <= 0
            or numerical["spatial_points"] < 16 or numerical["spatial_points"] % 2):
        raise ValueError("Invalid continuous interval, truth variance, or search settings.")
    samples = np.ascontiguousarray(bank[:, :n])
    identity = {"runner": CONTINUOUS_RUNNER_VERSION, "settings": numerical,
                "samples_sha256": _sample_hash(samples), "n": int(n), "M": len(samples),
                "qhat_version": QHAT_VERSION, "qr_version": CONTINUOUS_SELECTOR_VERSION}
    identity_json = json.dumps(identity, sort_keys=True)
    fingerprint = hashlib.sha256(identity_json.encode()).hexdigest()[:16]
    cache = (Path(cache_root) if cache_root is not None else Path(output_directory) / "shared_cache") / f"n{n}" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "checkpoint.npz"
    values = np.full((len(samples), len(METHODS), len(FIELDS)), np.nan)
    done = np.zeros(values.shape[:2], dtype=bool)
    producers = np.full(done.shape, "", dtype="U64")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["identity_json"].item()) != identity_json or tuple(saved["methods"]) != METHODS or tuple(saved["fields"]) != FIELDS:
                raise ValueError("Shared continuous checkpoint identity differs.")
            values, done, producers = (saved[key].copy() for key in ("values", "completed", "producers"))
        if done.shape != (len(samples), len(METHODS)) or producers.shape != done.shape:
            raise ValueError("Shared continuous checkpoint shape differs.")
        _validate(values, done, METHODS, numerical)
    imports = []
    if reuse:
        for directory in dict.fromkeys(map(Path, import_directories)):
            source = directory / f"n{n}.npz"
            if not source.is_file():
                continue
            try:
                old_methods, old_values, old_done, producer = _legacy(source, samples, numerical)
            except (ValueError, KeyError, OSError) as error:
                imports.append({"source": str(source), "status": "skipped", "reason": str(error)})
                continue
            imported = 0
            for old_column, method in enumerate(old_methods):
                column = METHODS.index(method)
                rows = old_done[:, old_column] & ~done[:, column]
                values[rows, column] = old_values[rows, old_column]
                done[rows, column] = True
                producers[rows, column] = producer
                imported += int(rows.sum())
            imports.append({"source": str(source), "status": "imported", "selections": imported})
    columns = [METHODS.index(method) for method in methods]
    if not reuse:
        done[:, columns] = False
        values[:, columns] = np.nan
        producers[:, columns] = ""
    reused = int(done[:, columns].sum())
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(configuration, sort_keys=True)

    def save():
        _atomic_checkpoint(path, identity_json=identity_json, methods=np.array(METHODS), fields=np.array(FIELDS),
                           values=values, completed=done, producers=producers)
        _atomic_checkpoint(output / f"n{n}.npz", configuration_json=config_json, identity_json=identity_json,
                           n=n, methods=np.array(methods), fields=np.array(FIELDS), values=values[:, columns],
                           completed=done[:, columns], producers=producers[:, columns])

    def select(repetition):
        missing = [method for method in methods if not done[repetition, METHODS.index(method)]]
        result = {}
        if "Q0" in missing:
            q = continuous_qhat_minimum(samples[repetition], **{k: v for k, v in numerical.items() if k != "sigma2"})
            result["Q0"] = dict(q, objective=q["qhat"], boundary=(
                -1 if q["h"] == numerical["h_min"] else 1) if q["boundary"] else 0)
        other = tuple(method for method in missing if method != "Q0")
        if other:
            result.update(continuous_ise_qr_minima(samples[repetition], **numerical, methods=other))
        return repetition, result

    pending = [r for r in range(len(samples)) if not np.all(done[r, columns])]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for repetition, result in pool.map(select, pending):
                for method, selection in result.items():
                    column = METHODS.index(method)
                    values[repetition, column] = [selection[field] for field in FIELDS]
                    done[repetition, column] = True
                    producers[repetition, column] = QHAT_VERSION if method == "Q0" else CONTINUOUS_SELECTOR_VERSION
                if (repetition + 1) % 25 == 0:
                    save()
    finally:
        save()
    _validate(values, done, METHODS, numerical)
    return {"values": values[:, columns].copy(), "methods": methods, "fields": FIELDS,
            "cache_directory": cache, "reused_selections": reused,
            "computed_selections": len(samples) * len(methods) - reused, "imports": imports}
