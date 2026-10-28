"""Canonical input data and reproducible sampling, independent of saved results."""

import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "inputs"
REFERENCE_SAMPLE_PATH = INPUT_DIRECTORY / "qhat_reference_sample.csv"


def sampling_inputs():
    return json.loads((INPUT_DIRECTORY / "qhat_sampling.json").read_text())


def load_reference_sample():
    metadata = sampling_inputs()["reference_sample"]
    samples = np.loadtxt(REFERENCE_SAMPLE_PATH, skiprows=1)
    if (len(samples) != metadata["sample_size"] or
            hashlib.sha256(samples.tobytes()).hexdigest() != metadata["sample_sha256"]):
        raise ValueError("The fixed KDE input sample does not match its recorded identity.")
    return samples, metadata


def paired_gaussian_sample_bank(repetitions=None, maximum_n=None):
    """Regenerate the recorded block-wise bank, including exact row prefixes."""
    spec = sampling_inputs()["paired_sample_bank"]
    repetitions = spec["sample_bank_shape"][0] if repetitions is None else repetitions
    maximum_n = spec["sample_bank_shape"][1] if maximum_n is None else maximum_n
    if not isinstance(repetitions, int) or not isinstance(maximum_n, int) or min(repetitions, maximum_n) < 1:
        raise ValueError("repetitions and maximum_n must be positive integers.")
    block_size, seed = spec["sample_bank_block_size"], spec["seed"]
    blocks = [np.random.default_rng(seed if i == 0 else np.random.SeedSequence([seed, i])).normal(
        0., np.sqrt(spec["sigma2"]), size=(repetitions, block_size))
        for i in range((maximum_n+block_size-1)//block_size)]
    samples = np.concatenate(blocks, axis=1)[:, :maximum_n]
    if list(samples.shape) == spec["sample_bank_shape"]:
        if hashlib.sha256(samples.tobytes()).hexdigest() != spec["sample_bank_sha256"]:
            raise ValueError("The regenerated KDE sample bank differs from its recorded identity.")
    return samples
