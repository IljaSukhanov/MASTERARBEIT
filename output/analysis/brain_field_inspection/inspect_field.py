"""Inspect one public EEG recording without fitting a random-field model.

EDF decoding follows https://www.edfplus.info/specs/edf.html.
Dataset: https://zenodo.org/records/4940267, record eeg2.edf.
"""

from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "eeg2.edf"
EXPECTED_MD5 = "810279775059258155c7dc2616156134"


def read_edf(path):
    content = path.read_bytes()
    assert hashlib.md5(content).hexdigest() == EXPECTED_MD5
    header = content[:256]
    header_bytes = int(header[184:192])
    records = int(header[236:244])
    duration = float(header[244:252])
    ns = int(header[252:256])
    pos = 256
    info = {}
    for key, width in [("label", 16), ("transducer", 80), ("unit", 8),
                       ("physical_min", 8), ("physical_max", 8),
                       ("digital_min", 8), ("digital_max", 8),
                       ("prefilter", 80), ("samples", 8), ("reserved", 32)]:
        info[key] = [content[pos + i * width:pos + (i + 1) * width]
                     .decode("latin-1").strip() for i in range(ns)]
        pos += width * ns
    assert pos == header_bytes
    samples = np.array(info["samples"], dtype=int)
    stride = int(samples.sum())
    assert len(content) == header_bytes + records * stride * 2
    digital = np.frombuffer(content, dtype="<i2", offset=header_bytes)
    digital = digital.reshape(records, stride)
    channels = []
    start = 0
    for i, count in enumerate(samples):
        if info["label"][i].startswith("EEG "):
            dmin, dmax = float(info["digital_min"][i]), float(info["digital_max"][i])
            pmin, pmax = float(info["physical_min"][i]), float(info["physical_max"][i])
            values = digital[:, start:start + count].reshape(-1).astype(float)
            values = (values - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
            assert info["unit"][i] in ("uV", "µV")
            channels.append({"name": info["label"][i].removeprefix("EEG ").removesuffix("-Ref"),
                             "values": values, "fs": count / duration,
                             "prefilter": info["prefilter"][i],
                             "physical_range": [pmin, pmax]})
        start += count
    return channels, records * duration, info


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def main():
    channels, duration, header = read_edf(SOURCE)
    fs = channels[0]["fs"]
    assert all(c["fs"] == fs for c in channels)
    raw = np.stack([c["values"] for c in channels])
    block_seconds = 5
    n = int(fs * block_seconds)
    windows = raw.shape[1] // n
    blocks = raw[:, :windows * n].reshape(len(channels), windows, n)
    maxima = np.max(np.abs(blocks), axis=2)
    snapshots = blocks[:, :, n // 2]
    annotations = []
    for name in ["annotations_2017_A_fixed.csv", "annotations_2017_B.csv", "annotations_2017_C.csv"]:
        with (ROOT / name).open(newline="") as file:
            reader = csv.reader(file)
            ids = next(reader)
            assert ids[1] == "2"
            column = [float(row[1]) if row[1].strip() else float("nan") for row in reader]
        a = np.asarray(column[:int(duration)])
        assert len(a) == int(duration)
        assert np.isin(a, [0, 1]).all()
        annotations.append(a)
    votes = np.sum(annotations, axis=0).astype(int)
    consensus = intervals(votes == 3)
    layout = (ROOT / "standard_1020.elc").read_text().splitlines()
    pstart, lstart = layout.index("Positions") + 1, layout.index("Labels")
    positions = [list(map(float, line.split())) for line in layout[pstart:lstart]]
    names = [line.strip() for line in layout[lstart + 1:] if line.strip()]
    assert len(positions) == len(names)
    coords = dict(zip(names, positions))
    summaries = []
    for c, maxima_c in zip(channels, maxima):
        summaries.append({"channel": c["name"], "min_peak_uv": float(maxima_c.min()),
                          "median_peak_uv": float(np.median(maxima_c)),
                          "p95_peak_uv": float(np.quantile(maxima_c, .95)),
                          "max_peak_uv": float(maxima_c.max()),
                          "prefilter": c["prefilter"]})
    allmax_channel, allmax_window = np.unravel_index(np.argmax(maxima), maxima.shape)
    initial_window = int(consensus[0][0] // block_seconds) if consensus else 0
    payload = {
        "source": "https://zenodo.org/records/4940267", "file": "eeg2.edf", "studyId": 2,
        "sampleRateHz": fs, "durationSeconds": duration, "windowSeconds": block_seconds,
        "windows": windows, "discardedSeconds": duration - windows * block_seconds,
        "initialWindow": initial_window,
        "channels": [{"name": c["name"], "xyz": coords[c["name"]]} for c in channels],
        "maxima": np.round(maxima, 2).tolist(), "snapshots": np.round(snapshots, 2).tolist(),
        "consensusIntervals": consensus,
        "windowVotes": [int(votes[k * block_seconds:(k + 1) * block_seconds].max()) for k in range(windows)],
        "maxPeak": {"channel": channels[allmax_channel]["name"],
                    "window": int(allmax_window), "uv": float(maxima.max())},
    }
    (ROOT / "field_data.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    summary = {"source": payload["source"], "recording": payload["file"],
               "sample_rate_hz": fs, "duration_seconds": duration, "eeg_channels": len(channels),
               "window_seconds": block_seconds, "complete_windows": windows,
               "consensus_seizure_intervals_seconds": consensus,
               "maximum_peak": payload["maxPeak"], "channels": summaries,
               "processing": "Original EDF reference and acquisition filtering retained; no additional filtering, detrending, artefact removal or marginal transformation.",
               "layout": "MNE-Python v1.10.2 standard_1020 reference coordinates; not subject-specific coordinates.",
               "interpretation": "Descriptive inspection of one recording, not a max-stability test or diagnostic analysis."}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (ROOT / "block_maxima.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_start_seconds"] + [c["name"] + "_peak_abs_uv" for c in channels])
        for k in range(windows):
            w.writerow([k * block_seconds] + np.round(maxima[:, k], 6).tolist())
    print(json.dumps({k: v for k, v in summary.items() if k != "channels"}, indent=2))
    print('Per-channel median / p95 / maximum absolute amplitude within 5s windows:')
    for c in summaries:
        print(c["channel"], *(round(c[key], 2) for key in ["median_peak_uv", "p95_peak_uv", "max_peak_uv"]))


if __name__ == "__main__":
    main()
