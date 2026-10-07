# First inspection of an observed EEG field

This is an exploratory example from a public dataset, not a user-supplied recording and not a fitted max-stable model.

## Data and provenance

- Dataset: Stevenson, Tapani, Lauronen and Vanhatalo, [Helsinki neonatal EEG dataset, version 4](https://zenodo.org/records/4940267).
- Recording: `eeg2.edf`, study ID 2. Verified MD5: `810279775059258155c7dc2616156134`.
- The EDF contains 19 channels labelled EEG, plus ECG and respiratory channels. Only the 19 EEG channels are shown.
- The recording is sampled at 256 Hz and lasts 3,761 seconds (62 min 41 s).
- Coordinates: [MNE-Python v1.10.2 standard_1020 reference montage](https://github.com/mne-tools/mne-python/blob/v1.10.2/mne/channels/data/montages/standard_1020.elc). These are standard reference coordinates, not this infant's measured electrode coordinates. The visualization uses their x/y projection with equal spatial scaling.
- Signal conversion follows the [EDF specification](https://www.edfplus.info/specs/edf.html): physical values are obtained from each channel's digital and physical calibration ranges. Header length, data length, units and sampling rates are checked.
- All three annotation CSV files were downloaded and checked against the Zenodo checksums. Their first row contains study identifiers; study ID 2 is the second column. The annotations are not used to diagnose the displayed patterns.

## Displayed quantities

Write the recorded voltage as V_t(s), where t labels an electrode location and s is recording time. For consecutive five-second intervals I_k, the proposed derived field is

    M_t^(k) = max_{s in I_k} |V_t(s)|.

This choice of feature and five-second window is an exploratory proposal. It is not claimed to have been validated for the paper's model.

The left map displays the exact recorded sample at each window's midpoint. The right map displays the maximum absolute voltage within that window at each channel. Different sites can attain their maxima at different times within a window.

There are 752 complete windows; the final second is excluded. Original acquisition filtering and reference are retained. No additional filtering, artifact rejection, centering, marginal standardization, spatial interpolation or statistical fitting is performed.

Colour limits remain fixed as the time window changes. Instantaneous voltages have a signed linear colour scale. Maxima use a log(1 + amplitude) scale to keep both small and large values visible. At screen resolutions too narrow to display every five-second window separately, each overview pixel bin retains the largest constituent maximum. The two spatial maps always display the selected individual five-second window.

## Initial numerical observations

These are descriptive calculations for this recording only; see `summary.json` and `block_maxima.csv`.

- The largest absolute EEG sample is approximately 804.904 microvolts at T3, within 58:45–58:50 (zero-based window index 705).
- The median five-second maximum at Fz is approximately 1.06 microvolts, versus 81.63 at C3 and 140.30 at T6. These marked differences motivate investigating the recording reference, signal quality and marginal scales before considering a common field model. The exact cause is not established by this inspection.
- Several channel-wise largest peaks occur at the beginning of the recording, which should be inspected for recording transients before model fitting.
- No second in this recording is marked as seizure by all three annotators. This statement concerns agreement in the supplied annotations, not a clinical conclusion.

## Connection to the paper

[Extrapolation of max-stable random fields with Fréchet marginals](https://arxiv.org/html/2604.16206v1), Section 2.1, defines the model assumptions; Section 5 gives the predictor. Here, M is only a candidate observed field from which a standardized X might later be constructed. Its joint max-stability and stationarity have not been tested. Marginal transformation alone would not establish joint max-stability, and consecutive time windows are not asserted to be independent.

## Reproducibility

Run `inspect_field.py` with NumPy and the four downloaded source files plus the reference montage in this directory. It writes `field_data.json`, `summary.json` and `block_maxima.csv`. No thesis files are modified.

The interactive view was checked in the in-app browser at 768- and 390-pixel viewport widths. Changing to window index 705 updated the time interval, both maps, the overview marker and the displayed T3 maximum to 804.9 microvolts. The browser reported no rendering errors during the check.
