# Sleep State GUI and Scoring

PyQt5 GUI plus offline scorer for mouse sleep/wake states. The pipeline lives in `sleep_score.py`; interactive review/editing is in `sleep_state_gui.py`.

## Install
- Python 3.9+ with: numpy, scipy, pandas, matplotlib, scikit-learn, opencv-python, PyQt5.
- Code expects the experiment folder structure resolved by `organise_paths.find_paths(user_id, exp_id)` (returns processed/raw roots and `recordings/` contents such as `ephys.npy`, `wheel.pickle`, `eye_frame_times.npy`, eye videos).

## Run the GUI
```bash
python sleep_state_gui.py
```
Steps in the app:
- Enter `User ID` and `Exp ID`, then click **Load Data** (loads existing `sleep_score/sleep_state*.pickle` if present; otherwise offers to run scoring).
- Use **Run Sleep Scoring** to (re)generate the pickle. Delta/theta bands can be edited before scoring.
- Playback: play/stop buttons, time box to jump, playhead slider at 10 Hz.
- Eye videos: toggle show/hide and pupil overlay; motion smoothing slider affects plotted face-motion trace.
- Modes: **Selection mode** (default) lets you drag a time span (cyan/magenta handles) and assign a state via buttons/number keys. **Threshold mode** lets you drag horizontal lines or sliders for EMG, low-frequency power, theta/delta ratio, and locomotion; optional auto-rescore applies changes instantly.
- Rescoring: **Rescore** reruns classification using cached epoch features; if delta/theta bands changed, a full run of `run_sleep_scoring` is triggered. **Show distributions** opens draggable histograms of thresholds.
- Saving: **Save scoring** writes the updated states/thresholds back to `sleep_score/sleep_state*.pickle` (respecting simulated vs real suffix).

## Batch scoring (no GUI)
```python
from sleep_score import run_sleep_scoring

sleep_state = run_sleep_scoring(
    user_id="pmateosaparicio",
    exp_id="2025-06-12_04_ESPM135",
    delta_band=(1.0, 4.0),
    theta_band=(5.0, 10.0),
    simulated_npz=None,           # or path to sample_data/*.npz
    filename_suffix="",           # e.g., "_sim" for simulated
)
```
Outputs go to `sleep_score/sleep_state{suffix}.pickle` plus QC figures in `sleep_score/figs/` (set `MAKE_QC_FIGS=False` in code to skip).

Key fields in the pickle:
- Epoch states/time: `state_epoch`, `state_epoch_t` (one per 10 s epoch).
- 10 Hz states/time: `state_10hz`, `state_10hz_t` (interpolated for plotting).
- Thresholds used: `emg_rms_threshold`, `theta_ratio_threshold`, `low_freq_threshold`, `locomotion_threshold`.
- Traces: EMG/EEG/wheel/face motion at 10 Hz with time vectors.
- Spectrogram: `eeg_spectrogram`, `eeg_spectrogram_freqs`, `eeg_spectrogram_t`.
- Epoch features: `epoch_features` (used for fast rescoring in the GUI).
- Band metadata: `delta_band`, `theta_band`.

## State logic (per epoch)
Rules in `score_from_epoch_features`:
1. If locomotion > locomotion_threshold → active wake (0).
2. Else if low_freq_power >= low_freq_threshold → NREM (2).
3. Else if theta/delta ratio >= theta_ratio_threshold and EMG < emg_rms_threshold → REM (3).
4. Else if EMG >= emg_rms_threshold → active wake (0).
5. Else → quiet wake (1).

## Tips
- Simulated data: choose a `.npz` from `sample_data/` via the GUI’s “Simulated EEG/EMG” checkbox (uses `_sim` suffix).
- Face motion is computed from the left-eye video and cached to `sleep_score/face_motion.npy` when available.
