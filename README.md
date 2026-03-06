# Sleep State GUI and Scoring

PyQt5 GUI plus offline scorer for mouse sleep/wake states.

- GUI: `sleep_state_gui.py`
- Scoring pipeline: `sleep_score.py`

## Install

Use Python 3.9+ and install:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `opencv-python`
- `PyQt5`

The code expects your data paths to be resolved by `organise_paths.find_paths(user_id, exp_id)`.

## Start the GUI

```bash
python sleep_state_gui.py
```

## Quick workflow

1. Enter `User ID` and `Exp ID`.
2. Click **Load Data**.
3. If sleep-state data is missing/outdated, run **Run Sleep Scoring**.
4. Review traces/videos and adjust thresholds.
5. Click **Rescore** (or enable **Auto rescore** in threshold mode).
6. Edit state labels manually if needed.
7. Click **Save scoring**.

## What the GUI can do

### Data loading and scoring

- Loads existing sleep-state file from:
  - `<processed_exp_dir>/sleep_score/sleep_state.pickle`
  - or simulation mode: `<processed_exp_dir>/sleep_score/sleep_state_sim.pickle`
- Runs full scoring from raw/simulated data.
- Supports simulation mode (`sample_data/*.npz`) using the `_sim` suffix.

### Playback and navigation

- Timeline slider at 10 Hz.
- Play/Stop controls.
- Jump to typed time (seconds).
- Jump to current view center.
- Matplotlib zoom/pan toolbar.

### Eye videos and pupil overlay

- Show/hide videos.
- Show/hide pupil overlay.
- Motion smoothing control (seconds).
- Double-click left/right eye video to set a crop (zoom region).
- Double-click again to clear that eye crop.
- Eye crop settings are saved and restored per experiment.

### Plot controls

- Line visibility checkboxes (all enabled by default):
  - Left pupil
  - Right pupil
  - EMG
  - Wheel
  - Motion
  - EEG
  - Low-freq
  - Ratio
  - State
- Pupil diameter traces shown when available (left/right selectable independently, clipped at 80th percentile).
- Wheel and motion axes are centered around zero.

### Spectrogram display controls

- Saturation percentile input for 0-20 Hz view.
- Manual min/max percentile sliders for spectrogram color scaling.
- Manual slider changes override automatic saturation mapping.

### State editing

- **Selection mode**:
  - Drag-select a time span.
  - Assign state via buttons (or numeric shortcuts).
- **Threshold mode**:
  - Drag horizontal threshold lines on plots.
  - Use threshold sliders.
  - Use threshold text boxes.

### Threshold management

- Editable thresholds:
  - EMG RMS threshold
  - Low-frequency threshold
  - Theta ratio threshold
  - Locomotion threshold
- Per-threshold **Revert** button (back to last saved value).
- Global **Revert thresholds** button.
- Slider ranges auto-expand if you manually enter out-of-range values.

### Band configuration

- Editable scoring bands:
  - Delta band low/high
  - Theta band low/high
  - Low-freq max (Hz)
- If band settings change, **Rescore** triggers full scoring rerun.

## Classification logic

Per 10 s epoch (`score_from_epoch_features`), states are assigned in this order:

1. If locomotion (`wheel_speed_mean`) > `locomotion_threshold` -> **Active wake** (`0`)
2. Else if low-freq power >= `low_freq_threshold` -> **NREM** (`2`)
3. Else if theta/delta ratio >= `theta_ratio_threshold` and EMG < `emg_rms_threshold` -> **REM** (`3`)
4. Else if EMG >= `emg_rms_threshold` -> **Active wake** (`0`)
5. Else -> **Quiet wake** (`1`)

Defaults include:

- `locomotion_threshold = 0.1`
- `theta_ratio_threshold` auto-estimated when scoring
- `low_freq_max_hz = 20.0`

## Output file location and format

### Main output

Saved as a Python pickle dictionary:

- Real data: `<processed_exp_dir>/sleep_score/sleep_state.pickle`
- Sim mode: `<processed_exp_dir>/sleep_score/sleep_state_sim.pickle`

### Key fields

- `state_epoch`, `state_epoch_t`: classified state per 10 s epoch and epoch center times.
- `state_10hz`, `state_10hz_t`: nearest-neighbor interpolated state timeline at 10 Hz.
- Thresholds:
  - `emg_rms_threshold`
  - `theta_ratio_threshold` (also mirrored as `theta_delta_ratio_threshold`)
  - `low_freq_threshold` (also mirrored as `delta_power_threshold`)
  - `locomotion_threshold`
- Band metadata:
  - `delta_band`
  - `theta_band`
  - `low_freq_max_hz`
- Traces:
  - `eeg_10hz`, `emg_10hz`, `emg_rms_10hz`, `wheel_10hz`, `face_motion_10hz`
  - each with corresponding `*_t` time vectors
- Spectrogram:
  - `eeg_spectrogram`
  - `eeg_spectrogram_freqs`
  - `eeg_spectrogram_t`
- Cached rescoring features:
  - `epoch_features`
- Eye zoom persistence:
  - `left_video_crop`
  - `right_video_crop`

## Batch scoring (without GUI)

```python
from sleep_score import run_sleep_scoring

sleep_state = run_sleep_scoring(
    user_id="pmateosaparicio",
    exp_id="2025-06-12_04_ESPM135",
    delta_band=(1.0, 4.0),
    theta_band=(5.0, 10.0),
    low_freq_max_hz=20.0,
    simulated_npz=None,     # or path to sample_data/*.npz
    filename_suffix="",    # use "_sim" for simulated
)
```

## Notes

- `state_epoch_t` / `state_10hz_t` are in experiment timeline seconds and align plotting/state assignment to timeline time.
- `sleep_score/face_motion.npy` is used as a cache for computed face-motion data when available.
