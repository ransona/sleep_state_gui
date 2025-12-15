import os
import time
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional, Tuple

import organise_paths


# ---------------------------------------------------------------------
# types / helper hints
# ---------------------------------------------------------------------

def _as_float_array(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

DEFAULT_USER_ID = "pmateosaparicio"
DEFAULT_EXP_ID = "2025-06-12_04_ESPM135"

FS_EPHYS = 1000                   # Hz, original sampling rate
EPOCH_LEN_SEC = 10.0
EPOCH_LEN_SAMPLES = int(FS_EPHYS * EPOCH_LEN_SEC)

TARGET_FS = 10                    # Hz, for downsampled traces
MAKE_QC_FIGS = True               # set False to skip plotting

# EMG RMS
RMS_WINDOW_SEC = 1.0              # 1 s RMS window
# spectrogram
SPEC_WINDOW_SEC = 5.0             # 5 s window
SPEC_STRIDE_SEC = 2.0             # 2 s stride

# thresholds / smoothing parameters
WHEEL_SPEED_THRESHOLD = 1.0       # abs(wheel) > this -> movement
EMG_SD_MULT = 3.0                 # EMG thr = mean + EMG_SD_MULT * sd (low component)
THETA_RATIO_SD_MULT = 2.0         # theta/delta thr = mean + THETA_RATIO_SD_MULT * sd
SAVGOL_BASE_WIN = 11              # base window for Savitzky–Golay (adaptive)

# state code mapping
STATE_ACTIVE_WAKE = 0
STATE_QUIET_WAKE = 1
STATE_NREM = 2
STATE_REM = 3

STATE_LABELS = {
    STATE_ACTIVE_WAKE: "active wake",
    STATE_QUIET_WAKE: "quiet wake",
    STATE_NREM: "nrem",
    STATE_REM: "rem",
}

DEFAULT_DELTA_BAND = (1.0, 10.0)
DEFAULT_THETA_BAND = (5.0, 10.0)


# ---------------------------------------------------------------------
# utility functions
# ---------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def moving_rms(x, window_size):
    """
    Compute RMS over a sliding window.

    - Uses convolution internally.
    - Returns an array with the SAME length as x.
    - The central region contains valid window-centred RMS values.
    - The edges are filled by nearest valid RMS (edge replication).
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > n:
        # degenerate case: single RMS for entire trace
        rms_val = np.sqrt(np.mean(x ** 2))
        return np.full_like(x, rms_val, dtype=float)

    squared = x ** 2
    window = np.ones(window_size, dtype=float) / window_size
    rms_valid = np.sqrt(np.convolve(squared, window, mode="valid"))  # length N - W + 1

    rms = np.empty_like(x, dtype=float)
    offset = window_size // 2
    start = offset
    end = start + rms_valid.size

    # central region
    rms[start:end] = rms_valid
    # edges replicated
    rms[:start] = rms_valid[0]
    rms[end:] = rms_valid[-1]

    return rms


def apply_eeg_emg_filters(eeg_raw, emg_raw, fs):
    """
    apply:
      - 50 Hz notch
      - 100–450 Hz notch harmonics
      - eeg: 1–30 Hz band-pass
      - emg: 100–499 Hz band-pass
    """
    print("  - applying notch and band-pass filters to eeg/emg")
    notch_freq = 50.0
    q_factor = 20.0

    # base notch
    b_notch, a_notch = signal.iirnotch(notch_freq, q_factor, fs)
    eeg_notched = filtfilt(b_notch, a_notch, eeg_raw)
    emg_notched = filtfilt(b_notch, a_notch, emg_raw)

    # additional harmonics
    eeg_series = eeg_notched
    emg_series = emg_notched
    for harmonic in range(100, 451, 50):
        b_h, a_h = signal.iirnotch(harmonic, q_factor, fs)
        eeg_series = filtfilt(b_h, a_h, eeg_series)
        emg_series = filtfilt(b_h, a_h, emg_series)

    # band-pass filters
    eeg_b = butter(2, [1, 30], btype="bandpass", fs=fs)
    emg_b = butter(2, [100, 499], btype="bandpass", fs=fs)

    eeg_filtered = filtfilt(eeg_b[0], eeg_b[1], eeg_series)
    emg_filtered = filtfilt(emg_b[0], emg_b[1], emg_series)

    return eeg_filtered, emg_filtered


def _adaptive_savgol(x, base_window, poly=3):
    """
    Adaptive Savitzky–Golay smoothing:

    - ensures window length is odd
    - ensures window length <= len(x)
    - if data too short, returns x unchanged
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return x.copy()

    # ensure odd window <= n
    win = min(base_window, n if n % 2 == 1 else n - 1)
    if win < poly + 2:
        return x.copy()

    return signal.savgol_filter(x, window_length=win, polyorder=poly, mode="nearest")


def downsample_to_target_fs(x, fs_original, fs_target, t=None):
    """
    downsample by integer factor (fs_original / fs_target).
    optionally downsample matching time vector t (same length as x).
    """
    if fs_original % fs_target != 0:
        raise ValueError("fs_original must be an integer multiple of fs_target")

    factor = fs_original // fs_target
    x_ds = x[::factor]
    if t is not None:
        t_ds = t[::factor]
        return x_ds, t_ds
    return x_ds


def epoch_total_power(signal_epochs, fs):
    """
    Compute total power per epoch using periodogram.

    signal_epochs: shape (n_epochs, epoch_len_samples)
    Returns: shape (n_epochs,)
    """
    n_epochs = signal_epochs.shape[0]
    power = np.empty(n_epochs, dtype=float)
    print(f"  - computing total power in {n_epochs} epochs")
    step = max(1, n_epochs // 20)

    for i in range(n_epochs):
        _, pxx = signal.periodogram(signal_epochs[i, :], fs=fs)
        power[i] = np.sum(np.abs(pxx))
        if (i + 1) % step == 0 or (i + 1) == n_epochs:
            progress = (i + 1) / n_epochs * 100
            print(f"\r    progress (power): {progress:5.1f}%", end="")
    print()
    return power


def _validate_band(band, name, fs):
    if band is None:
        return None
    if len(band) != 2:
        raise ValueError(f"{name} must be a pair (low, high)")
    low, high = float(band[0]), float(band[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{name} values must be finite")
    if low <= 0 or high <= 0:
        raise ValueError(f"{name} values must be positive")
    if high <= low:
        raise ValueError(f"{name} high must be greater than low")
    if high >= fs / 2:
        raise ValueError(f"{name} high must be less than Nyquist ({fs/2:.1f} Hz)")
    return (low, high)


def _report_progress(callback, step_idx, total_steps, message):
    if callback is None:
        return
    try:
        fraction = float(step_idx) / float(total_steps)
    except Exception:
        fraction = 0.0
    try:
        callback(fraction, str(message))
    except Exception:
        pass


def _compute_face_motion_series(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  - face motion: unable to open video {video_path}")
        return None

    diffs = []
    prev_gray = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            diffs.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(float(np.sum(diff)))
        prev_gray = gray

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    if not diffs:
        return None
    return np.asarray(diffs, dtype=float)


def _load_or_compute_face_motion(exp_dir_processed, exp_id, sleep_score_folder):
    cache_path = os.path.join(sleep_score_folder, "face_motion.npy")
    recordings_dir = os.path.join(exp_dir_processed, "recordings")
    eye_times_path = os.path.join(recordings_dir, "eye_frame_times.npy")
    video_path = os.path.join(exp_dir_processed, f"{exp_id}_eye1_left.avi")

    if not os.path.isfile(eye_times_path):
        print("  - face motion: eye_frame_times.npy not found, skipping motion metric")
        return None, None
    try:
        eye_times = np.load(eye_times_path).astype(float)
    except Exception as exc:
        print(f"  - face motion: unable to load eye_frame_times.npy ({exc}), skipping")
        return None, None

    if os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True).item()
            motion = np.asarray(cached.get("motion"), dtype=float)
            motion_t = np.asarray(cached.get("t"), dtype=float)
            if motion.size > 0 and motion_t.size == motion.size:
                return motion_t, motion
        except Exception:
            print("  - face motion: cached file unavailable, recomputing")

    if not os.path.isfile(video_path):
        print(f"  - face motion: video {video_path} not found, skipping")
        return None, None

    max_frames = int(eye_times.size)
    motion = _compute_face_motion_series(video_path, max_frames=max_frames)
    if motion is None:
        return None, None

    if motion.size != eye_times.size:
        n = min(motion.size, eye_times.size)
        motion = motion[:n]
        eye_times = eye_times[:n]

    try:
        np.save(cache_path, {"t": eye_times, "motion": motion})
    except Exception as exc:
        print(f"  - face motion: warning, unable to save cache ({exc})")
    return eye_times, motion


# ---------------------------------------------------------------------
# epoch feature helpers / scoring hooks
# ---------------------------------------------------------------------

def build_epoch_feature_dict(
    epoch_time,
    theta_power_epochs,
    delta_power_epochs,
    emg_rms_mean_by_epoch,
    wheel_speed_mean_by_epoch,
    *,
    theta_delta_ratio=None,
    theta_delta_ratio_smoothed=None,
    delta_power_smoothed=None,
):
    """
    Bundle epoch-level summary data required for reclassification.
    """
    epoch_time = _as_float_array(epoch_time).astype(np.float64)
    theta_power_epochs = _as_float_array(theta_power_epochs).astype(np.float32)
    delta_power_epochs = _as_float_array(delta_power_epochs).astype(np.float32)
    emg_rms_mean_by_epoch = _as_float_array(emg_rms_mean_by_epoch).astype(np.float32)
    wheel_speed_mean_by_epoch = _as_float_array(
        wheel_speed_mean_by_epoch
    ).astype(np.float32)

    if theta_delta_ratio is None:
        ratio = np.divide(
            theta_power_epochs,
            delta_power_epochs,
            out=np.full_like(theta_power_epochs, np.nan),
            where=np.isfinite(delta_power_epochs) & (delta_power_epochs != 0),
        )
    else:
        ratio = _as_float_array(theta_delta_ratio)

    if theta_delta_ratio_smoothed is None:
        ratio_smoothed = _adaptive_savgol(ratio, SAVGOL_BASE_WIN).astype(np.float32)
    else:
        ratio_smoothed = _as_float_array(theta_delta_ratio_smoothed).astype(np.float32)

    if delta_power_smoothed is None:
        delta_power_smoothed = _adaptive_savgol(
            delta_power_epochs, SAVGOL_BASE_WIN
        ).astype(np.float32)
    else:
        delta_power_smoothed = _as_float_array(delta_power_smoothed).astype(np.float32)

    return {
        "epoch_time": epoch_time,
        "theta_power": theta_power_epochs,
        "delta_power": delta_power_epochs,
        "theta_delta_ratio": ratio.astype(np.float32),
        "theta_delta_ratio_smoothed": ratio_smoothed,
        "delta_power_smoothed": delta_power_smoothed,
        "emg_rms_mean": emg_rms_mean_by_epoch,
        "wheel_speed_mean": wheel_speed_mean_by_epoch,
    }


def estimate_thresholds_from_epoch_features(epoch_features: Dict) -> Dict[str, float]:
    """
    Estimate thresholds using the current automated heuristics.
    """
    emg_vals = _as_float_array(epoch_features["emg_rms_mean"])
    delta_power_smoothed = _as_float_array(epoch_features["delta_power_smoothed"])

    n_epochs = emg_vals.size
    if n_epochs < 2:
        emg_threshold = float(
            np.nanmean(emg_vals) + EMG_SD_MULT * np.nanstd(emg_vals)
        )
    else:
        emg_rms_reshaped = emg_vals.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(emg_rms_reshaped)
        emg_component_idx = gmm.predict(emg_rms_reshaped)
        comp_means = [
            np.nanmean(emg_vals[emg_component_idx == i]) for i in range(2)
        ]
        low_emg_comp = int(np.nanargmin(comp_means))
        low_emg_values = emg_vals[emg_component_idx == low_emg_comp]
        emg_threshold = float(
            np.nanmean(low_emg_values) + EMG_SD_MULT * np.nanstd(low_emg_values)
        )

    # Maintain current behaviour: theta/delta threshold hard-coded
    theta_ratio_thr = 2.0
    delta_power_thr = float(np.nanmean(delta_power_smoothed))

    return {
        "emg_rms_threshold": emg_threshold,
        "wheel_speed_threshold": float(WHEEL_SPEED_THRESHOLD),
        "theta_delta_ratio_threshold": theta_ratio_thr,
        "delta_power_threshold": delta_power_thr,
    }


def score_from_epoch_features(
    epoch_features: Dict,
    thresholds: Optional[Dict[str, float]] = None,
    auto_thresholds: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Classify sleep states given epoch-level summary data.

    Parameters
    ----------
    epoch_features : dict
        Output of build_epoch_feature_dict (or equivalent).
    thresholds : dict, optional
        Manual thresholds. Missing entries fall back to auto-estimated values.
    auto_thresholds : bool
        When True, ignore manual thresholds and estimate using the automated heuristics.

    Returns
    -------
    state_epoch : np.ndarray (int8)
        Classified state per epoch (0=active wake, ...).
    resolved_thresholds : dict
        Threshold values actually used.
    """
    auto_vals = estimate_thresholds_from_epoch_features(epoch_features)

    if auto_thresholds or thresholds is None:
        resolved = auto_vals
    else:
        resolved = {}
        for key, auto_val in auto_vals.items():
            if thresholds is not None and key in thresholds and thresholds[key] is not None:
                resolved[key] = float(thresholds[key])
            else:
                resolved[key] = auto_val

    emg = _as_float_array(epoch_features["emg_rms_mean"])
    wheel = _as_float_array(epoch_features["wheel_speed_mean"])
    ratio = _as_float_array(epoch_features["theta_delta_ratio_smoothed"])
    delta_sm = _as_float_array(epoch_features["delta_power_smoothed"])

    n = emg.size
    if wheel.size != n or ratio.size != n or delta_sm.size != n:
        raise ValueError("epoch feature arrays must have matching lengths")

    emg_thr = resolved["emg_rms_threshold"]
    wheel_thr = resolved["wheel_speed_threshold"]
    ratio_thr = resolved["theta_delta_ratio_threshold"]
    delta_thr = resolved["delta_power_threshold"]

    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        moving = (emg[i] >= emg_thr) or (abs(wheel[i]) > wheel_thr)
        if moving:
            out[i] = STATE_ACTIVE_WAKE
        elif ratio[i] > ratio_thr and emg[i] < emg_thr:
            out[i] = STATE_REM
        elif delta_sm[i] >= delta_thr and emg[i] < emg_thr:
            out[i] = STATE_NREM
        else:
            out[i] = STATE_QUIET_WAKE

    return out.astype(np.int8), resolved


# ---------------------------------------------------------------------
# QC plotting (single function, called once at end)
# ---------------------------------------------------------------------

def make_qc_figures(
    t_ephys,
    eeg_filtered,
    emg_filtered,
    emg_rms_full,
    spec_t_abs,
    spec_freqs,
    spec_sxx,
    wheel_speed_full,
    epoch_time,
    sleep_state_epoch,
    emg_rms_mean_by_epoch,
    wheel_speed_mean_by_epoch,
    figs_folder,
):
    # 1) EMG filtered + RMS
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].plot(t_ephys, emg_filtered, linewidth=0.3)
    ax[0].set_title("emg_filtered")
    ax[1].plot(t_ephys, emg_rms_full, linewidth=0.3)
    ax[1].set_title("emg_rms (1 s window)")
    fig.suptitle("sanity check emg & rms", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_folder, "01_sanity_emg_rms.png"))
    plt.close(fig)

    # 2) multisignal: EEG spectrogram, EMG spectrogram, RMS, wheel
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)

    # eeg spectrogram
    pcm = axes[0].pcolormesh(
        spec_t_abs,
        spec_freqs,
        10 * np.log10(spec_sxx + 1e-20),
        shading="auto",
        cmap="jet",
    )
    axes[0].set_ylim(0, 30)
    axes[0].set_title("eeg spectrogram (5 s win, 2 s stride)")
    fig.colorbar(pcm, ax=axes[0], label="power (dB)")

    # emg spectrogram
    emg_nperseg = int(2 * FS_EPHYS)
    emg_noverlap = int(1 * FS_EPHYS)
    spec_emg = axes[1].specgram(
        emg_filtered,
        NFFT=emg_nperseg,
        Fs=FS_EPHYS,
        noverlap=emg_noverlap,
        cmap="jet",
    )
    axes[1].set_ylim(0, 300)
    axes[1].set_title("emg spectrogram")
    spec_emg[3].set_clim(-60, -30)

    axes[2].plot(t_ephys, emg_rms_full, linewidth=0.3)
    axes[2].set_title("emg_rms")

    axes[3].plot(t_ephys, wheel_speed_full, linewidth=0.3)
    axes[3].set_title("wheel_speed_resampled")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_folder, "05_multisignal_sanity.png"))
    plt.close(fig)

    # 3) hypnogram + EMG epoch mean + wheel epoch mean
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epoch_time, sleep_state_epoch, linewidth=0.5)
    axes[0].set_title("hypnogram (raw states)")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels([STATE_LABELS[i] for i in [0, 1, 2, 3]])
    axes[1].plot(epoch_time, emg_rms_mean_by_epoch, linewidth=0.5)
    axes[1].set_title("emg_rms_mean_by_epoch")
    axes[2].plot(epoch_time, wheel_speed_mean_by_epoch, linewidth=0.5)
    axes[2].set_title("wheel_speed_mean_by_epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_folder, "06_hypnogram_emg_wheel.png"))
    plt.close(fig)


# ---------------------------------------------------------------------
# main processing function (callable from GUI)
# ---------------------------------------------------------------------

def run_sleep_scoring(
    user_id,
    exp_id,
    *,
    delta_band=None,
    theta_band=None,
    progress_callback=None,
):
    """
    Sleep scoring pipeline.

    Parameters
    ----------
    user_id : str
        User / subject identifier, passed to organise_paths.find_paths.
    exp_id : str
        Experiment ID string.
    delta_band : tuple(float, float), optional
        Delta band passband (Hz). Defaults to DEFAULT_DELTA_BAND.
    theta_band : tuple(float, float), optional
        Theta band passband (Hz). Defaults to DEFAULT_THETA_BAND.
    progress_callback : callable, optional
        Invoked as progress_callback(fraction, message) as the pipeline advances.

    Returns
    -------
    sleep_state : dict
        Contains:
        - 10 Hz downsampled EEG, EMG, EMG RMS, wheel + time vectors
        - epoch-wise theta and delta power
        - moving-window EEG spectrogram (5 s window, 2 s stride)
        - state classification (per epoch and 10 Hz)
        - thresholds used for classification
        - epoch_features dict for quick rescoring
        - delta/theta band metadata
    """
    start_time = time.time()
    total_steps = 13
    step_idx = 0
    _report_progress(progress_callback, step_idx, total_steps, "Starting sleep scoring")
    print(f"starting sleep scoring for user_id={user_id}, exp_id={exp_id}")

    # -------------------------------------------------------------
    # 1. paths and loading
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Locating paths and loading data")
    print("[1/9] locating paths and loading data")
    animal_id, _, _, exp_dir_processed, _ = organise_paths.find_paths(user_id, exp_id)

    sleep_score_folder = ensure_dir(os.path.join(exp_dir_processed, "sleep_score"))
    figs_folder = ensure_dir(os.path.join(sleep_score_folder, "figs"))

    recordings_dir = os.path.join(exp_dir_processed, "recordings")

    all_ephys = np.load(open(os.path.join(recordings_dir, "ephys.npy"), "rb"))
    wheel_df = pd.read_pickle(open(os.path.join(recordings_dir, "wheel.pickle"), "rb"))

    t_ephys = all_ephys[0, :]
    eeg_raw = all_ephys[1, :]
    emg_raw = all_ephys[2, :]
    fs = FS_EPHYS

    print(f"  - ephys length: {len(t_ephys)} samples")

    if delta_band is None:
        delta_band = DEFAULT_DELTA_BAND
    if theta_band is None:
        theta_band = DEFAULT_THETA_BAND
    delta_band = _validate_band(delta_band, "delta_band", fs)
    theta_band = _validate_band(theta_band, "theta_band", fs)

    # -------------------------------------------------------------
    # 2. filter eeg and emg
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Filtering EEG and EMG")
    print("[2/9] filtering eeg and emg")
    eeg_filtered, emg_filtered = apply_eeg_emg_filters(eeg_raw, emg_raw, fs)

    # -------------------------------------------------------------
    # 3. emg rms (1 s window), aligned to full length
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Computing EMG RMS (1 s window)")
    print("[3/9] computing emg rms (1 s window)")
    rms_window = int(RMS_WINDOW_SEC * fs)
    emg_rms = moving_rms(emg_filtered, rms_window)
    emg_rms_full = emg_rms.copy()  # save full-length RMS for plotting

    # -------------------------------------------------------------
    # 4. eeg band-pass into delta and theta
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Filtering EEG into delta/theta bands")
    print(
        f"[4/9] band-pass filtering eeg into delta ({delta_band[0]:.2f}-{delta_band[1]:.2f} Hz) "
        f"and theta ({theta_band[0]:.2f}-{theta_band[1]:.2f} Hz)"
    )
    delta_b = butter(2, delta_band, btype="bandpass", fs=fs)
    theta_b = butter(2, theta_band, btype="bandpass", fs=fs)
    eeg_delta = filtfilt(delta_b[0], delta_b[1], eeg_filtered)
    eeg_theta = filtfilt(theta_b[0], theta_b[1], eeg_filtered)

    # -------------------------------------------------------------
    # 5. moving-window EEG spectrogram (5 s window, 2 s stride)
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Computing EEG spectrogram")
    print("[5/9] computing eeg spectrogram (5 s window, 2 s stride)")
    spec_nperseg = int(SPEC_WINDOW_SEC * fs)
    spec_step = int(SPEC_STRIDE_SEC * fs)
    spec_noverlap = spec_nperseg - spec_step
    if spec_noverlap >= spec_nperseg:
        raise ValueError("spectrogram noverlap must be < nperseg")

    spec_freqs, spec_t_rel, spec_sxx = signal.spectrogram(
        eeg_filtered,
        fs=fs,
        window="hann",
        nperseg=spec_nperseg,
        noverlap=spec_noverlap,
        scaling="density",
        mode="psd",
    )
    spec_t_abs = t_ephys[0] + spec_t_rel

    # -------------------------------------------------------------
    # 6. resample wheel speed to ephys time base
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Resampling wheel speed")
    print("[6/9] resampling wheel speed to ephys time base")
    wheel_speed_raw = wheel_df["speed"]
    wheel_t = wheel_df["t"]
    wheel_speed_resampled = np.interp(t_ephys, wheel_t, wheel_speed_raw)
    wheel_speed_full = wheel_speed_resampled.copy()  # for plotting

    # -------------------------------------------------------------
    # 7. epoching (10 s epochs) and epoch-level features
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Epoching signals (10 s)")
    print("[7/9] epoching signals (10 s epochs)")

    n_samples = len(eeg_theta)
    n_epochs = n_samples // EPOCH_LEN_SAMPLES
    if n_epochs < 1:
        raise ValueError("recording too short for a single 10 s epoch")

    n_samples_used = n_epochs * EPOCH_LEN_SAMPLES
    print(f"  - number of full epochs: {n_epochs}")

    eeg_theta = eeg_theta[:n_samples_used]
    eeg_delta = eeg_delta[:n_samples_used]
    emg_rms = emg_rms[:n_samples_used]
    wheel_speed_resampled = wheel_speed_resampled[:n_samples_used]

    eeg_theta_epochs = eeg_theta.reshape(n_epochs, EPOCH_LEN_SAMPLES)
    eeg_delta_epochs = eeg_delta.reshape(n_epochs, EPOCH_LEN_SAMPLES)
    emg_rms_epochs = emg_rms.reshape(n_epochs, EPOCH_LEN_SAMPLES)
    wheel_speed_epochs = wheel_speed_resampled.reshape(n_epochs, EPOCH_LEN_SAMPLES)

    start_time_ephys = t_ephys[0]
    epoch_time = start_time_ephys + (np.arange(n_epochs) + 0.5) * EPOCH_LEN_SEC

    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Computing epoch-level power features")
    print("[8/9] computing epoch-wise power features")
    theta_power_epochs = epoch_total_power(eeg_theta_epochs, fs)
    delta_power_epochs = epoch_total_power(eeg_delta_epochs, fs)

    delta_power_smoothed = _adaptive_savgol(delta_power_epochs, SAVGOL_BASE_WIN)
    theta_delta_ratio = theta_power_epochs / delta_power_epochs
    theta_delta_ratio_smoothed = _adaptive_savgol(theta_delta_ratio, SAVGOL_BASE_WIN)

    # -------------------------------------------------------------
    # 8. thresholds and state classification
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Classifying sleep states")
    print("[9/9] estimating thresholds and classifying states")
    emg_rms_mean_by_epoch = emg_rms_epochs.mean(axis=1)
    wheel_speed_mean_by_epoch = wheel_speed_epochs.mean(axis=1)

    epoch_features = build_epoch_feature_dict(
        epoch_time,
        theta_power_epochs,
        delta_power_epochs,
        emg_rms_mean_by_epoch,
        wheel_speed_mean_by_epoch,
        theta_delta_ratio=theta_delta_ratio,
        theta_delta_ratio_smoothed=theta_delta_ratio_smoothed,
        delta_power_smoothed=delta_power_smoothed,
    )

    sleep_state_epoch, thresholds_used = score_from_epoch_features(
        epoch_features,
        auto_thresholds=True,
    )

    emg_threshold = thresholds_used["emg_rms_threshold"]
    wheel_speed_thr = thresholds_used["wheel_speed_threshold"]
    theta_ratio_thr = thresholds_used["theta_delta_ratio_threshold"]
    delta_power_thr = thresholds_used["delta_power_threshold"]

    print(f"  - emg_threshold: {emg_threshold:.4f}")
    print(f"  - theta_ratio_thr: {theta_ratio_thr:.4f}")
    print(f"  - delta_power_thr: {delta_power_thr:.4f}")
    print(f"  - wheel_speed_thr: {wheel_speed_thr:.4f}")

    # -------------------------------------------------------------
    # 9. downsample to 10 Hz and interpolate states
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Downsampling to 10 Hz")
    print("downsampling signals to 10 hz")
    eeg_10hz, t_10hz = downsample_to_target_fs(eeg_filtered, fs, TARGET_FS, t=t_ephys)
    emg_10hz, _ = downsample_to_target_fs(emg_filtered, fs, TARGET_FS, t=t_ephys)
    emg_rms_10hz, _ = downsample_to_target_fs(emg_rms_full, fs, TARGET_FS, t=t_ephys)
    wheel_10hz, _ = downsample_to_target_fs(wheel_speed_full, fs, TARGET_FS, t=t_ephys)

    print("interpolating state to 10 hz")
    state_interp_fun = interp1d(
        epoch_time,
        sleep_state_epoch,
        kind="nearest",
        bounds_error=False,
        fill_value=(sleep_state_epoch[0], sleep_state_epoch[-1]),
    )
    state_10hz = state_interp_fun(t_10hz).astype(np.int8)

    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Computing face motion metric")
    face_motion_10hz = np.full(t_10hz.shape, np.nan, dtype=np.float32)
    try:
        face_motion_t, face_motion_vals = _load_or_compute_face_motion(
            exp_dir_processed,
            exp_id,
            sleep_score_folder,
        )
        if (
            face_motion_vals is not None
            and face_motion_t is not None
            and face_motion_vals.size > 0
            and face_motion_t.size > 0
        ):
            interp_motion = interp1d(
                face_motion_t,
                face_motion_vals,
                kind="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            face_motion_10hz = interp_motion(t_10hz).astype(np.float32)
    except Exception as exc:
        print(f"  - face motion: unable to compute ({exc})")

    # -------------------------------------------------------------
    # 10. build output structure and save
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Building output structures")
    print("building output structures")
    epoch_time_s = epoch_time.astype(float)

    theta_power_epochs = theta_power_epochs.astype(np.float32)
    delta_power_epochs = delta_power_epochs.astype(np.float32)

    sleep_state = {
        # sampling info
        "fs_raw": fs,
        "fs_downsampled": TARGET_FS,

        # 10 hz traces and time vector
        "emg_10hz": emg_10hz.astype(np.float32),
        "emg_10hz_t": t_10hz.astype(np.float64),
        "emg_rms_10hz": emg_rms_10hz.astype(np.float32),
        "emg_rms_10hz_t": t_10hz.astype(np.float64),
        "eeg_10hz": eeg_10hz.astype(np.float32),
        "eeg_10hz_t": t_10hz.astype(np.float64),
        "wheel_10hz": wheel_10hz.astype(np.float32),
        "wheel_10hz_t": t_10hz.astype(np.float64),
        "face_motion_10hz": face_motion_10hz.astype(np.float32),
        "face_motion_10hz_t": t_10hz.astype(np.float64),

        # epoch-level features
        "epoch_t": epoch_time_s,
        "theta_power": theta_power_epochs,
        "delta_power": delta_power_epochs,

        # moving-window spectrogram (5 s window, 2 s stride)
        "eeg_spectrogram": spec_sxx.astype(np.float32),  # (freqs × time_bins)
        "eeg_spectrogram_freqs": spec_freqs.astype(np.float32),
        "eeg_spectrogram_t": spec_t_abs.astype(np.float64),

        # state classification (epoch and 10 hz)
        "state_epoch": sleep_state_epoch.astype(np.int8),
        "state_epoch_t": epoch_time_s,
        "state_10hz": state_10hz.astype(np.int8),
        "state_10hz_t": t_10hz.astype(np.float64),
        "state_labels": STATE_LABELS,

        # thresholds used for classification
        "emg_rms_threshold": float(emg_threshold),
        "wheel_speed_threshold": float(wheel_speed_thr),
        "theta_delta_ratio_threshold": float(theta_ratio_thr),
        "delta_power_threshold": float(delta_power_thr),

        # cached epoch features for downstream rescoring
        "epoch_features": {
            key: np.asarray(val)
            for key, val in epoch_features.items()
        },
        "delta_band": tuple(delta_band),
        "theta_band": tuple(theta_band),
    }

    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Saving outputs")
    print("saving outputs")
    sleep_state_path = os.path.join(sleep_score_folder, "sleep_state.pickle")
    with open(sleep_state_path, "wb") as f:
        pickle.dump(sleep_state, f)

    # -------------------------------------------------------------
    # 11. QC figures (single call)
    # -------------------------------------------------------------
    if MAKE_QC_FIGS:
        print("creating QC figures")
        make_qc_figures(
            t_ephys=t_ephys,
            eeg_filtered=eeg_filtered,
            emg_filtered=emg_filtered,
            emg_rms_full=emg_rms_full,
            spec_t_abs=spec_t_abs,
            spec_freqs=spec_freqs,
            spec_sxx=spec_sxx,
            wheel_speed_full=wheel_speed_full,
            epoch_time=epoch_time,
            sleep_state_epoch=sleep_state_epoch,
            emg_rms_mean_by_epoch=emg_rms_mean_by_epoch,
            wheel_speed_mean_by_epoch=wheel_speed_mean_by_epoch,
            figs_folder=figs_folder,
        )

    duration = time.time() - start_time
    _report_progress(progress_callback, total_steps, total_steps, "Sleep scoring complete")
    print(f"done. scoring complete in {duration:.2f} s")
    print(f"  sleep_state: {sleep_state_path}")

    return sleep_state


if __name__ == "__main__":
    run_sleep_scoring(DEFAULT_USER_ID, DEFAULT_EXP_ID)
