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

# Watson log-spaced spectrogram
LOG_SPEC_WINDOW_SEC = 10.0
LOG_SPEC_STEP_SEC = 1.0
LOG_SPEC_FREQ_BINS = np.logspace(np.log10(1.0), np.log10(100.0), num=60)

# thresholds / smoothing parameters
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

DEFAULT_DELTA_BAND = (1.0, 4.0)
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
    if fs <= 0:
        raise ValueError("sampling frequency must be positive")
    notch_freq = 50.0
    q_factor = 20.0

    def _design_notch(freq):
        nyquist = 0.5 * fs
        if freq <= 0 or freq >= nyquist:
            return None
        return signal.iirnotch(freq, q_factor, fs)

    # base notch
    notch_coeff = _design_notch(notch_freq)
    if notch_coeff is not None:
        b_notch, a_notch = notch_coeff
        eeg_notched = filtfilt(b_notch, a_notch, eeg_raw)
        emg_notched = filtfilt(b_notch, a_notch, emg_raw)
    else:
        eeg_notched = eeg_raw
        emg_notched = emg_raw

    # additional harmonics
    eeg_series = eeg_notched
    emg_series = emg_notched
    for harmonic in range(100, 451, 50):
        coeffs = _design_notch(harmonic)
        if coeffs is None:
            continue
        b_h, a_h = coeffs
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


def _epoch_edges_from_centers(epoch_centers):
    et = np.asarray(epoch_centers, dtype=float)
    if et.size == 0:
        return np.array([], dtype=float)
    if et.size == 1:
        return np.array([et[0] - 5.0, et[0] + 5.0], dtype=float)
    mids = (et[:-1] + et[1:]) / 2.0
    first_edge = et[0] - (mids[0] - et[0])
    last_edge = et[-1] + (et[-1] - mids[-1])
    return np.concatenate(([first_edge], mids, [last_edge]))


def _epoch_mean_from_timeseries(t, x, epoch_centers):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    et = np.asarray(epoch_centers, dtype=float)
    edges = _epoch_edges_from_centers(et)
    means = np.full(et.shape, np.nan, dtype=float)
    for idx in range(et.size):
        start = edges[idx]
        stop = edges[idx + 1]
        mask = (t >= start) & (t < stop) & np.isfinite(x)
        if np.any(mask):
            means[idx] = float(np.mean(x[mask]))
    return means


def _resample_spectrogram(freqs, sxx, target_freqs):
    freq_arr = np.asarray(freqs, dtype=float)
    target = np.asarray(target_freqs, dtype=float)
    if target.size == 0:
        return np.empty((0, sxx.shape[1]), dtype=float)
    mask = (freq_arr >= target[0]) & (freq_arr <= target[-1])
    if not np.any(mask):
        raise ValueError("spectrogram does not cover target frequency range")
    freq_sel = freq_arr[mask]
    spec_sel = np.asarray(sxx[mask], dtype=float)
    interp = interp1d(
        freq_sel,
        spec_sel,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    resampled = interp(target)
    resampled[resampled < 0] = 0.0
    return resampled


def _first_principal_component(matrix):
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D")
    if mat.size == 0:
        return np.zeros(0, dtype=float), np.zeros(mat.shape[0], dtype=float)
    X = mat.T
    if X.size == 0:
        return np.zeros(X.shape[0], dtype=float), np.zeros(mat.shape[0], dtype=float)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        scores = np.zeros(X.shape[0], dtype=float)
        loadings = np.zeros(mat.shape[0], dtype=float)
    else:
        scores = u[:, 0] * s[0]
        loadings = vt[0]
    return scores, loadings


def _band_power_ratio(spectrogram, freqs, low_band, high_band):
    freqs_arr = np.asarray(freqs, dtype=float)
    low_mask = (freqs_arr >= low_band[0]) & (freqs_arr <= low_band[1])
    high_mask = (freqs_arr >= high_band[0]) & (freqs_arr <= high_band[1])
    low_power = np.sum(spectrogram[low_mask], axis=0) if np.any(low_mask) else np.zeros(
        spectrogram.shape[1], dtype=float
    )
    high_power = np.sum(spectrogram[high_mask], axis=0) if np.any(high_mask) else np.zeros(
        spectrogram.shape[1], dtype=float
    )
    ratio = np.zeros_like(low_power, dtype=float)
    valid = high_power > 0
    ratio[valid] = low_power[valid] / high_power[valid]
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def _trough_of_bimodal_distribution(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(np.nan)
    bin_count = max(10, min(80, int(np.sqrt(finite.size) * 2)))
    counts, edges = np.histogram(finite, bins=bin_count)
    if counts.size == 0:
        return float(np.nan)
    peaks, _ = signal.find_peaks(counts)
    if peaks.size < 2:
        return float(np.nanmedian(finite))
    peak_heights = counts[peaks]
    top_idxs = np.argsort(peak_heights)[-2:]
    left_peak, right_peak = np.sort(peaks[top_idxs])
    trough_segment = counts[left_peak : right_peak + 1]
    if trough_segment.size == 0:
        return float(np.nanmedian(finite))
    trough_idx = left_peak + int(np.nanargmin(trough_segment))
    midpoint = 0.5 * (edges[trough_idx] + edges[trough_idx + 1])
    return float(midpoint)


def downsample_to_target_fs(x, fs_original, fs_target, t=None):
    """
    downsample by integer factor (fs_original / fs_target).
    optionally downsample matching time vector t (same length as x).
    """
    fs_orig_val = float(fs_original)
    if fs_orig_val % fs_target != 0:
        raise ValueError("fs_original must be an integer multiple of fs_target")

    factor = int(fs_orig_val // fs_target)
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
    low_freq_power=None,
    low_freq_power_smoothed=None,
    theta_ratio=None,
    theta_ratio_smoothed=None,
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

    if theta_ratio is None:
        theta_ratio_vals = ratio.astype(np.float32)
    else:
        theta_ratio_vals = _as_float_array(theta_ratio).astype(np.float32)

    if theta_ratio_smoothed is None:
        theta_ratio_smooth = _adaptive_savgol(theta_ratio_vals, SAVGOL_BASE_WIN).astype(np.float32)
    else:
        theta_ratio_smooth = _as_float_array(theta_ratio_smoothed).astype(np.float32)

    if low_freq_power is None:
        low_freq_mean = np.full(theta_power_epochs.shape, np.nan, dtype=np.float32)
    else:
        low_freq_mean = _as_float_array(low_freq_power).astype(np.float32)

    if low_freq_power_smoothed is None:
        low_freq_smooth = _adaptive_savgol(low_freq_mean, SAVGOL_BASE_WIN).astype(np.float32)
    else:
        low_freq_smooth = _as_float_array(low_freq_power_smoothed).astype(np.float32)

    return {
        "epoch_time": epoch_time,
        "theta_power": theta_power_epochs,
        "delta_power": delta_power_epochs,
        "theta_delta_ratio": ratio.astype(np.float32),
        "theta_delta_ratio_smoothed": ratio_smoothed,
        "theta_ratio": theta_ratio_vals,
        "theta_ratio_smoothed": theta_ratio_smooth,
        "delta_power_smoothed": delta_power_smoothed,
        "emg_rms_mean": emg_rms_mean_by_epoch,
        "wheel_speed_mean": wheel_speed_mean_by_epoch,
        "low_freq_power": low_freq_mean,
        "low_freq_power_smoothed": low_freq_smooth,
    }


def estimate_thresholds_from_epoch_features(epoch_features: Dict) -> Dict[str, float]:
    """
    Estimate thresholds using trough detection on the available epoch metrics.
    """

    default_locomotion_thr = 0.1

    def _resolve(arr):
        arr_vals = _as_float_array(arr)
        thr = _trough_of_bimodal_distribution(arr_vals)
        if np.isfinite(thr):
            return thr
        if arr_vals.size == 0:
            return float(np.nan)
        return float(np.nanmedian(arr_vals))

    low_freq = epoch_features.get("low_freq_power_smoothed", epoch_features.get("low_freq_power", []))
    theta_ratio = epoch_features.get("theta_delta_ratio_smoothed", epoch_features.get("theta_ratio_smoothed", []))
    emg_vals = epoch_features.get("emg_rms_mean", [])
    return {
        "low_freq_threshold": _resolve(low_freq),
        "theta_ratio_threshold": _resolve(theta_ratio),
        "emg_rms_threshold": _resolve(emg_vals),
        "locomotion_threshold": default_locomotion_thr,
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
    def _normalize_thresholds(thr_dict):
        if not thr_dict:
            return {}
        mapping = {
            "low_freq_threshold": "low_freq_threshold",
            "delta_power_threshold": "low_freq_threshold",
            "wheel_speed_threshold": "low_freq_threshold",
            "theta_ratio_threshold": "theta_ratio_threshold",
            "theta_delta_ratio_threshold": "theta_ratio_threshold",
            "emg_rms_threshold": "emg_rms_threshold",
            "locomotion_threshold": "locomotion_threshold",
        }
        normalized = {}
        for key, value in thr_dict.items():
            if value is None:
                continue
            mapped = mapping.get(key, key)
            normalized[mapped] = float(value)
        return normalized

    normalized_thresholds = _normalize_thresholds(thresholds)
    auto_vals = estimate_thresholds_from_epoch_features(epoch_features)

    if auto_thresholds or thresholds is None:
        resolved = auto_vals
    else:
        resolved = {}
        for key, auto_val in auto_vals.items():
            if key in normalized_thresholds:
                resolved[key] = normalized_thresholds[key]
            else:
                resolved[key] = auto_val

    theta_thr = resolved.get("theta_ratio_threshold", float(np.nan))
    if not np.isfinite(theta_thr):
        resolved["theta_ratio_threshold"] = 2.0

    emg = _as_float_array(epoch_features["emg_rms_mean"])
    ratio = _as_float_array(epoch_features["theta_delta_ratio_smoothed"])
    locomotion = _as_float_array(
        epoch_features.get("wheel_speed_mean", np.zeros_like(emg))
    )

    n = emg.size
    low_freq = _as_float_array(epoch_features["low_freq_power_smoothed"])
    if locomotion.size != n:
        locomotion = np.zeros(n, dtype=float)
    if ratio.size != n or low_freq.size != n:
        raise ValueError("epoch feature arrays must have matching lengths")

    emg_thr = resolved.get("emg_rms_threshold", float(np.nan))
    ratio_thr = resolved.get("theta_ratio_threshold", float(np.nan))
    low_freq_thr = resolved.get("low_freq_threshold", float(np.nan))
    locomotion_thr = resolved.get("locomotion_threshold", 0.1)
    if not np.isfinite(locomotion_thr):
        locomotion_thr = 0.1

    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        locomotion_high = (
            np.isfinite(locomotion[i])
            and np.isfinite(locomotion_thr)
            and locomotion[i] > locomotion_thr
        )
        low_freq_high = (
            np.isfinite(low_freq[i])
            and np.isfinite(low_freq_thr)
            and low_freq[i] >= low_freq_thr
        )
        ratio_high = (
            np.isfinite(ratio[i])
            and np.isfinite(ratio_thr)
            and ratio[i] >= ratio_thr
        )
        emg_low = (
            np.isfinite(emg[i])
            and np.isfinite(emg_thr)
            and emg[i] < emg_thr
        )
        emg_high = (
            np.isfinite(emg[i])
            and np.isfinite(emg_thr)
            and emg[i] >= emg_thr
        )
        if locomotion_high:
            out[i] = STATE_ACTIVE_WAKE
        elif low_freq_high:
            out[i] = STATE_NREM
        elif ratio_high and emg_low:
            out[i] = STATE_REM
        elif emg_high:
            out[i] = STATE_ACTIVE_WAKE
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
    filename_suffix="",
):
    # 1) EMG filtered + RMS
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].plot(t_ephys, emg_filtered, linewidth=0.3)
    ax[0].set_title("emg_filtered")
    ax[1].plot(t_ephys, emg_rms_full, linewidth=0.3)
    ax[1].set_title("emg_rms (1 s window)")
    fig.suptitle("sanity check emg & rms", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_folder, f"01_sanity_emg_rms{filename_suffix}.png"))
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
    fig.savefig(os.path.join(figs_folder, f"05_multisignal_sanity{filename_suffix}.png"))
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
    fig.savefig(os.path.join(figs_folder, f"06_hypnogram_emg_wheel{filename_suffix}.png"))
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
    simulated_npz=None,
    filename_suffix="",
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
    simulated_npz : str, optional
        Path to an `.npz` file that contains `eeg` and `emg` arrays for simulated scoring.
    filename_suffix : str, optional
        String appended before the extension on any output files (e.g., `_sim`).

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
    total_steps = 14
    step_idx = 0
    _report_progress(progress_callback, step_idx, total_steps, "Starting sleep scoring")
    print(f"starting sleep scoring for user_id={user_id}, exp_id={exp_id}")
    suffix = filename_suffix or ""
    simulated = simulated_npz is not None

    # -------------------------------------------------------------
    # 1. paths and loading
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Locating paths and loading data")
    print("[1/9] locating paths and loading data")
    animal_id, _, _, exp_dir_processed, _ = organise_paths.find_paths(user_id, exp_id)

    sleep_score_folder = ensure_dir(os.path.join(exp_dir_processed, "sleep_score"))
    figs_folder = ensure_dir(os.path.join(sleep_score_folder, "figs"))

    if simulated:
        with np.load(simulated_npz) as sim_data:
            eeg_raw = np.asarray(sim_data["eeg"], dtype=float)
            emg_raw = np.asarray(sim_data["emg"], dtype=float)
            fs_val = sim_data.get("fs", sim_data.get("fs_original", FS_EPHYS))
            fs = float(fs_val)
            if eeg_raw.shape != emg_raw.shape:
                raise ValueError("simulated eeg and emg arrays must be the same length")
            n_samples = eeg_raw.size
            t_ephys = np.arange(n_samples, dtype=float) / fs
            print(f"  - simulated ephys length: {n_samples} samples")
    else:
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
    print("[3/14] computing emg rms (1 s window)")
    rms_window = int(RMS_WINDOW_SEC * fs)
    emg_rms = moving_rms(emg_filtered, rms_window)
    emg_rms_full = emg_rms.copy()  # save full-length RMS for plotting

    # -------------------------------------------------------------
    # 4. log-spaced spectrogram (10 s window, 1 s stride)
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Computing log-frequency spectrogram")
    print("[4/14] computing log-frequency spectrogram (10 s window, 1 s stride)")
    log_nperseg = int(LOG_SPEC_WINDOW_SEC * fs)
    log_step = int(LOG_SPEC_STEP_SEC * fs)
    if log_step <= 0:
        raise ValueError("log spectrogram step must be positive")
    log_noverlap = log_nperseg - log_step
    if log_noverlap >= log_nperseg:
        raise ValueError("log-frequency spectrogram noverlap must be < nperseg")

    log_freqs, log_rel_t, log_sxx = signal.spectrogram(
        eeg_filtered,
        fs=fs,
        window="hann",
        nperseg=log_nperseg,
        noverlap=log_noverlap,
        scaling="density",
        mode="psd",
    )
    log_t_abs = t_ephys[0] + log_rel_t
    log_sxx_resampled = _resample_spectrogram(log_freqs, log_sxx, LOG_SPEC_FREQ_BINS)
    log_sxx_db = np.log10(log_sxx_resampled + 1e-12)
    freq_mean = np.nanmean(log_sxx_db, axis=1, keepdims=True)
    freq_std = np.nanstd(log_sxx_db, axis=1, keepdims=True)
    freq_std[~np.isfinite(freq_std) | (freq_std <= 0)] = 1.0
    log_sxx_z = (log_sxx_db - freq_mean) / freq_std
    low_freq_series, low_freq_weights = _first_principal_component(log_sxx_z)
    low_freq_mask = LOG_SPEC_FREQ_BINS <= 20.0
    if np.any(low_freq_mask) and np.isfinite(np.nansum(low_freq_weights[low_freq_mask])):
        if np.nansum(low_freq_weights[low_freq_mask]) < 0:
            low_freq_series = -low_freq_series
    theta_ratio_series = _band_power_ratio(
        log_sxx_resampled,
        LOG_SPEC_FREQ_BINS,
        theta_band,
        delta_band,
    )

    # -------------------------------------------------------------
    # 5. eeg band-pass into delta and theta
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Filtering EEG into delta/theta bands")
    print(
        f"[5/14] band-pass filtering eeg into delta ({delta_band[0]:.2f}-{delta_band[1]:.2f} Hz) "
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
    print("[6/14] computing eeg spectrogram (5 s window, 2 s stride)")
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
    wheel_step_msg = "Using simulated wheel speed (zeros)" if simulated else "Resampling wheel speed"
    _report_progress(progress_callback, step_idx, total_steps, wheel_step_msg)
    if simulated:
        print("[7/14] using simulated wheel speed (zeros)")
        wheel_speed_full = np.zeros_like(t_ephys, dtype=float)
        wheel_speed_resampled = wheel_speed_full.copy()
    else:
        print("[7/14] resampling wheel speed to ephys time base")
        wheel_speed_raw = wheel_df["speed"]
        wheel_t = wheel_df["t"]
        wheel_speed_resampled = np.interp(t_ephys, wheel_t, wheel_speed_raw)
        wheel_speed_full = wheel_speed_resampled.copy()  # for plotting

    # -------------------------------------------------------------
    # 7. epoching (10 s epochs) and epoch-level features
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Epoching signals (10 s)")
    print("[8/14] epoching signals (10 s epochs)")

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
    print("[9/14] computing epoch-wise power features")
    theta_power_epochs = epoch_total_power(eeg_theta_epochs, fs)
    delta_power_epochs = epoch_total_power(eeg_delta_epochs, fs)

    delta_power_smoothed = _adaptive_savgol(delta_power_epochs, SAVGOL_BASE_WIN)
    theta_delta_ratio = theta_power_epochs / delta_power_epochs
    theta_delta_ratio_smoothed = _adaptive_savgol(theta_delta_ratio, SAVGOL_BASE_WIN)

    low_freq_by_epoch = _epoch_mean_from_timeseries(log_t_abs, low_freq_series, epoch_time)
    theta_ratio_by_epoch = _epoch_mean_from_timeseries(log_t_abs, theta_ratio_series, epoch_time)

    # -------------------------------------------------------------
    # 8. thresholds and state classification
    # -------------------------------------------------------------
    step_idx += 1
    _report_progress(progress_callback, step_idx, total_steps, "Classifying sleep states")
    print("[10/14] estimating thresholds and classifying states")
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
        low_freq_power=low_freq_by_epoch,
        theta_ratio=theta_ratio_by_epoch,
    )

    sleep_state_epoch, thresholds_used = score_from_epoch_features(
        epoch_features,
        auto_thresholds=True,
    )

    emg_threshold = thresholds_used["emg_rms_threshold"]
    theta_ratio_thr = thresholds_used["theta_ratio_threshold"]
    low_freq_thr = thresholds_used["low_freq_threshold"]
    locomotion_thr = thresholds_used.get("locomotion_threshold", 0.1)

    print(f"  - emg_threshold: {emg_threshold:.4f}")
    print(f"  - theta_ratio_thr: {theta_ratio_thr:.4f}")
    print(f"  - low_freq_thr: {low_freq_thr:.4f}")
    print(f"  - locomotion_thr: {locomotion_thr:.4f}")

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
        "theta_ratio_threshold": float(theta_ratio_thr),
        "theta_delta_ratio_threshold": float(theta_ratio_thr),
        "low_freq_threshold": float(low_freq_thr),
        "delta_power_threshold": float(low_freq_thr),
        "wheel_speed_threshold": float(np.nan),
        "locomotion_threshold": float(locomotion_thr),

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
    sleep_state_path = os.path.join(sleep_score_folder, f"sleep_state{suffix}.pickle")
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
            filename_suffix=suffix,
        )

    duration = time.time() - start_time
    _report_progress(progress_callback, total_steps, total_steps, "Sleep scoring complete")
    print(f"done. scoring complete in {duration:.2f} s")
    print(f"  sleep_state: {sleep_state_path}")

    return sleep_state


if __name__ == "__main__":
    run_sleep_scoring(DEFAULT_USER_ID, DEFAULT_EXP_ID)
