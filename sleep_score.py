import os
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

import organise_paths


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
    plt.show()
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

def run_sleep_scoring(user_id, exp_id):
    """
    Sleep scoring pipeline.

    Parameters
    ----------
    user_id : str
        User / subject identifier, passed to organise_paths.find_paths.
    exp_id : str
        Experiment ID string.

    Returns
    -------
    sleep_state : dict
        Contains:
        - 10 Hz downsampled EEG, EMG, EMG RMS, wheel + time vectors
        - epoch-wise theta and delta power
        - moving-window EEG spectrogram (5 s window, 2 s stride)
        - state classification (per epoch and 10 Hz)
        - thresholds used for classification
    """
    start_time = time.time()
    print(f"starting sleep scoring for user_id={user_id}, exp_id={exp_id}")

    # -------------------------------------------------------------
    # 1. paths and loading
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 2. filter eeg and emg
    # -------------------------------------------------------------
    print("[2/9] filtering eeg and emg")
    eeg_filtered, emg_filtered = apply_eeg_emg_filters(eeg_raw, emg_raw, fs)

    # -------------------------------------------------------------
    # 3. emg rms (1 s window), aligned to full length
    # -------------------------------------------------------------
    print("[3/9] computing emg rms (1 s window)")
    rms_window = int(RMS_WINDOW_SEC * fs)
    emg_rms = moving_rms(emg_filtered, rms_window)
    emg_rms_full = emg_rms.copy()  # save full-length RMS for plotting

    # -------------------------------------------------------------
    # 4. eeg band-pass into delta and theta
    # -------------------------------------------------------------
    print("[4/9] band-pass filtering eeg into delta and theta")
    delta_b = butter(2, [1, 10], btype="bandpass", fs=fs)
    theta_b = butter(2, [5, 10], btype="bandpass", fs=fs)
    eeg_delta = filtfilt(delta_b[0], delta_b[1], eeg_filtered)
    eeg_theta = filtfilt(theta_b[0], theta_b[1], eeg_filtered)

    # -------------------------------------------------------------
    # 5. moving-window EEG spectrogram (5 s window, 2 s stride)
    # -------------------------------------------------------------
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
    print("[6/9] resampling wheel speed to ephys time base")
    wheel_speed_raw = wheel_df["speed"]
    wheel_t = wheel_df["t"]
    wheel_speed_resampled = np.interp(t_ephys, wheel_t, wheel_speed_raw)
    wheel_speed_full = wheel_speed_resampled.copy()  # for plotting

    # -------------------------------------------------------------
    # 7. epoching (10 s epochs) and epoch-level features
    # -------------------------------------------------------------
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

    print("[8/9] computing epoch-wise power features")
    theta_power_epochs = epoch_total_power(eeg_theta_epochs, fs)
    delta_power_epochs = epoch_total_power(eeg_delta_epochs, fs)

    delta_power_smoothed = _adaptive_savgol(delta_power_epochs, SAVGOL_BASE_WIN)
    theta_delta_ratio = theta_power_epochs / delta_power_epochs
    theta_delta_ratio_smoothed = _adaptive_savgol(theta_delta_ratio, SAVGOL_BASE_WIN)

    # -------------------------------------------------------------
    # 8. thresholds and state classification
    # -------------------------------------------------------------
    print("[9/9] estimating emg threshold via gmm and classifying states")
    emg_rms_mean_by_epoch = emg_rms_epochs.mean(axis=1)
    wheel_speed_mean_by_epoch = wheel_speed_epochs.mean(axis=1)

    # GMM for EMG
    if n_epochs < 2:
        emg_threshold = emg_rms_mean_by_epoch.mean() + \
                        EMG_SD_MULT * emg_rms_mean_by_epoch.std()
    else:
        emg_rms_reshaped = emg_rms_mean_by_epoch.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(emg_rms_reshaped)
        emg_component_idx = gmm.predict(emg_rms_reshaped)

        comp_means = [emg_rms_mean_by_epoch[emg_component_idx == i].mean()
                      for i in range(2)]
        low_emg_comp = int(np.argmin(comp_means))
        low_emg_values = emg_rms_mean_by_epoch[emg_component_idx == low_emg_comp]
        emg_threshold = low_emg_values.mean() + EMG_SD_MULT * low_emg_values.std()

    print(f"  - emg_threshold: {emg_threshold:.4f}")

    theta_ratio_thr = 2 #theta_delta_ratio_smoothed.mean() + \
                        #THETA_RATIO_SD_MULT * theta_delta_ratio_smoothed.std()
    delta_power_thr = float(delta_power_smoothed.mean())

    print(f"  - theta_ratio_thr: {theta_ratio_thr:.4f}")
    print(f"  - delta_power_thr: {delta_power_thr:.4f}")
    print(f"  - wheel_speed_thr: {WHEEL_SPEED_THRESHOLD:.4f}")

    # classify states
    sleep_state_epoch = np.empty(n_epochs, dtype=np.int8)
    step = max(1, n_epochs // 20)

    for i in range(n_epochs):
        emg_val = emg_rms_mean_by_epoch[i]
        theta_ratio_val = theta_delta_ratio_smoothed[i]
        delta_power_val = delta_power_smoothed[i]
        wheel_val = wheel_speed_mean_by_epoch[i]

        moving = (emg_val >= emg_threshold) or (abs(wheel_val) > WHEEL_SPEED_THRESHOLD)

        if moving:
            state = STATE_ACTIVE_WAKE
        elif theta_ratio_val > theta_ratio_thr and emg_val < emg_threshold:
            state = STATE_REM
        elif delta_power_val >= delta_power_thr and emg_val < emg_threshold:
            state = STATE_NREM
        else:
            state = STATE_QUIET_WAKE

        sleep_state_epoch[i] = state

        if (i + 1) % step == 0 or (i + 1) == n_epochs:
            progress = (i + 1) / n_epochs * 100
            print(f"\r    progress (scoring): {progress:5.1f}%", end="")
    print()

    # -------------------------------------------------------------
    # 9. downsample to 10 Hz and interpolate states
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 10. build output structure and save
    # -------------------------------------------------------------
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
        "wheel_speed_threshold": float(WHEEL_SPEED_THRESHOLD),
        "theta_delta_ratio_threshold": float(theta_ratio_thr),
        "delta_power_threshold": float(delta_power_thr),
    }

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
    print(f"done. scoring complete in {duration:.2f} s")
    print(f"  sleep_state: {sleep_state_path}")

    return sleep_state


if __name__ == "__main__":
    run_sleep_scoring(DEFAULT_USER_ID, DEFAULT_EXP_ID)
