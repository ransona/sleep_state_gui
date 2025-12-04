import os
import pickle
import time

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

default_animal_id = 'pmateosaparicio'
default_exp_id = '2025-06-12_04_ESPM135'

fs_ephys = 1000                  # hz, original sampling rate
epoch_len_samples = 10000        # 10 s epochs at 1 khz
target_fs = 10                   # hz, target rate for downsampled traces
make_qc_figs = True              # set false to skip plotting

# state code mapping
state_active_wake = 0
state_quiet_wake = 1
state_nrem = 2
state_rem = 3

state_labels = {
    state_active_wake: "active wake",
    state_quiet_wake: "quiet wake",
    state_nrem: "nrem",
    state_rem: "rem",
}


# ---------------------------------------------------------------------
# utility functions
# ---------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def moving_rms(x, window_size):
    """
    compute rms over a sliding window using convolution
    """
    squared = np.square(x)
    window = np.ones(window_size, dtype=float) / window_size
    rms = np.sqrt(np.convolve(squared, window, mode='valid'))
    return rms


def apply_eeg_emg_filters(eeg_raw, emg_raw, fs):
    """
    apply notch filters (50 hz and harmonics) and band-pass filters:
      - eeg: 1–30 hz
      - emg: 100–499 hz
    returns filtered eeg and emg.
    """
    print("  - applying notch and band-pass filters to eeg/emg")
    notch_freq = 50.0
    q_factor = 20.0
    b_notch, a_notch = signal.iirnotch(notch_freq, q_factor, fs)
    eeg_notched = filtfilt(b_notch, a_notch, eeg_raw)
    emg_notched = filtfilt(b_notch, a_notch, emg_raw)

    # additional harmonics (100, 150, ..., 450 hz)
    eeg_series = eeg_notched.copy()
    emg_series = emg_notched.copy()
    for harmonic in range(99, 499, 50):
        b_harm, a_harm = signal.iirnotch(harmonic, q_factor, fs)
        eeg_series = filtfilt(b_harm, a_harm, eeg_series)
        emg_series = filtfilt(b_harm, a_harm, emg_series)

    # band-pass filters
    eeg_b = butter(2, [1, 30], btype='bandpass', fs=fs)
    emg_b = butter(2, [100, 499], btype='bandpass', fs=fs)

    eeg_filtered = filtfilt(eeg_b[0], eeg_b[1], eeg_series)
    emg_filtered = filtfilt(emg_b[0], emg_b[1], emg_series)

    return eeg_filtered, emg_filtered


def compute_psd(signal_data, fs, window_samples):
    """
    welch power spectral density of an entire time series.
    """
    freqs, psd = signal.welch(signal_data, fs, nperseg=window_samples)
    return freqs, psd


def calculate_epoch_power(epoch_list, fs, label=""):
    """
    for each epoch (array) in epoch_list, compute total power using periodogram.
    returns list of summed power values.

    includes simple progress feedback.
    """
    summed_power = []
    n_epochs = len(epoch_list)
    print(f"  - computing {label} power per epoch ({n_epochs} epochs)")
    if n_epochs == 0:
        return summed_power

    step = max(1, n_epochs // 20)  # ~5% steps
    for i, epoch in enumerate(epoch_list):
        _, power = signal.periodogram(epoch, fs=fs)
        summed_power.append(np.sum(np.abs(power)))
        if (i + 1) % step == 0 or (i + 1) == n_epochs:
            progress = (i + 1) / n_epochs * 100
            print(f"\r    progress {label}: {progress:5.1f}%", end="")
    print()
    return summed_power


def downsample_to_target_fs(x, fs_original, fs_target, t=None):
    """
    downsample by integer factor (fs_original / fs_target).
    optionally downsample matching time vector t (same length as x).
    """
    factor = fs_original // fs_target
    if fs_original % fs_target != 0:
        raise ValueError("fs_original must be an integer multiple of fs_target.")
    x_ds = x[::factor]
    if t is not None:
        t_ds = t[::factor]
        return x_ds, t_ds
    return x_ds


# ---------------------------------------------------------------------
# main processing function (callable from GUI)
# ---------------------------------------------------------------------

def run_sleep_scoring(animal_id, exp_id):
    """
    Main entry point for sleep scoring.

    animal_id: string passed to organise_paths.find_paths (user / animal identifier)
    exp_id:    experiment ID string

    Returns:
        sleep_state (dict) with 10 hz traces, epoch-level features,
        moving-window spectrogram, and state classifications.
    """
    start_time = time.time()
    print(f"starting sleep scoring for {animal_id}, {exp_id}")

    # -----------------------------------------------------------------
    # paths and loading
    # -----------------------------------------------------------------
    print("[1/9] locating paths and loading data")
    returned_animal_id, remote_root, processed_root, exp_dir_processed, exp_dir_raw = \
        organise_paths.find_paths(animal_id, exp_id)

    # save under experiment processed dir
    sleep_score_folder = ensure_dir(os.path.join(exp_dir_processed, 'sleep_score'))
    figs_folder = ensure_dir(os.path.join(sleep_score_folder, 'figs'))

    recordings_dir = os.path.join(exp_dir_processed, 'recordings')

    all_ephys = np.load(open(os.path.join(recordings_dir, 'ephys.npy'), "rb"))
    wheel_df = pd.read_pickle(open(os.path.join(recordings_dir, 'wheel.pickle'), "rb"))

    t_ephys = all_ephys[0, :]
    eeg_raw = all_ephys[1, :]
    emg_raw = all_ephys[2, :]
    fs = fs_ephys

    print(f"  - ephys length: {len(t_ephys)} samples")

    # -----------------------------------------------------------------
    # filter eeg and emg
    # -----------------------------------------------------------------
    print("[2/9] filtering eeg and emg")
    eeg_filtered, emg_filtered = apply_eeg_emg_filters(eeg_raw, emg_raw, fs)

    # -----------------------------------------------------------------
    # emg rms (1 s window) aligned to full rate
    # -----------------------------------------------------------------
    print("[3/9] computing emg rms (1 s window)")
    rms_window = fs  # 1 second
    emg_rms_valid = moving_rms(emg_filtered, rms_window)

    t_rms_valid = np.linspace(t_ephys[0], t_ephys[len(emg_rms_valid)], len(emg_rms_valid))
    emg_rms = np.interp(t_ephys, t_rms_valid, emg_rms_valid)

    if make_qc_figs:
        fig, ax = plt.subplots(2, 1, figsize=(16, 8))
        ax[0].plot(t_ephys, emg_filtered, linewidth=0.3)
        ax[0].set_title('emg_filtered')
        ax[1].plot(t_ephys, emg_rms, linewidth=0.3)
        ax[1].set_title('emg_rms (1 s window)')
        fig.suptitle('sanity check emg & rms', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(figs_folder, '01_sanity_emg_rms.png'))
        plt.close(fig)

    # -----------------------------------------------------------------
    # eeg band-pass into delta and theta
    # -----------------------------------------------------------------
    print("[4/9] band-pass filtering eeg into delta and theta")
    delta_b = butter(2, [1, 4], btype='bandpass', fs=fs)
    theta_b = butter(2, [5, 10], btype='bandpass', fs=fs)
    eeg_delta = filtfilt(delta_b[0], delta_b[1], eeg_filtered)
    eeg_theta = filtfilt(theta_b[0], theta_b[1], eeg_filtered)

    welch_win = 4 * fs  # 4 s window

    print("  - computing welch psd for whole recording")
    freqs_eeg, psd_eeg = compute_psd(eeg_filtered, fs, welch_win)
    freqs_delta, psd_delta = compute_psd(eeg_delta, fs, welch_win)
    freqs_theta, psd_theta = compute_psd(eeg_theta, fs, welch_win)

    if make_qc_figs:
        for freqs, psd, title, fname in [
            (freqs_eeg, psd_eeg, "eeg (1–30 hz) psd", "02_eeg_welch_psd.png"),
            (freqs_delta, psd_delta, "delta band (1–4 hz) psd", "03_delta_psd.png"),
            (freqs_theta, psd_theta, "theta band (5–10 hz) psd", "04_theta_psd.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(freqs, psd, linewidth=1)
            ax.set_xlim(0, 30)
            ax.set_xlabel('frequency (hz)')
            ax.set_ylabel('psd (v^2/hz)')
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(os.path.join(figs_folder, fname))
            plt.close(fig)

    # -----------------------------------------------------------------
    # moving-window EEG spectrogram (5 s window, 2 s stride)
    # -----------------------------------------------------------------
    print("[5/9] computing eeg spectrogram (5 s window, 2 s stride)")
    win_len_samples = int(5 * fs)
    step_len_samples = int(2 * fs)
    noverlap = win_len_samples - step_len_samples  # 3 s overlap = 3000 at 1 khz

    spec_freqs, spec_t_rel, spec_sxx = signal.spectrogram(
        eeg_filtered,
        fs=fs,
        window='hann',
        nperseg=win_len_samples,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )
    # absolute times for spectrogram time bins
    spec_t_abs = t_ephys[0] + spec_t_rel

    # -----------------------------------------------------------------
    # resample wheel speed to ephys time base (no multiplication)
    # -----------------------------------------------------------------
    print("[6/9] resampling wheel speed to ephys time base")
    wheel_speed_raw = wheel_df['speed']
    wheel_t = wheel_df['t']
    wheel_speed_resampled = np.interp(t_ephys, wheel_t, wheel_speed_raw)

    if make_qc_figs:
        print("  - generating multisignal sanity plots")
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)

        # eeg spectrogram (from spec_sxx)
        pcm = axes[0].pcolormesh(
            spec_t_abs,
            spec_freqs,
            10 * np.log10(spec_sxx + 1e-20),
            shading='auto',
            cmap='jet'
        )
        axes[0].set_ylim(0, 30)
        axes[0].set_title('eeg spectrogram (5 s win, 2 s stride)')
        fig.colorbar(pcm, ax=axes[0], label='power (dB)')

        # emg spectrogram (coarser, just for QC)
        spec_emg = axes[1].specgram(
            emg_filtered,
            NFFT=epoch_len_samples,
            Fs=fs,
            noverlap=epoch_len_samples // 2,
            cmap='jet'
        )
        axes[1].set_ylim(0, 300)
        axes[1].set_title('emg spectrogram')
        spec_emg[3].set_clim(-60, -30)

        # emg rms
        axes[2].plot(t_ephys, emg_rms, linewidth=0.3)
        axes[2].set_title('emg_rms')

        # wheel speed
        axes[3].plot(t_ephys, wheel_speed_resampled, linewidth=0.3)
        axes[3].set_title('wheel_speed_resampled')

        fig.tight_layout()
        fig.savefig(os.path.join(figs_folder, '05_multisignal_sanity.png'))
        plt.close(fig)

    # -----------------------------------------------------------------
    # epoching (10 s epochs) for scoring features
    # -----------------------------------------------------------------
    print("[7/9] epoching signals (10 s epochs)")
    epoch_duration_s = epoch_len_samples / fs
    n_epochs = len(eeg_theta) // epoch_len_samples
    n_samples_used = n_epochs * epoch_len_samples
    print(f"  - number of full epochs: {n_epochs}")

    eeg_theta_trimmed = eeg_theta[:n_samples_used]
    eeg_delta_trimmed = eeg_delta[:n_samples_used]
    emg_rms_trimmed = emg_rms[:n_samples_used]
    wheel_speed_trimmed = wheel_speed_resampled[:n_samples_used]

    eeg_theta_epochs = np.array_split(eeg_theta_trimmed, n_epochs)
    eeg_delta_epochs = np.array_split(eeg_delta_trimmed, n_epochs)
    emg_rms_epochs = np.array_split(emg_rms_trimmed, n_epochs)
    wheel_speed_epochs = np.array_split(wheel_speed_trimmed, n_epochs)

    # epoch time: middle of each epoch
    start_time_ephys = t_ephys[0]
    epoch_time = start_time_ephys + (np.arange(n_epochs) + 0.5) * epoch_duration_s

    # -----------------------------------------------------------------
    # epoch-wise power features (theta, delta)
    # -----------------------------------------------------------------
    print("[8/9] computing epoch-wise power features")
    theta_power_epochs = calculate_epoch_power(eeg_theta_epochs, fs, label="theta")
    delta_power_epochs = calculate_epoch_power(eeg_delta_epochs, fs, label="delta")

    print("  - smoothing delta power across epochs")
    delta_power_smoothed = signal.savgol_filter(
        delta_power_epochs, window_length=10, polyorder=3, mode="nearest"
    )

    print("  - computing theta/delta ratio and smoothing")
    theta_delta_ratio = np.array(theta_power_epochs) / np.array(delta_power_epochs)
    theta_delta_ratio_smoothed = signal.savgol_filter(
        theta_delta_ratio, window_length=10, polyorder=3, mode="nearest"
    )

    # -----------------------------------------------------------------
    # emg gmm (for thresholding movement / emg amplitude)
    # -----------------------------------------------------------------
    print("[9/9] estimating emg threshold via gmm and classifying states")
    emg_rms_mean_by_epoch = np.array([np.mean(ep) for ep in emg_rms_epochs])
    emg_rms_reshaped = emg_rms_mean_by_epoch.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2)
    gmm.fit(emg_rms_reshaped)
    emg_component_idx = gmm.predict(emg_rms_reshaped)

    comp_means = [emg_rms_mean_by_epoch[emg_component_idx == i].mean() for i in range(2)]
    low_emg_comp = int(np.argmin(comp_means))

    low_emg_values = emg_rms_mean_by_epoch[emg_component_idx == low_emg_comp]
    emg_threshold = low_emg_values.mean() + 3.0 * low_emg_values.std()
    print(f"  - emg_threshold (low mean + 3 sd): {emg_threshold:.4f}")

    theta_ratio_thr = theta_delta_ratio_smoothed.mean() + 2.0 * theta_delta_ratio_smoothed.std()
    delta_power_thr = np.array(delta_power_smoothed).mean()
    print(f"  - theta_ratio_thr (mean + 2 sd): {theta_ratio_thr:.4f}")
    print(f"  - delta_power_thr (mean): {delta_power_thr:.4f}")

    wheel_speed_mean_by_epoch = np.array([np.mean(ep) for ep in wheel_speed_epochs])

    # raw sleep scoring (per epoch)
    sleep_score_raw = []
    step = max(1, n_epochs // 20)

    for i in range(n_epochs):
        emg_val = emg_rms_mean_by_epoch[i]
        theta_ratio_val = theta_delta_ratio_smoothed[i]
        delta_power_val = delta_power_smoothed[i]
        wheel_val = wheel_speed_mean_by_epoch[i]

        high_emg_or_running = (
            emg_val >= emg_threshold or np.abs(wheel_val) > 1.0
        )

        if high_emg_or_running:
            state = state_active_wake
        elif theta_ratio_val > theta_ratio_thr and emg_val < emg_threshold:
            state = state_rem
        elif delta_power_val >= delta_power_thr and emg_val < emg_threshold:
            state = state_nrem
        else:
            state = state_quiet_wake

        sleep_score_raw.append(state)

        if (i + 1) % step == 0 or (i + 1) == n_epochs:
            progress = (i + 1) / n_epochs * 100
            print(f"\r    progress scoring: {progress:5.1f}%", end="")
    print()

    sleep_score_raw = np.array(sleep_score_raw, dtype=int)

    if make_qc_figs:
        print("  - plotting hypnogram, emg, and wheel")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(epoch_time, sleep_score_raw, linewidth=0.5)
        axes[0].set_title('hypnogram (raw states)')
        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].set_yticklabels(
            [state_labels[i] for i in [0, 1, 2, 3]]
        )
        axes[1].plot(epoch_time, emg_rms_mean_by_epoch, linewidth=0.5)
        axes[1].set_title('emg_rms_mean_by_epoch')
        axes[2].plot(epoch_time, wheel_speed_mean_by_epoch, linewidth=0.5)
        axes[2].set_title('wheel_speed_mean_by_epoch')

        fig.tight_layout()
        fig.savefig(os.path.join(figs_folder, '06_hypnogram_emg_wheel.png'))
        plt.close(fig)

    # -----------------------------------------------------------------
    # downsample eeg, emg, rms, wheel to 10 hz
    # -----------------------------------------------------------------
    print("downsampling signals to 10 hz")
    eeg_10hz, t_10hz = downsample_to_target_fs(eeg_filtered, fs, target_fs, t=t_ephys)
    emg_10hz, _ = downsample_to_target_fs(emg_filtered, fs, target_fs, t=t_ephys)
    emg_rms_10hz, _ = downsample_to_target_fs(emg_rms, fs, target_fs, t=t_ephys)
    wheel_10hz, _ = downsample_to_target_fs(wheel_speed_resampled, fs, target_fs, t=t_ephys)

    # -----------------------------------------------------------------
    # interpolate state to 10 hz over full experiment
    # -----------------------------------------------------------------
    print("interpolating state to 10 hz")
    state_interp_fun = interp1d(
        epoch_time,
        sleep_score_raw,
        kind='nearest',
        bounds_error=False,
        fill_value=(sleep_score_raw[0], sleep_score_raw[-1])
    )
    state_10hz = state_interp_fun(t_10hz).astype(int)

    # -----------------------------------------------------------------
    # build output structure
    # -----------------------------------------------------------------
    print("building output structures")
    epoch_time_s = epoch_time.astype(float)

    theta_power_epochs = np.array(theta_power_epochs, dtype=np.float32)
    delta_power_epochs = np.array(delta_power_epochs, dtype=np.float32)

    sleep_state = {
        # sampling info
        'fs_raw': fs,
        'fs_downsampled': target_fs,

        # 10 hz traces and time vector
        'emg_10hz': emg_10hz.astype(np.float32),
        'emg_10hz_t': t_10hz.astype(np.float64),
        'emg_rms_10hz': emg_rms_10hz.astype(np.float32),
        'emg_rms_10hz_t': t_10hz.astype(np.float64),
        'eeg_10hz': eeg_10hz.astype(np.float32),
        'eeg_10hz_t': t_10hz.astype(np.float64),
        'wheel_10hz': wheel_10hz.astype(np.float32),
        'wheel_10hz_t': t_10hz.astype(np.float64),

        # epoch-level features
        'epoch_t': epoch_time_s,
        'theta_power': theta_power_epochs,
        'delta_power': delta_power_epochs,

        # moving-window spectrogram (5 s window, 2 s stride)
        'eeg_spectrogram': spec_sxx.astype(np.float32),           # shape (freqs × time_bins)
        'eeg_spectrogram_freqs': spec_freqs.astype(np.float32),
        'eeg_spectrogram_t': spec_t_abs.astype(np.float64),

        # state classification (epoch and 10 hz)
        # 0 = active wake, 1 = quiet wake, 2 = nrem, 3 = rem
        'state_epoch': sleep_score_raw.astype(np.int8),
        'state_epoch_t': epoch_time_s,
        'state_10hz': state_10hz.astype(np.int8),
        'state_10hz_t': t_10hz.astype(np.float64),
        'state_labels': state_labels,
    }

    # -----------------------------------------------------------------
    # save structure
    # -----------------------------------------------------------------
    print("saving outputs")
    sleep_state_path = os.path.join(sleep_score_folder, 'sleep_state.pickle')
    with open(sleep_state_path, 'wb') as f:
        pickle.dump(sleep_state, f)

    duration = time.time() - start_time
    print(f"done. scoring complete in {duration:.2f} s")
    print(f"  sleep_state: {sleep_state_path}")

    return sleep_state


# ---------------------------------------------------------------------
# example usage (from GUI or CLI)
# ---------------------------------------------------------------------
# From another script (e.g. gui.py) in the same folder:
#
#   from sleep_score import run_sleep_scoring
#
#   animal_id = 'pmateosaparicio'
#   exp_id = '2025-06-12_04_ESPM135'
#   sleep_state = run_sleep_scoring(animal_id, exp_id)
#
# This will create:
#   <exp_dir_processed>/sleep_score/sleep_state.pickle
# and QC figures in:
#   <exp_dir_processed>/sleep_score/figs/


if __name__ == "__main__":
    # direct command-line run using defaults
    run_sleep_scoring(default_animal_id, default_exp_id)
