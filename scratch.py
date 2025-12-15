import os
import numpy as np
import pandas as pd

# ----- paths -----
parquet_path = "/home/adamranson/data/mouse_EEG/mouse3.parquet"
out_dir = os.path.dirname(parquet_path)
base = os.path.splitext(os.path.basename(parquet_path))[0]
out_path = os.path.join(out_dir, f"{base}.npz")  # same name, different extension

# ----- sampling rates -----
fs_old = 515.0
fs_new = 1000.0

# ----- load -----
df = pd.read_parquet(parquet_path)
eeg = df["eeg"].to_numpy(dtype=np.float64)
emg = df["emg"].to_numpy(dtype=np.float64)

# ----- build time axes -----
n = eeg.size
t_old = np.arange(n, dtype=np.float64) / fs_old

# choose output length to preserve total duration as closely as possible
n_new = int(np.round((n - 1) * fs_new / fs_old)) + 1
t_new = np.arange(n_new, dtype=np.float64) / fs_new

# ----- interpolate (linear) -----
eeg_up = np.interp(t_new, t_old, eeg)
emg_up = np.interp(t_new, t_old, emg)

# ----- save -----
# .npz supports multiple named arrays ("fields")
np.savez(out_path, eeg=eeg_up, emg=emg_up, fs=fs_new, fs_original=fs_old)

print(f"Saved: {out_path}")
print(f"Original samples: {n} @ {fs_old} Hz")
print(f"Upsampled samples: {n_new} @ {fs_new} Hz")
