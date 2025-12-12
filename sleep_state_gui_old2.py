import sys
import os
import pickle
import shutil
import cv2
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector

import organise_paths
from sleep_score import run_sleep_scoring


class VideoAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loaded = False
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.playFrame)
        self.vlines = []

        # number of timeline samples (10 Hz)
        self.total_frames = 0

        # drag selection (time range, seconds)
        self.selection_range = None
        self.selection_patch = None
        self.span_selectors = []

        # state label map (0–3 -> name)
        self.state_label_map = None

        # thresholds (may or may not be present in file)
        self.emg_rms_threshold = None
        self.wheel_speed_threshold = None
        self.theta_delta_ratio_threshold = None
        self.delta_power_threshold = None

        # eye frame times (timeline time of each video frame)
        self.eye_frame_times = None

        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Sleep State GUI")
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)
        
        # --- Top Input Fields ---
        inputLayout = QHBoxLayout()
        self.userIdEdit = QLineEdit()
        self.userIdEdit.setPlaceholderText("Enter User ID")
        self.expIdEdit = QLineEdit()
        self.expIdEdit.setPlaceholderText("Enter Experiment ID")
        self.loadButton = QPushButton("Load Data")
        self.loadButton.clicked.connect(self.loadData)

        # explicit "Run Sleep Scoring" button
        self.scoreButton = QPushButton("Run Sleep Scoring")
        self.scoreButton.clicked.connect(self.runSleepScoringClicked)

        inputLayout.addWidget(QLabel("User ID:"))
        inputLayout.addWidget(self.userIdEdit)
        inputLayout.addWidget(QLabel("Experiment ID:"))
        inputLayout.addWidget(self.expIdEdit)
        inputLayout.addWidget(self.loadButton)
        inputLayout.addWidget(self.scoreButton)
        mainLayout.addLayout(inputLayout)
        
        # --- Video controls ---
        controlLayout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.updateFrame)

        self.playButton = QPushButton("Play")
        self.playButton.setEnabled(False)
        self.playButton.clicked.connect(self.startPlayback)

        self.stopButton = QPushButton("Stop")
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self.stopPlayback)

        self.frameJumpEdit = QLineEdit()
        self.frameJumpEdit.setPlaceholderText("Time (s)")
        self.frameJumpEdit.setFixedWidth(100)
        self.frameJumpEdit.setEnabled(False)
        self.frameJumpEdit.returnPressed.connect(self.jumpToTypedFrame)

        self.centerViewBtn = QPushButton("Jump to View Center")
        self.centerViewBtn.setEnabled(False)
        self.centerViewBtn.clicked.connect(self.jumpToViewCenter)

        controlLayout.addWidget(self.slider)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(QLabel("Current / Go to:"))
        controlLayout.addWidget(self.frameJumpEdit)
        controlLayout.addWidget(self.centerViewBtn)
        mainLayout.addLayout(controlLayout)
        
        # --- Video display ---
        videoLayout = QHBoxLayout()
        self.leftVideoLabel = QLabel("Left Eye Video")
        self.leftVideoLabel.setFixedSize(320, 240)
        self.rightVideoLabel = QLabel("Right Eye Video")
        self.rightVideoLabel.setFixedSize(320, 240)
        videoLayout.addWidget(self.leftVideoLabel)
        videoLayout.addWidget(self.rightVideoLabel)
        mainLayout.addLayout(videoLayout)

        # --- Spectrogram saturation + locomotion SD controls ---
        satLayout = QHBoxLayout()
        satLayout.addWidget(QLabel("Spectrogram saturation (% in 0–20 Hz):"))
        self.satPercentEdit = QLineEdit("90")
        self.satPercentEdit.setFixedWidth(60)
        self.satPercentEdit.setEnabled(False)
        self.satPercentEdit.editingFinished.connect(self.plotTraces)
        satLayout.addWidget(self.satPercentEdit)

        satLayout.addSpacing(20)
        satLayout.addWidget(QLabel("Locomotion SD threshold:"))
        self.locSdEdit = QLineEdit("1.0")
        self.locSdEdit.setFixedWidth(60)
        self.locSdEdit.setEnabled(False)
        self.locSdEdit.editingFinished.connect(self.plotTraces)
        satLayout.addWidget(self.locSdEdit)

        satLayout.addStretch(1)
        mainLayout.addLayout(satLayout)
        
        # --- Matplotlib canvas ---
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        # make canvas expand with window size
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        mainLayout.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        mainLayout.addWidget(self.toolbar)

        # --- shortcuts for state assignment (0–3) ---
        self.state_shortcuts = []
        for s in range(4):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(str(s)), self)
            sc.activated.connect(lambda val=s: self.assign_state_key(val))
            self.state_shortcuts.append(sc)

    def loadData(self):
        self.userID = self.userIdEdit.text().strip()
        self.expID = self.expIdEdit.text().strip()
        if not self.userID or not self.expID:
            QMessageBox.warning(self, "Input Error", "Please enter both User ID and Experiment ID")
            return
        
        # paths
        self.animalID, self.remote_repository_root, self.processed_root, \
            self.exp_dir_processed, self.exp_dir_raw = organise_paths.find_paths(self.userID, self.expID)
        self.exp_dir_processed_recordings = os.path.join(self.exp_dir_processed, 'recordings')

        # --- metadata with state labels (if present) ---
        self.state_label_map = None
        metadata_path = os.path.join(self.exp_dir_processed, "metadata.pickle")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                if 'state_labels' in metadata:
                    self.state_label_map = dict(metadata['state_labels'])
            except Exception:
                pass

        # --- eye videos ---
        self.video_path_left = os.path.join(self.exp_dir_processed, f"{self.expID}_eye1_left.avi")
        self.video_path_right = os.path.join(self.exp_dir_processed, f"{self.expID}_eye1_right.avi")
        
        if not os.path.isfile(self.video_path_left):
            try:
                shutil.copyfile(os.path.join(self.exp_dir_raw, f"{self.expID}_eye1_left.avi"), self.video_path_left)
                shutil.copyfile(os.path.join(self.exp_dir_raw, f"{self.expID}_eye1_right.avi"), self.video_path_right)
            except Exception:
                QMessageBox.critical(self, "File Error", "Eye videos not found. Please check the paths.")
                return
        
        # eye data (read-only)
        try:
            with open(os.path.join(self.exp_dir_processed_recordings, 'dlcEyeLeft.pickle'), "rb") as file:
                self.left_eyedat = pickle.load(file)
            with open(os.path.join(self.exp_dir_processed_recordings, 'dlcEyeRight.pickle'), "rb") as file:
                self.right_eyedat = pickle.load(file)
        except Exception:
            QMessageBox.critical(self, "Data Error", "Error loading pupil data.")
            return

        # --- eye_frame_times.npy (required for proper alignment) ---
        eye_times_path = os.path.join(self.exp_dir_processed_recordings, "eye_frame_times.npy")
        if not os.path.isfile(eye_times_path):
            QMessageBox.critical(self, "File Error", "eye_frame_times.npy not found in processed experiment folder.")
            return
        try:
            self.eye_frame_times = np.load(eye_times_path).astype(float)
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Error loading eye_frame_times.npy:\n{e}")
            return

        # --- sleep_state data (load or offer to generate) ---
        sleep_state_path = os.path.join(self.exp_dir_processed, "sleep_score", "sleep_state.pickle")
        sleep_state = None

        if os.path.isfile(sleep_state_path):
            try:
                with open(sleep_state_path, "rb") as f:
                    sleep_state = pickle.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Data Error", f"Error loading sleep_state.pickle:\n{e}")
                return
        else:
            reply = QMessageBox.question(
                self,
                "Sleep scoring not found",
                "sleep_state.pickle was not found.\nDo you want to run sleep scoring now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    # pass userID and expID to scoring
                    sleep_state = run_sleep_scoring(self.userID, self.expID)
                except Exception as e:
                    QMessageBox.critical(self, "Sleep scoring error", f"Error running sleep scoring:\n{e}")
                    return
                if sleep_state is None:
                    QMessageBox.critical(self, "Sleep scoring error", "run_sleep_scoring returned no data.")
                    return
            else:
                QMessageBox.information(self, "Sleep scoring", "Load cancelled: sleep scoring data not available.")
                return

        # 10 Hz traces
        self.emg_rms_10hz   = np.asarray(sleep_state['emg_rms_10hz'],   dtype=float)
        self.emg_rms_10hz_t = np.asarray(sleep_state['emg_rms_10hz_t'], dtype=float)

        self.wheel_10hz     = np.asarray(sleep_state['wheel_10hz'],     dtype=float)
        self.wheel_10hz_t   = np.asarray(sleep_state['wheel_10hz_t'],   dtype=float)

        self.eeg_10hz       = np.asarray(sleep_state['eeg_10hz'],       dtype=float)
        self.eeg_10hz_t     = np.asarray(sleep_state['eeg_10hz_t'],     dtype=float)

        # epoch-level
        self.epoch_t        = np.asarray(sleep_state['epoch_t'],        dtype=float)
        self.theta_power    = np.asarray(sleep_state['theta_power'],    dtype=float)
        self.delta_power    = np.asarray(sleep_state['delta_power'],    dtype=float)

        # spectrogram
        self.eeg_spectrogram       = np.asarray(sleep_state['eeg_spectrogram'],       dtype=float)
        self.eeg_spectrogram_freqs = np.asarray(sleep_state['eeg_spectrogram_freqs'], dtype=float)
        self.eeg_spectrogram_t     = np.asarray(sleep_state['eeg_spectrogram_t'],     dtype=float)

        # state
        self.state_epoch    = np.asarray(sleep_state['state_epoch'],    dtype=int)
        self.state_epoch_t  = np.asarray(sleep_state['state_epoch_t'],  dtype=float)
        self.state_10hz     = np.asarray(sleep_state['state_10hz'],     dtype=int)
        self.state_10hz_t   = np.asarray(sleep_state['state_10hz_t'],   dtype=float)

        # thresholds (if present)
        self.emg_rms_threshold = sleep_state.get('emg_rms_threshold', None)
        self.wheel_speed_threshold = sleep_state.get('wheel_speed_threshold', None)
        self.theta_delta_ratio_threshold = sleep_state.get('theta_delta_ratio_threshold', None)
        self.delta_power_threshold = sleep_state.get('delta_power_threshold', None)

        # fallback: if metadata didn't provide labels, use sleep_state['state_labels']
        if self.state_label_map is None and 'state_labels' in sleep_state:
            labels_obj = sleep_state['state_labels']
            if isinstance(labels_obj, dict):
                self.state_label_map = dict(labels_obj)
            else:
                self.state_label_map = {i: str(name) for i, name in enumerate(labels_obj)}

        # ---- debug: compare time spans (using eye_frame_times, not FPS) ----
        video_min = float(np.nanmin(self.eye_frame_times))
        video_max = float(np.nanmax(self.eye_frame_times))

        # collect sleep-state time vectors
        time_arrays = [
            self.emg_rms_10hz_t,
            self.wheel_10hz_t,
            self.eeg_10hz_t,
            self.epoch_t,
            self.eeg_spectrogram_t,
            self.state_epoch_t,
            self.state_10hz_t,
        ]
        data_min = min(np.nanmin(a) for a in time_arrays if a.size > 0)
        data_max = max(np.nanmax(a) for a in time_arrays if a.size > 0)

        print(f"[DEBUG] Video timeline span (eye_frame_times): {video_min:.3f} to {video_max:.3f} s")
        print(f"[DEBUG] Sleep data time span: {data_min:.3f} to {data_max:.3f} s")

        # ---- slider over 10 Hz timeline ----
        self.total_frames = len(self.emg_rms_10hz_t)
        if self.total_frames <= 0:
            QMessageBox.critical(self, "Data Error", "emg_rms_10hz_t is empty.")
            return

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setEnabled(True)
        self.playButton.setEnabled(True)
        self.stopButton.setEnabled(True)

        self.frameJumpEdit.setEnabled(True)
        self.centerViewBtn.setEnabled(True)
        self.frameJumpEdit.setText(f"{self.emg_rms_10hz_t[0]:.3f}")

        self.satPercentEdit.setEnabled(True)
        self.locSdEdit.setEnabled(True)

        self.loaded = True
        
        self.updateFrame()
        self.plotTraces()

    def runSleepScoringClicked(self):
        """
        Manual button to run sleep scoring using the IDs from the GUI.
        Does NOT reload plots automatically; user can press Load Data afterwards.
        """
        user_id = self.userIdEdit.text().strip()
        exp_id = self.expIdEdit.text().strip()
        if not user_id or not exp_id:
            QMessageBox.warning(self, "Input Error", "Please enter both User ID and Experiment ID")
            return

        # resolve paths (if needed)
        try:
            organise_paths.find_paths(user_id, exp_id)
        except Exception as e:
            QMessageBox.critical(self, "Path Error", f"Error resolving paths:\n{e}")
            return

        try:
            run_sleep_scoring(user_id, exp_id)
        except Exception as e:
            QMessageBox.critical(self, "Sleep scoring error", f"Error running sleep scoring:\n{e}")
            return

        QMessageBox.information(self, "Sleep scoring", "Sleep scoring finished successfully.\nYou can now press 'Load Data'.")

    # --- timeline helpers ---
    def _current_time(self, idx=None):
        """
        Return current timeline time (seconds) based on 10 Hz EMG time vector.
        """
        if not self.loaded:
            return 0.0
        if idx is None:
            idx = self.slider.value()
        idx = max(0, min(int(idx), self.total_frames - 1))
        return float(self.emg_rms_10hz_t[idx])

    # --- video overlay ---
    def overlay_plot(self, frame, position, eyeDat):
        try:
            if (
                np.isnan(eyeDat['x'][position]) or
                np.isnan(eyeDat['y'][position]) or
                np.isnan(eyeDat['radius'][position])
            ):
                return frame
        except Exception:
            return frame

        color = (0, 0, 255)
        center = (int(eyeDat['x'][position]), int(eyeDat['y'][position]))
        radius = int(eyeDat['radius'][position])
        frame = cv2.circle(frame, center, radius, color, 2)
        return frame

    def playVideoFrame(self, frame_position, video_path, eyedat):
        """
        frame_position is the zero-based frame index in the video file.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:, :, 0]
        p70 = np.percentile(frame, 70)
        frame[frame >= p70] = p70
        min_val = np.min(frame)
        max_val = np.max(frame)
        if max_val > min_val:
            frame = (frame - min_val) / (max_val - min_val) * 255
        frame = np.stack((frame,) * 3, axis=-1).astype(np.uint8)
        frame = self.overlay_plot(frame, int(frame_position), eyedat)
        return frame

    def updateFrame(self):
        if not self.loaded:
            return
        
        # current timeline time from slider
        t = self._current_time()

        # find nearest video frame index using eye_frame_times
        if self.eye_frame_times is not None and self.eye_frame_times.size > 0:
            idx_frame = int(np.argmin(np.abs(self.eye_frame_times - t)))
        else:
            idx_frame = 0

        # Update left video frame
        frame_left = self.playVideoFrame(idx_frame, self.video_path_left, self.left_eyedat)
        if frame_left is not None:
            image_left = QtGui.QImage(
                frame_left.data, frame_left.shape[1], frame_left.shape[0],
                frame_left.strides[0], QtGui.QImage.Format_RGB888
            )
            pixmap_left = QtGui.QPixmap.fromImage(image_left)
            self.leftVideoLabel.setPixmap(pixmap_left.scaled(self.leftVideoLabel.size(), Qt.KeepAspectRatio))
        
        # Update right video frame
        frame_right = self.playVideoFrame(idx_frame, self.video_path_right, self.right_eyedat)
        if frame_right is not None:
            image_right = QtGui.QImage(
                frame_right.data, frame_right.shape[1], frame_right.shape[0],
                frame_right.strides[0], QtGui.QImage.Format_RGB888
            )
            pixmap_right = QtGui.QPixmap.fromImage(image_right)
            self.rightVideoLabel.setPixmap(pixmap_right.scaled(self.rightVideoLabel.size(), Qt.KeepAspectRatio))
        
        # move vertical line in time
        if self.vlines:
            for vline in self.vlines:
                vline.set_xdata(t)
            self.canvas.draw_idle()

        # update time box with current time
        if self.frameJumpEdit.isEnabled():
            self.frameJumpEdit.setText(f"{t:.3f}")

    def startPlayback(self):
        if not self.loaded:
            return
        self.playing = True
        self.timer.start(33)

    def stopPlayback(self):
        self.playing = False
        self.timer.stop()

    def playFrame(self):
        if self.slider.value() < self.total_frames - 1:
            self.slider.setValue(self.slider.value() + 1)
        else:
            self.stopPlayback()

    # --- navigation ---
    def jumpToViewCenter(self):
        if not self.loaded or not self.figure.axes:
            return
        bottom_ax = self.figure.axes[-1]
        xmin, xmax = bottom_ax.get_xlim()
        center_time = 0.5 * (xmin + xmax)

        # map center_time to nearest index in 10 Hz timeline
        idx = int(np.argmin(np.abs(self.emg_rms_10hz_t - center_time)))
        idx = max(0, min(idx, self.total_frames - 1))
        self.slider.setValue(idx)

    def jumpToTypedFrame(self):
        """
        Interpret the text box as *time in seconds* and jump to the nearest 10 Hz sample.
        """
        if not self.loaded:
            return
        text = self.frameJumpEdit.text().strip()
        try:
            t = float(text)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Enter a valid time in seconds.")
            return

        # find nearest time in emg_rms_10hz_t
        idx = int(np.argmin(np.abs(self.emg_rms_10hz_t - t)))
        idx = max(0, min(idx, self.total_frames - 1))
        self.slider.setValue(idx)

    # --- span selection ---
    def _setup_span_selectors(self):
        for sp in self.span_selectors:
            try:
                sp.disconnect_events()
            except Exception:
                pass
        self.span_selectors = []

        def make_on_select(ax):
            def on_select(xmin, xmax):
                if not self.loaded:
                    return
                self._set_selection_visual(ax, xmin, xmax)
            return on_select

        for ax in self.figure.axes:
            selector = SpanSelector(
                ax, make_on_select(ax), direction='horizontal',
                useblit=True, interactive=True,
                props=dict(alpha=0.15, facecolor='yellow')
            )
            self.span_selectors.append(selector)

    def _set_selection_visual(self, ax, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self.selection_range = (float(xmin), float(xmax))

        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
            self.selection_patch = None

        self.selection_patch = ax.axvspan(xmin, xmax, color='yellow', alpha=0.3)
        self.canvas.draw_idle()

    # --- helpers ---
    @staticmethod
    def _zscore_safe(x):
        x = np.asarray(x, dtype=float)
        mu = np.nanmean(x)
        sigma = np.nanstd(x)
        if sigma == 0 or not np.isfinite(sigma):
            return np.zeros_like(x)
        return (x - mu) / sigma

    def _compute_epoch_motion_mask(self):
        """
        Returns a boolean array per epoch: True where locomotion is 'high'.
        High = wheel_10hz > mean + N * std, where N is from locSdEdit.
        Any high sample in an epoch marks that epoch as moving.
        """
        if not hasattr(self, "wheel_10hz"):
            return np.zeros_like(self.epoch_t, dtype=bool)

        try:
            n_sd = float(self.locSdEdit.text())
        except Exception:
            n_sd = 1.0
        if n_sd < 0:
            n_sd = 0.0

        w = np.asarray(self.wheel_10hz, dtype=float)
        w_t = np.asarray(self.wheel_10hz_t, dtype=float)

        finite_mask = np.isfinite(w)
        if not np.any(finite_mask):
            return np.zeros_like(self.epoch_t, dtype=bool)

        mu = np.mean(w[finite_mask])
        sigma = np.std(w[finite_mask])
        if sigma <= 0 or not np.isfinite(sigma):
            return np.zeros_like(self.epoch_t, dtype=bool)

        thresh = mu + n_sd * sigma
        fast = np.zeros_like(w, dtype=bool)
        fast[finite_mask] = w[finite_mask] > thresh

        et = np.asarray(self.epoch_t, dtype=float)
        n_epoch = et.size
        epoch_moving = np.zeros(n_epoch, dtype=bool)
        if n_epoch == 0:
            return epoch_moving

        if n_epoch == 1:
            if np.any(fast):
                epoch_moving[0] = True
            return epoch_moving

        mids = (et[:-1] + et[1:]) / 2.0
        first_edge = et[0] - (mids[0] - et[0])
        last_edge = et[-1] + (et[-1] - mids[-1])
        edges = np.concatenate(([first_edge], mids, [last_edge]))

        for i in range(n_epoch):
            t0 = edges[i]
            t1 = edges[i + 1]
            mask = (w_t >= t0) & (w_t < t1)
            if np.any(fast[mask]):
                epoch_moving[i] = True

        return epoch_moving

    # --- state assignment by key (0–3) ---
    def assign_state_key(self, state_value: int):
        if not self.loaded or self.selection_range is None:
            return
        tmin, tmax = self.selection_range

        # update epoch-level state
        mask_epoch = (self.state_epoch_t >= tmin) & (self.state_epoch_t <= tmax)
        self.state_epoch[mask_epoch] = int(state_value)

        # update 10 Hz state
        mask_10 = (self.state_10hz_t >= tmin) & (self.state_10hz_t <= tmax)
        self.state_10hz[mask_10] = int(state_value)

        # clear selection
        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
            self.selection_patch = None
        self.selection_range = None

        self.plotTraces()

    # --- plotting ---
    def plotTraces(self):
        if not hasattr(self, "emg_rms_10hz"):
            return

        # read saturation percentage
        try:
            sat_pct = float(self.satPercentEdit.text())
        except Exception:
            sat_pct = 90.0
        sat_pct = min(max(sat_pct, 1.0), 99.0)

        # epoch motion mask and masked theta/delta
        epoch_moving = self._compute_epoch_motion_mask()
        theta_masked = self.theta_power.copy()
        delta_masked = self.delta_power.copy()
        theta_masked[epoch_moving] = np.nan
        delta_masked[epoch_moving] = np.nan

        # ratio delta/theta using non-z-scored values
        ratio = theta_masked/delta_masked
        ratio[~np.isfinite(ratio)] = np.nan

        # z-scored masked theta/delta
        theta_z = self._zscore_safe(theta_masked)
        delta_z = self._zscore_safe(delta_masked)

        # delta power threshold in z-units (for plotting on z-scored axis)
        delta_thr_z = None
        if self.delta_power_threshold is not None:
            mu_delta = np.nanmean(delta_masked)
            sigma_delta = np.nanstd(delta_masked)
            if sigma_delta > 0 and np.isfinite(sigma_delta):
                delta_thr_z = (float(self.delta_power_threshold) - mu_delta) / sigma_delta

        self.figure.clear()
        # 6 axes, spectrogram 3x height
        axs = self.figure.subplots(
            6, 1, sharex=True,
            gridspec_kw={'height_ratios': [1, 1, 3, 1, 1, 1]}
        )

        # colors
        emg_color = 'tab:blue'
        wheel_color = 'tab:orange'

        # 1) EMG RMS + wheel (two y-axes)
        ax1 = axs[0]
        ax1.plot(self.emg_rms_10hz_t, self.emg_rms_10hz, color=emg_color, label="EMG RMS")
        ax1.set_ylabel("EMG RMS", color=emg_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='y', labelcolor=emg_color)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.spines['left'].set_color(emg_color)

        # EMG threshold (same color as EMG line)
        if self.emg_rms_threshold is not None:
            ax1.axhline(float(self.emg_rms_threshold), color=emg_color, linestyle='--', linewidth=1)

        ax1b = ax1.twinx()
        ax1b.plot(self.wheel_10hz_t, self.wheel_10hz, color=wheel_color, label="Wheel")
        ax1b.set_ylabel("Wheel", color=wheel_color)
        ax1b.spines['top'].set_visible(False)
        ax1b.tick_params(axis='y', labelcolor=wheel_color)
        ax1b.spines['right'].set_color(wheel_color)

        # wheel speed threshold (same color as wheel line)
        if self.wheel_speed_threshold is not None:
            ax1b.axhline(float(self.wheel_speed_threshold), color=wheel_color, linestyle='--', linewidth=1)

        # 2) EEG 10 Hz
        ax2 = axs[1]
        ax2.plot(self.eeg_10hz_t, self.eeg_10hz)
        ax2.set_ylabel("EEG (10 Hz)")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='x', labelbottom=False)

        # 3) EEG spectrogram, 0–20 Hz, saturation, pixelated
        ax3 = axs[2]
        max_freq = 20.0
        if (
            self.eeg_spectrogram.size > 0 and
            self.eeg_spectrogram_freqs.size == self.eeg_spectrogram.shape[0]
        ):
            freqs = self.eeg_spectrogram_freqs.astype(float)
            spec = self.eeg_spectrogram

            band_mask = (freqs >= 0) & (freqs <= max_freq)
            if np.any(band_mask):
                freqs_band = freqs[band_mask]
                spec_band = spec[band_mask, :]

                t0 = float(self.eeg_spectrogram_t[0])
                t1 = float(self.eeg_spectrogram_t[-1])

                vals = spec_band[np.isfinite(spec_band)]
                vmin = vmax = None
                if vals.size > 0:
                    low_p = (100.0 - sat_pct) / 2.0
                    high_p = 100.0 - low_p
                    vmin = np.percentile(vals, low_p)
                    vmax = np.percentile(vals, high_p)
                    if vmin == vmax:
                        vmin -= 1e-6
                        vmax += 1e-6

                im_args = dict(
                    aspect="auto",
                    origin="lower",
                    extent=[t0, t1, float(freqs_band[0]), float(freqs_band[-1])],
                    interpolation='nearest'
                )
                if vmin is not None and vmax is not None:
                    im_args["vmin"] = vmin
                    im_args["vmax"] = vmax

                ax3.imshow(spec_band, **im_args)
                ax3.set_ylim(0, max_freq)

        # white dotted lines for delta (1–4 Hz) and theta (4–8 Hz)
        for y in [1.0, 4.0, 8.0]:
            ax3.axhline(y=y, color='white', linestyle=':', linewidth=1)

        ax3.set_ylabel("Freq (Hz)")
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(axis='x', labelbottom=False)

        # 4) Theta + Delta, z-scored, same axis
        ax4 = axs[3]
        ax4.plot(self.epoch_t, theta_z, label="Theta (z)")
        ax4.plot(self.epoch_t, delta_z, label="Delta (z)")
        ax4.set_ylabel("Power (z)")
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(axis='x', labelbottom=False)
        ax4.legend(loc="upper right")

        # delta power threshold on z-scored axis
        if delta_thr_z is not None:
            ax4.axhline(delta_thr_z, color='r', linestyle='--', linewidth=1)

        # 5) Delta/Theta ratio
        ax5 = axs[4]
        ax5.plot(self.epoch_t, ratio)
        ax5.set_ylabel("Δ/Θ")
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.tick_params(axis='x', labelbottom=False)

        # theta/delta ratio threshold
        if self.theta_delta_ratio_threshold is not None:
            ax5.axhline(float(self.theta_delta_ratio_threshold), color='r', linestyle='--', linewidth=1)

        # 6) State (epoch-level) — color-coded line and labels
        ax6 = axs[5]
        ax6.set_ylabel("State")
        ax6.set_xlabel("Time (s)")
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)

        t = self.state_epoch_t
        s = self.state_epoch.astype(int)
        color_map = {0: 'k', 1: '0.5', 2: 'tab:blue', 3: 'tab:green'}

        # color-coded step-like segments per state
        for state_val, col in color_map.items():
            mask = (s == state_val)
            y = np.where(mask, s, np.nan)
            ax6.step(t, y, where="post", color=col, linewidth=1.5)

        # y tick labels, color-coded
        if self.state_label_map is not None:
            try:
                keys = sorted(int(k) for k in self.state_label_map.keys())
                labels = [self.state_label_map[k] for k in keys]
                ax6.set_yticks(keys)
                ax6.set_yticklabels(labels)
                for tick, key in zip(ax6.get_yticklabels(), keys):
                    tick.set_color(color_map.get(key, 'k'))
            except Exception:
                pass
        else:
            ticks = [0, 1, 2, 3]
            ax6.set_yticks(ticks)
            ax6.set_yticklabels([str(tk) for tk in ticks])
            for tick, key in zip(ax6.get_yticklabels(), ticks):
                tick.set_color(color_map.get(key, 'k'))

        # vertical line in all plots (current time)
        self.vlines = []
        current_time = self._current_time()
        for ax in axs:
            vline = ax.axvline(x=current_time, color='k', linestyle='--')
            self.vlines.append(vline)

        self._setup_span_selectors()
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoAnalysisApp()
    win.showMaximized()
    sys.exit(app.exec_())
