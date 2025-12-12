import sys
import os
import pickle
import shutil
import cv2
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QMessageBox, QSizePolicy, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, Slider
from scipy import signal
from scipy.interpolate import interp1d

import organise_paths
from sleep_score import (
    run_sleep_scoring,
    score_from_epoch_features,
    build_epoch_feature_dict,
)


# -----------------------------
# Robust draggable threshold line
# -----------------------------
class DraggableHLine:
    """
    Robust draggable horizontal line using explicit proximity detection.
    - Creates axhline internally.
    - Binds to a single figure canvas.
    - No picking, no widgetlock, no toolbar logic.
    """

    def __init__(
        self,
        ax,
        y0,
        color="r",
        linestyle="--",
        linewidth=1.5,
        tolerance_px=6,
        on_changed=None,
        is_enabled_fn=None,
    ):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.on_changed = on_changed
        self.tolerance_px = tolerance_px
        self.is_enabled_fn = is_enabled_fn

        self.line = ax.axhline(
            y0,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=10,
        )

        self._dragging = False

        self.cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)

    def disconnect(self):
        for cid in [self.cid_press, self.cid_motion, self.cid_release]:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass

    def get_y(self):
        return float(self.line.get_ydata()[0])

    def set_y(self, y, trigger_callback=True):
        self.line.set_ydata([y, y])
        if trigger_callback and self.on_changed is not None:
            self.on_changed(float(y))
        self.canvas.draw_idle()

    def _enabled(self):
        if self.is_enabled_fn is None:
            return True
        try:
            return bool(self.is_enabled_fn())
        except Exception:
            return True

    def _is_near_line(self, event):
        if event.inaxes is not self.ax:
            return False
        if event.ydata is None:
            return False

        y_line = self.get_y()
        x_dummy = 0.0

        p_event = self.ax.transData.transform((x_dummy, event.ydata))
        p_line = self.ax.transData.transform((x_dummy, y_line))
        dy = abs(p_event[1] - p_line[1])

        return dy <= self.tolerance_px

    def _on_press(self, event):
        if not self._enabled():
            return
        if event.button != 1:
            return
        if self._is_near_line(event):
            self._dragging = True

    def _on_motion(self, event):
        if not self._enabled():
            return
        if not self._dragging:
            return
        if event.inaxes is not self.ax:
            return
        if event.ydata is None:
            return
        self.set_y(event.ydata, trigger_callback=True)

    def _on_release(self, event):
        self._dragging = False


# -----------------------------
# GUI
# -----------------------------
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

        # drag selection
        self.selection_range = None
        self.selection_patch = None
        self.span_selectors = []

        # state label map
        self.state_label_map = None

        # thresholds
        self.emg_rms_threshold = None
        self.wheel_speed_threshold = None
        self.theta_delta_ratio_threshold = None
        self.delta_power_threshold = None

        # eye frame times
        self.eye_frame_times = None

        # cached epoch features for rescoring
        self.epoch_features_data = None
        self._state_dirty = False

        # Matplotlib widgets/objects
        self.figure = None
        self.canvas = None
        self.toolbar = None

        self._axs = None
        self._ax_emg = None
        self._ax_wheel = None
        self._ax_ratio = None
        self._ax_state = None

        self._thr_drag_emg = None
        self._thr_drag_wheel = None
        self._thr_drag_ratio = None

        self._slider_emg = None
        self._slider_wheel = None
        self._slider_ratio = None

        self._syncing_slider = False

        # cached epoch features for reclassification
        self.emg_rms_mean_by_epoch = None
        self.wheel_speed_mean_by_epoch = None
        self.theta_delta_ratio_smoothed = None
        self.delta_power_smoothed = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Sleep State GUI")
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)

        # --- Top input fields ---
        inputLayout = QHBoxLayout()
        self.userIdEdit = QLineEdit()
        self.userIdEdit.setPlaceholderText("Enter User ID")
        self.expIdEdit = QLineEdit()
        self.expIdEdit.setPlaceholderText("Enter Experiment ID")

        self.loadButton = QPushButton("Load Data")
        self.loadButton.clicked.connect(self.loadData)

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

        # --- Spectrogram saturation + locomotion SD controls + threshold edit toggle ---
        satLayout = QHBoxLayout()
        satLayout.addWidget(QLabel("Spectrogram saturation (% in 0–20 Hz):"))
        self.satPercentEdit = QLineEdit("90")
        self.satPercentEdit.setFixedWidth(60)
        self.satPercentEdit.setEnabled(False)
        self.satPercentEdit.editingFinished.connect(self.plotTraces)
        satLayout.addWidget(self.satPercentEdit)

        satLayout.addSpacing(20)
        satLayout.addWidget(QLabel("Locomotion SD threshold (masking):"))
        self.locSdEdit = QLineEdit("1.0")
        self.locSdEdit.setFixedWidth(60)
        self.locSdEdit.setEnabled(False)
        self.locSdEdit.editingFinished.connect(self.plotTraces)
        satLayout.addWidget(self.locSdEdit)

        satLayout.addSpacing(20)
        self.thresholdEditCheck = QCheckBox("Threshold edit mode")
        self.thresholdEditCheck.setEnabled(False)
        self.thresholdEditCheck.setChecked(True)
        satLayout.addWidget(self.thresholdEditCheck)

        satLayout.addSpacing(12)
        self.autoRescoreCheck = QCheckBox("Auto rescore")
        self.autoRescoreCheck.setEnabled(False)
        self.autoRescoreCheck.setChecked(False)
        satLayout.addWidget(self.autoRescoreCheck)

        self.rescoreButton = QPushButton("Rescore")
        self.rescoreButton.setEnabled(False)
        self.rescoreButton.clicked.connect(self.onRescoreClicked)
        satLayout.addWidget(self.rescoreButton)

        satLayout.addStretch(1)
        mainLayout.addLayout(satLayout)

        # --- Matplotlib canvas ---
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        mainLayout.addWidget(self.canvas)

        # Toolbar retained (no functionality loss)
        self.toolbar = NavigationToolbar(self.canvas, self)
        mainLayout.addWidget(self.toolbar)

        # --- shortcuts for state assignment (0–3) ---
        self.state_shortcuts = []
        for s in range(4):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(str(s)), self)
            sc.activated.connect(lambda val=s: self.assign_state_key(val))
            self.state_shortcuts.append(sc)

    # -----------------------------
    # Load data
    # -----------------------------
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

        # metadata with state labels (if present)
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

        # eye videos
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

        # eye_frame_times.npy (required)
        eye_times_path = os.path.join(self.exp_dir_processed_recordings, "eye_frame_times.npy")
        if not os.path.isfile(eye_times_path):
            QMessageBox.critical(self, "File Error", "eye_frame_times.npy not found in processed experiment folder.")
            return
        try:
            self.eye_frame_times = np.load(eye_times_path).astype(float)
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Error loading eye_frame_times.npy:\n{e}")
            return

        # sleep_state data (load or offer to generate)
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

        # thresholds
        self.emg_rms_threshold = float(sleep_state.get('emg_rms_threshold', np.nan))
        self.wheel_speed_threshold = float(sleep_state.get('wheel_speed_threshold', np.nan))
        self.theta_delta_ratio_threshold = float(sleep_state.get('theta_delta_ratio_threshold', np.nan))
        self.delta_power_threshold = float(sleep_state.get('delta_power_threshold', np.nan))

        if not np.isfinite(self.wheel_speed_threshold):
            self.wheel_speed_threshold = 1.0
        if not np.isfinite(self.emg_rms_threshold):
            self.emg_rms_threshold = float(np.nanmedian(self.emg_rms_10hz))
        if not np.isfinite(self.theta_delta_ratio_threshold):
            self.theta_delta_ratio_threshold = 2.0
        if not np.isfinite(self.delta_power_threshold):
            self.delta_power_threshold = float(np.nanmean(self.delta_power))

        stored_features = sleep_state.get("epoch_features")
        self.epoch_features_data = None
        if stored_features is not None:
            try:
                self.epoch_features_data = {
                    key: np.asarray(val, dtype=float)
                    for key, val in stored_features.items()
                }
            except Exception:
                self.epoch_features_data = None

        # fallback labels
        if self.state_label_map is None and 'state_labels' in sleep_state:
            labels_obj = sleep_state['state_labels']
            if isinstance(labels_obj, dict):
                self.state_label_map = dict(labels_obj)
            else:
                self.state_label_map = {i: str(name) for i, name in enumerate(labels_obj)}

        # debug spans
        video_min = float(np.nanmin(self.eye_frame_times))
        video_max = float(np.nanmax(self.eye_frame_times))

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

        # slider over 10 Hz timeline
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
        self.thresholdEditCheck.setEnabled(True)
        self.autoRescoreCheck.setEnabled(True)
        self.autoRescoreCheck.setChecked(False)
        self.rescoreButton.setEnabled(True)

        self.loaded = True
        self._state_dirty = False

        # Build epoch features for interactive reclassification
        self._rebuild_epoch_features()

        self.updateFrame()
        self.plotTraces()

    def runSleepScoringClicked(self):
        user_id = self.userIdEdit.text().strip()
        exp_id = self.expIdEdit.text().strip()
        if not user_id or not exp_id:
            QMessageBox.warning(self, "Input Error", "Please enter both User ID and Experiment ID")
            return

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

    # -----------------------------
    # Timeline helpers
    # -----------------------------
    def _current_time(self, idx=None):
        if not self.loaded:
            return 0.0
        if idx is None:
            idx = self.slider.value()
        idx = max(0, min(int(idx), self.total_frames - 1))
        return float(self.emg_rms_10hz_t[idx])

    # -----------------------------
    # Video overlay
    # -----------------------------
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

        t = self._current_time()

        if self.eye_frame_times is not None and self.eye_frame_times.size > 0:
            idx_frame = int(np.argmin(np.abs(self.eye_frame_times - t)))
        else:
            idx_frame = 0

        frame_left = self.playVideoFrame(idx_frame, self.video_path_left, self.left_eyedat)
        if frame_left is not None:
            image_left = QtGui.QImage(
                frame_left.data, frame_left.shape[1], frame_left.shape[0],
                frame_left.strides[0], QtGui.QImage.Format_RGB888
            )
            pixmap_left = QtGui.QPixmap.fromImage(image_left)
            self.leftVideoLabel.setPixmap(pixmap_left.scaled(self.leftVideoLabel.size(), Qt.KeepAspectRatio))

        frame_right = self.playVideoFrame(idx_frame, self.video_path_right, self.right_eyedat)
        if frame_right is not None:
            image_right = QtGui.QImage(
                frame_right.data, frame_right.shape[1], frame_right.shape[0],
                frame_right.strides[0], QtGui.QImage.Format_RGB888
            )
            pixmap_right = QtGui.QPixmap.fromImage(image_right)
            self.rightVideoLabel.setPixmap(pixmap_right.scaled(self.rightVideoLabel.size(), Qt.KeepAspectRatio))

        if self.vlines:
            for vline in self.vlines:
                try:
                    vline.set_xdata(t)
                except Exception:
                    pass
            self.canvas.draw_idle()

        if self.frameJumpEdit.isEnabled():
            self.frameJumpEdit.setText(f"{t:.3f}")

    # -----------------------------
    # Playback
    # -----------------------------
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

    # -----------------------------
    # Navigation
    # -----------------------------
    def jumpToViewCenter(self):
        if not self.loaded or not self.figure.axes:
            return
        bottom_ax = self.figure.axes[-1]
        xmin, xmax = bottom_ax.get_xlim()
        center_time = 0.5 * (xmin + xmax)

        idx = int(np.argmin(np.abs(self.emg_rms_10hz_t - center_time)))
        idx = max(0, min(idx, self.total_frames - 1))
        self.slider.setValue(idx)

    def jumpToTypedFrame(self):
        if not self.loaded:
            return
        text = self.frameJumpEdit.text().strip()
        try:
            t = float(text)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Enter a valid time in seconds.")
            return

        idx = int(np.argmin(np.abs(self.emg_rms_10hz_t - t)))
        idx = max(0, min(idx, self.total_frames - 1))
        self.slider.setValue(idx)

    # -----------------------------
    # Span selection
    # -----------------------------
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
                ax,
                make_on_select(ax),
                direction='horizontal',
                useblit=True,
                interactive=True,
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

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _zscore_safe(x):
        x = np.asarray(x, dtype=float)
        mu = np.nanmean(x)
        sigma = np.nanstd(x)
        if sigma == 0 or not np.isfinite(sigma):
            return np.zeros_like(x)
        return (x - mu) / sigma

    @staticmethod
    def _adaptive_savgol(x, base_window=11, poly=3):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n < 3:
            return x.copy()
        win = min(int(base_window), n if n % 2 == 1 else n - 1)
        if win < poly + 2:
            return x.copy()
        return signal.savgol_filter(x, window_length=win, polyorder=poly, mode="nearest")

    def _compute_epoch_edges(self, epoch_centers):
        et = np.asarray(epoch_centers, dtype=float)
        n = et.size
        if n == 0:
            return np.array([0.0, 0.0], dtype=float)
        if n == 1:
            # arbitrary 10s width if only one epoch
            return np.array([et[0] - 5.0, et[0] + 5.0], dtype=float)

        mids = (et[:-1] + et[1:]) / 2.0
        first_edge = et[0] - (mids[0] - et[0])
        last_edge = et[-1] + (et[-1] - mids[-1])
        edges = np.concatenate(([first_edge], mids, [last_edge]))
        return edges

    def _epoch_mean_from_timeseries(self, t, x, epoch_centers):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        et = np.asarray(epoch_centers, dtype=float)

        edges = self._compute_epoch_edges(et)
        means = np.full(et.shape, np.nan, dtype=float)

        for i in range(et.size):
            t0, t1 = edges[i], edges[i + 1]
            mask = (t >= t0) & (t < t1) & np.isfinite(x)
            if np.any(mask):
                means[i] = float(np.mean(x[mask]))
        return means

    def _compute_epoch_motion_mask(self):
        """
        Masking only (existing behaviour).
        High = wheel_10hz > mean + N*std; any high sample in epoch marks moving.
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

        edges = self._compute_epoch_edges(et)
        for i in range(n_epoch):
            t0 = edges[i]
            t1 = edges[i + 1]
            mask = (w_t >= t0) & (w_t < t1)
            if np.any(fast[mask]):
                epoch_moving[i] = True

        return epoch_moving

    def _rebuild_epoch_features(self):
        """
        Build epoch-level features needed for interactive reclassification
        from the stored sleep_state fields.
        """
        stored = getattr(self, "epoch_features_data", None)
        if stored is not None:
            self.emg_rms_mean_by_epoch = np.asarray(
                stored.get("emg_rms_mean", []), dtype=float
            )
            self.wheel_speed_mean_by_epoch = np.asarray(
                stored.get("wheel_speed_mean", []), dtype=float
            )
            self.theta_delta_ratio_smoothed = np.asarray(
                stored.get("theta_delta_ratio_smoothed", []), dtype=float
            )
            self.delta_power_smoothed = np.asarray(
                stored.get("delta_power_smoothed", []), dtype=float
            )
            return

        # Epoch means for EMG RMS and wheel from 10Hz traces
        self.emg_rms_mean_by_epoch = self._epoch_mean_from_timeseries(
            self.emg_rms_10hz_t, self.emg_rms_10hz, self.epoch_t
        )
        self.wheel_speed_mean_by_epoch = self._epoch_mean_from_timeseries(
            self.wheel_10hz_t, self.wheel_10hz, self.epoch_t
        )

        theta_power = np.asarray(self.theta_power, dtype=float)
        delta_power = np.asarray(self.delta_power, dtype=float)

        ratio = theta_power / delta_power
        ratio[~np.isfinite(ratio)] = np.nan

        self.theta_delta_ratio_smoothed = self._adaptive_savgol(ratio, base_window=11)
        self.delta_power_smoothed = self._adaptive_savgol(delta_power, base_window=11)

        self.epoch_features_data = build_epoch_feature_dict(
            self.epoch_t,
            theta_power,
            delta_power,
            self.emg_rms_mean_by_epoch,
            self.wheel_speed_mean_by_epoch,
            theta_delta_ratio=ratio,
            theta_delta_ratio_smoothed=self.theta_delta_ratio_smoothed,
            delta_power_smoothed=self.delta_power_smoothed,
        )

    # -----------------------------
    # Manual state assignment (no functionality loss)
    # -----------------------------
    def assign_state_key(self, state_value: int):
        if not self.loaded or self.selection_range is None:
            return
        tmin, tmax = self.selection_range

        mask_epoch = (self.state_epoch_t >= tmin) & (self.state_epoch_t <= tmax)
        self.state_epoch[mask_epoch] = int(state_value)

        mask_10 = (self.state_10hz_t >= tmin) & (self.state_10hz_t <= tmax)
        self.state_10hz[mask_10] = int(state_value)

        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
            self.selection_patch = None
        self.selection_range = None

        # Only update state plot (avoid full rebuild)
        self._update_state_axis_only()

    def rerun_classification(self):
        """
        Recompute state_epoch and state_10hz, then update only the state axis.
        """
        if not self.loaded:
            return
        if self.epoch_features_data is None:
            self._rebuild_epoch_features()
        thresholds = {
            "emg_rms_threshold": float(self.emg_rms_threshold),
            "wheel_speed_threshold": float(self.wheel_speed_threshold),
            "theta_delta_ratio_threshold": float(self.theta_delta_ratio_threshold),
            "delta_power_threshold": float(self.delta_power_threshold),
        }
        try:
            state_epoch, resolved = score_from_epoch_features(
                self.epoch_features_data,
                thresholds=thresholds,
                auto_thresholds=False,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Rescore error", f"Error during rescoring:\n{exc}")
            return

        self.emg_rms_threshold = float(resolved["emg_rms_threshold"])
        self.wheel_speed_threshold = float(resolved["wheel_speed_threshold"])
        self.theta_delta_ratio_threshold = float(resolved["theta_delta_ratio_threshold"])
        self.delta_power_threshold = float(resolved["delta_power_threshold"])

        self.state_epoch = state_epoch.astype(np.int8)

        interp = interp1d(
            self.epoch_t,
            self.state_epoch,
            kind="nearest",
            bounds_error=False,
            fill_value=(self.state_epoch[0], self.state_epoch[-1]),
        )
        self.state_10hz = interp(self.state_10hz_t).astype(np.int8)

        self._state_dirty = False
        self._sync_threshold_widgets_to_values()
        self._update_state_axis_only()

    def onRescoreClicked(self):
        if not self.loaded:
            return
        self.rerun_classification()

    # -----------------------------
    # Threshold widgets (drag + slider sync)
    # -----------------------------
    def _threshold_edit_enabled(self):
        # Keep toolbar functionality but avoid fighting it:
        # - threshold edit mode must be on
        # - toolbar must not be in pan/zoom mode (best-effort)
        if not self.thresholdEditCheck.isChecked():
            return False
        try:
            mode = getattr(self.toolbar, "mode", "")
            if isinstance(mode, str) and mode.strip():
                return False
        except Exception:
            pass
        return True

    def _auto_rescore_enabled(self):
        return (
            self.autoRescoreCheck.isEnabled()
            and self.autoRescoreCheck.isChecked()
        )

    def _handle_threshold_change(self):
        self._state_dirty = True
        if self._auto_rescore_enabled():
            self.rerun_classification()

    def _sync_threshold_widgets_to_values(self):
        if (
            self._thr_drag_emg is None
            and self._thr_drag_wheel is None
            and self._thr_drag_ratio is None
        ):
            return
        self._syncing_slider = True
        try:
            if self._thr_drag_emg is not None:
                self._thr_drag_emg.set_y(float(self.emg_rms_threshold), trigger_callback=False)
            if self._slider_emg is not None:
                self._slider_emg.set_val(float(self.emg_rms_threshold))

            if self._thr_drag_wheel is not None:
                self._thr_drag_wheel.set_y(float(self.wheel_speed_threshold), trigger_callback=False)
            if self._slider_wheel is not None:
                self._slider_wheel.set_val(float(self.wheel_speed_threshold))

            if self._thr_drag_ratio is not None:
                self._thr_drag_ratio.set_y(float(self.theta_delta_ratio_threshold), trigger_callback=False)
            if self._slider_ratio is not None:
                self._slider_ratio.set_val(float(self.theta_delta_ratio_threshold))
        except Exception:
            pass
        finally:
            self._syncing_slider = False

    def _disconnect_threshold_widgets(self):
        for obj in [self._thr_drag_emg, self._thr_drag_wheel, self._thr_drag_ratio]:
            if obj is not None:
                try:
                    obj.disconnect()
                except Exception:
                    pass
        self._thr_drag_emg = None
        self._thr_drag_wheel = None
        self._thr_drag_ratio = None
        self._slider_emg = None
        self._slider_wheel = None
        self._slider_ratio = None

    def _setup_threshold_widgets(self):
        """
        Create draggable lines + sliders AFTER axes exist.
        Sliders live in reserved bottom space.
        """
        self._disconnect_threshold_widgets()

        if self._ax_emg is None or self._ax_ratio is None or self._ax_wheel is None:
            return

        fig = self.figure

        # Reserve space at bottom for sliders
        fig.subplots_adjust(bottom=0.16, top=0.98, left=0.07, right=0.98, hspace=0.12)

        # Slider axes (normalized figure coords)
        ax_s_emg = fig.add_axes([0.08, 0.06, 0.28, 0.03])
        ax_s_wheel = fig.add_axes([0.40, 0.06, 0.28, 0.03])
        ax_s_ratio = fig.add_axes([0.72, 0.06, 0.20, 0.03])

        # Data-driven slider ranges
        emg_vals = self.emg_rms_mean_by_epoch
        if emg_vals is None:
            emg_vals = np.asarray(self.emg_rms_10hz, dtype=float)
        emg_finite = emg_vals[np.isfinite(emg_vals)]
        if emg_finite.size > 0:
            emg_min = float(np.percentile(emg_finite, 1))
            emg_max = float(np.percentile(emg_finite, 99))
        else:
            emg_min, emg_max = 0.0, max(1.0, float(self.emg_rms_threshold) * 2.0)
        if emg_min == emg_max:
            emg_max = emg_min + 1e-6

        wheel_vals = self.wheel_speed_mean_by_epoch
        if wheel_vals is None:
            wheel_vals = np.asarray(self.wheel_10hz, dtype=float)
        wheel_finite = wheel_vals[np.isfinite(wheel_vals)]
        wheel_max = float(np.percentile(np.abs(wheel_finite), 99)) if wheel_finite.size else 5.0
        wheel_max = max(wheel_max, 0.1)

        ratio_vals = np.asarray(self.theta_delta_ratio_smoothed, dtype=float)
        ratio_finite = ratio_vals[np.isfinite(ratio_vals)]
        if ratio_finite.size > 0:
            ratio_min = float(np.percentile(ratio_finite, 1))
            ratio_max = float(np.percentile(ratio_finite, 99))
        else:
            ratio_min, ratio_max = 0.0, 5.0
        ratio_min = min(ratio_min, float(self.theta_delta_ratio_threshold) * 0.5)
        ratio_max = max(ratio_max, float(self.theta_delta_ratio_threshold) * 1.5)
        if ratio_min == ratio_max:
            ratio_max = ratio_min + 1e-6

        # --- Drag callbacks (update slider + rerun classification) ---
        def on_emg_drag(y):
            if self._syncing_slider:
                return
            self.emg_rms_threshold = float(y)
            if self._slider_emg is not None:
                self._syncing_slider = True
                try:
                    self._slider_emg.set_val(float(y))
                finally:
                    self._syncing_slider = False
            self._handle_threshold_change()

        def on_wheel_drag(y):
            if self._syncing_slider:
                return
            self.wheel_speed_threshold = float(y)
            if self._slider_wheel is not None:
                self._syncing_slider = True
                try:
                    self._slider_wheel.set_val(float(y))
                finally:
                    self._syncing_slider = False
            self._handle_threshold_change()

        def on_ratio_drag(y):
            if self._syncing_slider:
                return
            self.theta_delta_ratio_threshold = float(y)
            if self._slider_ratio is not None:
                self._syncing_slider = True
                try:
                    self._slider_ratio.set_val(float(y))
                finally:
                    self._syncing_slider = False
            self._handle_threshold_change()

        # Create draggable lines (each owns its line)
        self._thr_drag_emg = DraggableHLine(
            self._ax_emg,
            float(self.emg_rms_threshold),
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            tolerance_px=7,
            on_changed=on_emg_drag,
            is_enabled_fn=self._threshold_edit_enabled,
        )

        self._thr_drag_wheel = DraggableHLine(
            self._ax_wheel,
            float(self.wheel_speed_threshold),
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            tolerance_px=7,
            on_changed=on_wheel_drag,
            is_enabled_fn=self._threshold_edit_enabled,
        )

        self._thr_drag_ratio = DraggableHLine(
            self._ax_ratio,
            float(self.theta_delta_ratio_threshold),
            color="r",
            linestyle="--",
            linewidth=1.5,
            tolerance_px=7,
            on_changed=on_ratio_drag,
            is_enabled_fn=self._threshold_edit_enabled,
        )

        # Slider callbacks (update draggable line; draggable callback does rerun)
        def on_emg_slider(val):
            if self._syncing_slider:
                return
            self._syncing_slider = True
            try:
                self._thr_drag_emg.set_y(float(val), trigger_callback=True)
            finally:
                self._syncing_slider = False

        def on_wheel_slider(val):
            if self._syncing_slider:
                return
            self._syncing_slider = True
            try:
                self._thr_drag_wheel.set_y(float(val), trigger_callback=True)
            finally:
                self._syncing_slider = False

        def on_ratio_slider(val):
            if self._syncing_slider:
                return
            self._syncing_slider = True
            try:
                self._thr_drag_ratio.set_y(float(val), trigger_callback=True)
            finally:
                self._syncing_slider = False

        # Create sliders
        self._slider_emg = Slider(ax_s_emg, "EMG thr", emg_min, emg_max, valinit=float(self.emg_rms_threshold))
        self._slider_wheel = Slider(ax_s_wheel, "Wheel thr", 0.0, wheel_max, valinit=float(self.wheel_speed_threshold))
        self._slider_ratio = Slider(ax_s_ratio, "θ/δ thr", ratio_min, ratio_max, valinit=float(self.theta_delta_ratio_threshold))

        self._slider_emg.on_changed(on_emg_slider)
        self._slider_wheel.on_changed(on_wheel_slider)
        self._slider_ratio.on_changed(on_ratio_slider)

    # -----------------------------
    # Plotting
    # -----------------------------
    def plotTraces(self):
        if not hasattr(self, "emg_rms_10hz"):
            return

        # saturation percentage
        try:
            sat_pct = float(self.satPercentEdit.text())
        except Exception:
            sat_pct = 90.0
        sat_pct = min(max(sat_pct, 1.0), 99.0)

        # masking (existing behaviour)
        epoch_moving = self._compute_epoch_motion_mask()
        theta_masked = self.theta_power.copy()
        delta_masked = self.delta_power.copy()
        theta_masked[epoch_moving] = np.nan
        delta_masked[epoch_moving] = np.nan

        ratio_masked = theta_masked / delta_masked
        ratio_masked[~np.isfinite(ratio_masked)] = np.nan

        theta_z = self._zscore_safe(theta_masked)
        delta_z = self._zscore_safe(delta_masked)

        # delta power threshold in z units (for display only)
        delta_thr_z = None
        if self.delta_power_threshold is not None:
            mu_delta = np.nanmean(delta_masked)
            sigma_delta = np.nanstd(delta_masked)
            if sigma_delta > 0 and np.isfinite(sigma_delta):
                delta_thr_z = (float(self.delta_power_threshold) - mu_delta) / sigma_delta

        # Full rebuild
        self.figure.clear()

        axs = self.figure.subplots(
            6, 1, sharex=True,
            gridspec_kw={'height_ratios': [1, 1, 3, 1, 1, 1]}
        )
        self._axs = axs

        # Colors
        emg_color = 'tab:blue'
        wheel_color = 'tab:orange'

        # 1) EMG RMS + wheel (twin axes)
        ax1 = axs[0]
        ax1.plot(self.emg_rms_10hz_t, self.emg_rms_10hz, color=emg_color, label="EMG RMS")
        ax1.set_ylabel("EMG RMS", color=emg_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='y', labelcolor=emg_color)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.spines['left'].set_color(emg_color)

        ax1b = ax1.twinx()
        ax1b.plot(self.wheel_10hz_t, self.wheel_10hz, color=wheel_color, label="Wheel")
        ax1b.set_ylabel("Wheel", color=wheel_color)
        ax1b.spines['top'].set_visible(False)
        ax1b.tick_params(axis='y', labelcolor=wheel_color)
        ax1b.spines['right'].set_color(wheel_color)

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

        for y in [1.0, 4.0, 8.0]:
            ax3.axhline(y=y, color='white', linestyle=':', linewidth=1)

        ax3.set_ylabel("Freq (Hz)")
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(axis='x', labelbottom=False)

        # 4) Theta + Delta (z-scored)
        ax4 = axs[3]
        ax4.plot(self.epoch_t, theta_z, label="Theta (z)")
        ax4.plot(self.epoch_t, delta_z, label="Delta (z)")
        ax4.set_ylabel("Power (z)")
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(axis='x', labelbottom=False)
        ax4.legend(loc="upper right")
        if delta_thr_z is not None:
            ax4.axhline(delta_thr_z, color='r', linestyle='--', linewidth=1)

        # 5) Ratio plot (label kept as Δ/Θ, but value is theta/delta as in your current GUI)
        ax5 = axs[4]
        ax5.plot(self.epoch_t, ratio_masked)
        ax5.set_ylabel("Δ/Θ")
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.tick_params(axis='x', labelbottom=False)

        # 6) State axis (epoch-level)
        ax6 = axs[5]
        ax6.set_ylabel("State")
        ax6.set_xlabel("Time (s)")
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)

        self._ax_emg = ax1
        self._ax_wheel = ax1b
        self._ax_ratio = ax5
        self._ax_state = ax6

        # Draw state once (from current arrays)
        self._draw_state_axis(ax6)

        # Vertical lines
        self.vlines = []
        current_time = self._current_time()
        for ax in axs:
            vline = ax.axvline(x=current_time, color='k', linestyle='--')
            self.vlines.append(vline)

        self._setup_span_selectors()

        # Threshold widgets (draggable lines + sliders)
        self._setup_threshold_widgets()

        self.canvas.draw()

    def _draw_state_axis(self, ax6):
        ax6.cla()
        ax6.set_ylabel("State")
        ax6.set_xlabel("Time (s)")
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)

        t = np.asarray(self.state_epoch_t, dtype=float)
        s = np.asarray(self.state_epoch, dtype=int)

        color_map = {0: 'k', 1: '0.5', 2: 'tab:blue', 3: 'tab:green'}

        for state_val, col in color_map.items():
            mask = (s == state_val)
            y = np.where(mask, s, np.nan)
            ax6.step(t, y, where="post", color=col, linewidth=1.5)

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

    def _update_state_axis_only(self):
        if self._ax_state is None:
            return

        # remove any selection highlight safely
        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
            self.selection_patch = None

        self._draw_state_axis(self._ax_state)

        # restore vertical line in state axis and keep others untouched
        current_time = self._current_time()
        try:
            self._ax_state.axvline(x=current_time, color='k', linestyle='--')
        except Exception:
            pass

        self.canvas.draw_idle()

    # -----------------------------
    # Main
    # -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoAnalysisApp()
    win.showMaximized()
    sys.exit(app.exec_())
