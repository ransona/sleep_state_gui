import sys
import os
import pickle
import shutil
import time
import cv2
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QMessageBox, QSizePolicy,
    QCheckBox, QRadioButton, QButtonGroup, QProgressBar, QDoubleSpinBox,
    QInputDialog,
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
    DEFAULT_DELTA_BAND,
    DEFAULT_THETA_BAND,
    DEFAULT_USER_ID,
    DEFAULT_EXP_ID,
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


class DraggableVLine:
    """
    Robust draggable vertical line using explicit proximity detection.
    - Creates axvline internally for every supplied axis.
    - Binds to the shared figure canvas.
    - No picking, no widgetlock, no toolbar logic.
    """

    def __init__(
        self,
        axes,
        x0,
        color="r",
        linestyle="--",
        linewidth=1.5,
        tolerance_px=6,
        on_changed=None,
        is_enabled_fn=None,
    ):
        self.axes = list(axes) if axes else []
        self.canvas = None
        if self.axes:
            self.canvas = self.axes[0].figure.canvas
        self.on_changed = on_changed
        self.tolerance_px = tolerance_px
        self.is_enabled_fn = is_enabled_fn

        self.lines = []
        for ax in self.axes:
            line = ax.axvline(
                x0,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=10,
            )
            self.lines.append(line)

        self._dragging = False

        self.cid_press = None
        self.cid_motion = None
        self.cid_release = None
        if self.canvas is not None:
            self.cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
            self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)

    def disconnect(self):
        for cid in [self.cid_press, self.cid_motion, self.cid_release]:
            try:
                if cid is not None:
                    self.canvas.mpl_disconnect(cid)
            except Exception:
                pass

    def get_x(self):
        if not self.lines:
            return 0.0
        return float(self.lines[0].get_xdata()[0])

    def set_x(self, x, trigger_callback=True):
        for line in self.lines:
            line.set_xdata([x, x])
        if trigger_callback and self.on_changed is not None:
            self.on_changed(float(x))
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _enabled(self):
        if self.is_enabled_fn is None:
            return True
        try:
            return bool(self.is_enabled_fn())
        except Exception:
            return True

    def _is_near_line(self, event):
        if event.inaxes not in self.axes:
            return False
        if event.xdata is None:
            return False

        x_line = self.get_x()
        trans = event.inaxes.transData
        p_event = trans.transform((event.xdata, 0.0))
        p_line = trans.transform((x_line, 0.0))
        dx = abs(p_event[0] - p_line[0])
        return dx <= self.tolerance_px

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
        if event.inaxes not in self.axes:
            return
        if event.xdata is None:
            return
        self.set_x(event.xdata, trigger_callback=True)

    def _on_release(self, event):
        self._dragging = False


class ScoringProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sleep scoring")
        self.setModal(True)
        layout = QVBoxLayout()
        self.statusLabel = QLabel("Starting…")
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.progressBar)
        self.setLayout(layout)
        self.resize(360, 120)

    def update_progress(self, fraction, message):
        try:
            frac = float(fraction)
        except Exception:
            frac = 0.0
        frac = min(max(frac, 0.0), 1.0)
        self.statusLabel.setText(str(message))
        self.progressBar.setValue(int(round(frac * 100)))
        QApplication.processEvents()

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
        self.selection_patches = []
        self.span_selectors = []
        self._selection_line_start = None
        self._selection_line_end = None
        self._selection_patch_axis_index = None

        # state label map
        self.state_label_map = None

        # thresholds
        self.emg_rms_threshold = None
        self.wheel_speed_threshold = None
        self.theta_delta_ratio_threshold = None
        self.delta_power_threshold = None

        # eye frame times
        self.eye_frame_times = None
        self.face_motion_10hz = None
        self.face_motion_10hz_t = None
        self.show_eye_videos = True
        self.show_pupil_overlay = True

        # cached epoch features for rescoring
        self.epoch_features_data = None
        self.loaded_sleep_state = None
        self.sleep_state_path = None
        self.left_video_cap = None
        self.right_video_cap = None
        self.debug_frame_timing = True
        self.motion_smoothing_secs = 0.5
        self._state_dirty = False
        self.available_state_values = set()
        self.delta_band = DEFAULT_DELTA_BAND
        self.theta_band = DEFAULT_THETA_BAND
        self.simulation_enabled = False
        self.simulated_sample_name = None
        self.simulated_sample_path = None
        self.sample_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

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
        self.userIdEdit.setText(DEFAULT_USER_ID)
        self.expIdEdit = QLineEdit()
        self.expIdEdit.setPlaceholderText("Enter Experiment ID")
        self.expIdEdit.setText(DEFAULT_EXP_ID)

        self.loadButton = QPushButton("Load Data")
        self.loadButton.clicked.connect(self.loadData)

        self.scoreButton = QPushButton("Run Sleep Scoring")
        self.scoreButton.clicked.connect(self.runSleepScoringClicked)

        self.simulationCheck = QCheckBox("Simulated EEG/EMG")
        self.simulationCheck.toggled.connect(self.onSimModeToggled)
        self.simulationLabel = QLabel("")
        self.simulationLabel.setFixedWidth(140)
        self.simulationLabel.setStyleSheet("color: gray;")

        inputLayout.addWidget(QLabel("User ID:"))
        inputLayout.addWidget(self.userIdEdit)
        inputLayout.addWidget(QLabel("Experiment ID:"))
        inputLayout.addWidget(self.expIdEdit)
        inputLayout.addWidget(self.loadButton)
        inputLayout.addWidget(self.scoreButton)
        inputLayout.addWidget(self.simulationCheck)
        inputLayout.addWidget(self.simulationLabel)
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
        self.leftVideoLabel.setFixedSize(320, 120)
        self.rightVideoLabel = QLabel("Right Eye Video")
        self.rightVideoLabel.setFixedSize(320, 120)
        videoLayout.addWidget(self.leftVideoLabel)
        videoLayout.addWidget(self.rightVideoLabel)
        mainLayout.addLayout(videoLayout)

        videoOptionsLayout = QHBoxLayout()
        self.showVideosCheck = QCheckBox("Show eye videos")
        self.showVideosCheck.setChecked(True)
        self.showVideosCheck.setEnabled(False)
        self.showVideosCheck.toggled.connect(self.onShowVideosToggled)
        videoOptionsLayout.addWidget(self.showVideosCheck)

        self.showOverlayCheck = QCheckBox("Show pupil overlay")
        self.showOverlayCheck.setChecked(True)
        self.showOverlayCheck.setEnabled(False)
        self.showOverlayCheck.toggled.connect(self.onOverlayToggled)
        videoOptionsLayout.addWidget(self.showOverlayCheck)
        videoOptionsLayout.addSpacing(20)
        videoOptionsLayout.addWidget(QLabel("Motion smoothing (s):"))
        self.motionSmoothSpin = QDoubleSpinBox()
        self.motionSmoothSpin.setRange(0.0, 5.0)
        self.motionSmoothSpin.setSingleStep(0.1)
        self.motionSmoothSpin.setValue(self.motion_smoothing_secs)
        self.motionSmoothSpin.setFixedWidth(100)
        self.motionSmoothSpin.valueChanged.connect(self.onMotionSmoothingChanged)
        videoOptionsLayout.addWidget(self.motionSmoothSpin)
        videoOptionsLayout.addStretch(1)
        mainLayout.addLayout(videoOptionsLayout)

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
        self.modeThresholdButton = QRadioButton("Threshold mode")
        self.modeSelectionButton = QRadioButton("Selection mode")
        self.modeGroup = QButtonGroup(self)
        self.modeGroup.addButton(self.modeThresholdButton)
        self.modeGroup.addButton(self.modeSelectionButton)
        self.modeSelectionButton.setChecked(True)
        self.modeThresholdButton.setEnabled(False)
        self.modeSelectionButton.setEnabled(False)
        self.modeGroup.buttonToggled.connect(self.onModeToggled)
        satLayout.addWidget(self.modeThresholdButton)
        satLayout.addWidget(self.modeSelectionButton)

        satLayout.addSpacing(12)
        self.autoRescoreCheck = QCheckBox("Auto rescore")
        self.autoRescoreCheck.setEnabled(False)
        self.autoRescoreCheck.setChecked(False)
        satLayout.addWidget(self.autoRescoreCheck)

        self.rescoreButton = QPushButton("Rescore")
        self.rescoreButton.setEnabled(False)
        self.rescoreButton.clicked.connect(self.onRescoreClicked)
        satLayout.addWidget(self.rescoreButton)

        self.saveButton = QPushButton("Save scoring")
        self.saveButton.setEnabled(False)
        self.saveButton.clicked.connect(self.saveSleepScoring)
        satLayout.addWidget(self.saveButton)

        satLayout.addStretch(1)
        mainLayout.addLayout(satLayout)

        # --- Delta/Theta band controls ---
        bandLayout = QHBoxLayout()
        bandLayout.addWidget(QLabel("Delta band (Hz):"))
        self.deltaBandLowEdit = QLineEdit(f"{DEFAULT_DELTA_BAND[0]:.1f}")
        self.deltaBandLowEdit.setFixedWidth(60)
        self.deltaBandLowEdit.setEnabled(False)
        self.deltaBandHighEdit = QLineEdit(f"{DEFAULT_DELTA_BAND[1]:.1f}")
        self.deltaBandHighEdit.setFixedWidth(60)
        self.deltaBandHighEdit.setEnabled(False)
        bandLayout.addWidget(self.deltaBandLowEdit)
        bandLayout.addWidget(QLabel("to"))
        bandLayout.addWidget(self.deltaBandHighEdit)

        bandLayout.addSpacing(20)
        bandLayout.addWidget(QLabel("Theta band (Hz):"))
        self.thetaBandLowEdit = QLineEdit(f"{DEFAULT_THETA_BAND[0]:.1f}")
        self.thetaBandLowEdit.setFixedWidth(60)
        self.thetaBandLowEdit.setEnabled(False)
        self.thetaBandHighEdit = QLineEdit(f"{DEFAULT_THETA_BAND[1]:.1f}")
        self.thetaBandHighEdit.setFixedWidth(60)
        self.thetaBandHighEdit.setEnabled(False)
        bandLayout.addWidget(self.thetaBandLowEdit)
        bandLayout.addWidget(QLabel("to"))
        bandLayout.addWidget(self.thetaBandHighEdit)
        bandLayout.addStretch(1)
        mainLayout.addLayout(bandLayout)

        # --- State assignment controls (buttons + shortcuts) ---
        stateAssignLayout = QHBoxLayout()
        stateAssignLayout.addWidget(QLabel("Assign state:"))
        self.stateButtonsLayout = QHBoxLayout()
        stateAssignLayout.addLayout(self.stateButtonsLayout)
        stateAssignLayout.addStretch(1)
        mainLayout.addLayout(stateAssignLayout)
        self.state_buttons = []

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
        self.state_shortcut_map = {}

    def onSimModeToggled(self, checked):
        checked = bool(checked)
        if checked:
            selection = self._prompt_for_simulation_sample()
            if selection is None:
                self.simulationCheck.blockSignals(True)
                self.simulationCheck.setChecked(False)
                self.simulationCheck.blockSignals(False)
                self.simulation_enabled = False
            else:
                self.simulation_enabled = True
        else:
            self.simulation_enabled = False
        self._update_simulation_label()

    def _prompt_for_simulation_sample(self):
        options = self._list_simulation_files()
        if not options:
            QMessageBox.warning(self, "Simulation data", "No sample_data files were found.")
            return None
        default_index = 0
        if self.simulated_sample_name in options:
            default_index = options.index(self.simulated_sample_name)
        selection, ok = QInputDialog.getItem(
            self,
            "Select simulated EEG/EMG",
            "Dataset:",
            options,
            default_index,
            False,
        )
        if not ok:
            return None
        path = os.path.join(self.sample_data_dir, selection)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Simulation data", f"Selected dataset is not available: {selection}")
            return None
        self.simulated_sample_name = selection
        self.simulated_sample_path = path
        return selection

    def _list_simulation_files(self):
        try:
            files = sorted(
                entry
                for entry in os.listdir(self.sample_data_dir)
                if entry.lower().endswith(".npz")
            )
        except Exception:
            files = []
        return files

    def _update_simulation_label(self):
        if self.simulation_enabled and self.simulated_sample_name:
            self.simulationLabel.setText(self.simulated_sample_name)
            self.simulationLabel.setToolTip("Simulated EEG/EMG data will be used for scoring.")
        else:
            self.simulationLabel.setText("")
            self.simulationLabel.setToolTip("")

    def _simulation_suffix(self):
        return "_sim" if self.simulation_enabled else ""

    def _simulation_npz_path(self):
        if self.simulation_enabled and self.simulated_sample_path:
            return self.simulated_sample_path
        return None

    def _sleep_state_basename(self):
        return f"sleep_state{self._simulation_suffix()}.pickle"

    def _sleep_state_path(self):
        processed_root = getattr(self, "exp_dir_processed", None)
        if not processed_root:
            return None
        return os.path.join(processed_root, "sleep_score", self._sleep_state_basename())

    # -----------------------------
    # Load data
    # -----------------------------
    def loadData(self):
        self._close_eye_video_caps()
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

        if not self._open_eye_video_caps():
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
        sleep_state_name = self._sleep_state_basename()
        sleep_state_path = os.path.join(self.exp_dir_processed, "sleep_score", sleep_state_name)
        sleep_state = None

        if os.path.isfile(sleep_state_path):
            try:
                with open(sleep_state_path, "rb") as f:
                    sleep_state = pickle.load(f)
            except Exception as e:
                reply = QMessageBox.question(
                    self,
                    "Sleep scoring error",
                    f"Error loading {sleep_state_name}:\n{e}\n\nDo you want to rerun sleep scoring now?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    bands = self._parse_band_entries()
                    if bands is None:
                        return
                    delta_band_vals, theta_band_vals = bands
                    sleep_state = self._run_sleep_scoring_for_current_ids(
                        delta_band=delta_band_vals,
                        theta_band=theta_band_vals,
                    )
                    if sleep_state is None:
                        return
                else:
                    QMessageBox.information(self, "Sleep scoring", "Load cancelled: sleep scoring data not available.")
                    return
        else:
            reply = QMessageBox.question(
                self,
                "Sleep scoring not found",
                f"{sleep_state_name} was not found.\nDo you want to run sleep scoring now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                bands = self._parse_band_entries()
                if bands is None:
                    return
                delta_band_vals, theta_band_vals = bands
                sleep_state = self._run_sleep_scoring_for_current_ids(
                    delta_band=delta_band_vals,
                    theta_band=theta_band_vals,
                )
                if sleep_state is None:
                    return
            else:
                QMessageBox.information(self, "Sleep scoring", "Load cancelled: sleep scoring data not available.")
                return

        ok, err_msg = self._validate_sleep_state_dict(sleep_state, sleep_state_name)
        if not ok:
            reply = QMessageBox.question(
                self,
                "Sleep scoring file outdated",
                f"{err_msg}\nThis experiment needs to be rescored.\nRun sleep scoring now?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                bands = self._parse_band_entries()
                if bands is None:
                    return
                delta_band_vals, theta_band_vals = bands
                sleep_state = self._run_sleep_scoring_for_current_ids(
                    delta_band=delta_band_vals,
                    theta_band=theta_band_vals,
                )
                if sleep_state is None:
                    return
                ok, err_msg = self._validate_sleep_state_dict(sleep_state, sleep_state_name)
                if not ok:
                    QMessageBox.critical(
                        self,
                        "Sleep scoring error",
                        f"Rescoring did not produce a valid data file:\n{err_msg}",
                    )
                    return
            else:
                QMessageBox.information(self, "Sleep scoring", "Load cancelled: incompatible data.")
                return

        self.sleep_state_path = sleep_state_path

        self._populate_from_sleep_state(sleep_state, sleep_state_path)

    def _validate_sleep_state_dict(self, sleep_state, file_label=None):
        required_keys = [
            "emg_rms_10hz",
            "emg_rms_10hz_t",
            "wheel_10hz",
            "wheel_10hz_t",
            "face_motion_10hz",
            "face_motion_10hz_t",
            "eeg_10hz",
            "eeg_10hz_t",
            "state_epoch",
            "state_epoch_t",
            "state_10hz",
            "state_10hz_t",
            "epoch_t",
            "theta_power",
            "delta_power",
            "epoch_features",
        ]
        label = file_label or "sleep_state data"
        for key in required_keys:
            if key not in sleep_state:
                return False, f"{label} is missing '{key}'."

        epoch_features = sleep_state.get("epoch_features")
        if not isinstance(epoch_features, dict) or not epoch_features:
            return False, f"{label} has no cached epoch features."

        return True, ""

    def _run_sleep_scoring_for_current_ids(self, delta_band=None, theta_band=None):
        progress_dialog = ScoringProgressDialog(self)
        progress_dialog.show()
        QApplication.processEvents()

        def _progress_cb(fraction, message):
            progress_dialog.update_progress(fraction, message)

        try:
            sleep_state = run_sleep_scoring(
                self.userID,
                self.expID,
                delta_band=delta_band,
                theta_band=theta_band,
                progress_callback=_progress_cb,
                simulated_npz=self._simulation_npz_path(),
                filename_suffix=self._simulation_suffix(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Sleep scoring error", f"Error running sleep scoring:\n{e}")
            return None
        finally:
            progress_dialog.close()
        if sleep_state is None:
            QMessageBox.critical(self, "Sleep scoring error", "run_sleep_scoring returned no data.")
            return None
        return sleep_state

    def _populate_from_sleep_state(self, sleep_state, sleep_state_path):
        self.sleep_state_path = sleep_state_path
        self.loaded_sleep_state = sleep_state

        # 10 Hz traces
        self.emg_rms_10hz = np.asarray(sleep_state['emg_rms_10hz'], dtype=float)
        self.emg_rms_10hz_t = np.asarray(sleep_state['emg_rms_10hz_t'], dtype=float)

        self.wheel_10hz = np.asarray(sleep_state['wheel_10hz'], dtype=float)
        self.wheel_10hz_t = np.asarray(sleep_state['wheel_10hz_t'], dtype=float)
        self.face_motion_10hz = np.asarray(sleep_state['face_motion_10hz'], dtype=float)
        self.face_motion_10hz_t = np.asarray(sleep_state['face_motion_10hz_t'], dtype=float)

        self.eeg_10hz = np.asarray(sleep_state['eeg_10hz'], dtype=float)
        self.eeg_10hz_t = np.asarray(sleep_state['eeg_10hz_t'], dtype=float)

        # epoch-level
        self.epoch_t = np.asarray(sleep_state['epoch_t'], dtype=float)
        self.theta_power = np.asarray(sleep_state['theta_power'], dtype=float)
        self.delta_power = np.asarray(sleep_state['delta_power'], dtype=float)

        # spectrogram
        self.eeg_spectrogram = np.asarray(sleep_state['eeg_spectrogram'], dtype=float)
        self.eeg_spectrogram_freqs = np.asarray(sleep_state['eeg_spectrogram_freqs'], dtype=float)
        self.eeg_spectrogram_t = np.asarray(sleep_state['eeg_spectrogram_t'], dtype=float)

        # state
        self.state_epoch = np.asarray(sleep_state['state_epoch'], dtype=int)
        self.state_epoch_t = np.asarray(sleep_state['state_epoch_t'], dtype=float)
        self.state_10hz = np.asarray(sleep_state['state_10hz'], dtype=int)
        self.state_10hz_t = np.asarray(sleep_state['state_10hz_t'], dtype=float)

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

        # band metadata
        delta_band = sleep_state.get("delta_band", DEFAULT_DELTA_BAND)
        theta_band = sleep_state.get("theta_band", DEFAULT_THETA_BAND)
        try:
            self.delta_band = (float(delta_band[0]), float(delta_band[1]))
        except Exception:
            self.delta_band = DEFAULT_DELTA_BAND
        try:
            self.theta_band = (float(theta_band[0]), float(theta_band[1]))
        except Exception:
            self.theta_band = DEFAULT_THETA_BAND

        self.deltaBandLowEdit.setText(f"{self.delta_band[0]:.2f}")
        self.deltaBandHighEdit.setText(f"{self.delta_band[1]:.2f}")
        self.thetaBandLowEdit.setText(f"{self.theta_band[0]:.2f}")
        self.thetaBandHighEdit.setText(f"{self.theta_band[1]:.2f}")
        for widget in [
            self.deltaBandLowEdit,
            self.deltaBandHighEdit,
            self.thetaBandLowEdit,
            self.thetaBandHighEdit,
        ]:
            widget.setEnabled(True)

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
        self.modeThresholdButton.setEnabled(True)
        self.modeSelectionButton.setEnabled(True)
        self.modeSelectionButton.setChecked(True)
        self.autoRescoreCheck.setEnabled(True)
        self.autoRescoreCheck.setChecked(False)
        self.rescoreButton.setEnabled(True)
        self.saveButton.setEnabled(False)
        self.showVideosCheck.setEnabled(True)
        self.showOverlayCheck.setEnabled(True)
        self.showVideosCheck.blockSignals(True)
        self.showVideosCheck.setChecked(self.show_eye_videos)
        self.showVideosCheck.blockSignals(False)
        self.onShowVideosToggled(self.show_eye_videos)
        self.showOverlayCheck.blockSignals(True)
        self.showOverlayCheck.setChecked(self.show_pupil_overlay)
        self.showOverlayCheck.blockSignals(False)

        self.loaded = True
        self._state_dirty = False

        self._rebuild_epoch_features()
        self._refresh_state_controls()
        self._update_save_button_state()
        self.updateFrame()
        self.plotTraces()

    def _parse_band_entries(self):
        try:
            delta_low = float(self.deltaBandLowEdit.text())
            delta_high = float(self.deltaBandHighEdit.text())
            theta_low = float(self.thetaBandLowEdit.text())
            theta_high = float(self.thetaBandHighEdit.text())
        except Exception:
            QMessageBox.warning(self, "Band input error", "Delta/theta band values must be numeric.")
            return None

        if delta_low <= 0 or delta_high <= 0 or delta_high <= delta_low:
            QMessageBox.warning(self, "Band input error", "Delta band must be positive and low < high.")
            return None
        if theta_low <= 0 or theta_high <= 0 or theta_high <= theta_low:
            QMessageBox.warning(self, "Band input error", "Theta band must be positive and low < high.")
            return None

        return (delta_low, delta_high), (theta_low, theta_high)

    def _bands_changed(self, delta_band, theta_band):
        tol = 1e-6
        def diff(a, b):
            return abs(a - b) > tol
        if (
            diff(delta_band[0], self.delta_band[0])
            or diff(delta_band[1], self.delta_band[1])
            or diff(theta_band[0], self.theta_band[0])
            or diff(theta_band[1], self.theta_band[1])
        ):
            return True
        return False

    def _run_full_rescore_with_new_bands(self, delta_band, theta_band):
        sleep_state = self._run_sleep_scoring_for_current_ids(
            delta_band=delta_band,
            theta_band=theta_band,
        )
        if sleep_state is None:
            return
        sleep_state_path = self._sleep_state_path()
        self._populate_from_sleep_state(sleep_state, sleep_state_path)
        QMessageBox.information(
            self,
            "Sleep scoring",
            "Data reloaded with updated delta/theta bands.",
        )

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

        bands = self._parse_band_entries()
        if bands is None:
            return
        delta_band_vals, theta_band_vals = bands

        progress_dialog = ScoringProgressDialog(self)
        progress_dialog.show()
        QApplication.processEvents()

        success = False
        try:
            run_sleep_scoring(
                user_id,
                exp_id,
                delta_band=delta_band_vals,
                theta_band=theta_band_vals,
                progress_callback=progress_dialog.update_progress,
                simulated_npz=self._simulation_npz_path(),
                filename_suffix=self._simulation_suffix(),
            )
            success = True
        except Exception as e:
            QMessageBox.critical(self, "Sleep scoring error", f"Error running sleep scoring:\n{e}")
        finally:
            progress_dialog.close()

        if success:
            QMessageBox.information(
                self,
                "Sleep scoring",
                "Sleep scoring finished successfully.\nYou can now press 'Load Data'.",
            )

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

    def _move_cursor_to_time(self, time_value):
        if not self.loaded or self.total_frames <= 0:
            return
        if not hasattr(self, "slider") or self.slider is None:
            return
        try:
            target = float(time_value)
        except Exception:
            return
        idx = int(np.argmin(np.abs(self.emg_rms_10hz_t - target)))
        idx = max(0, min(idx, self.total_frames - 1))
        self.slider.setValue(idx)

    # -----------------------------
    # Video overlay
    # -----------------------------
    def overlay_plot(self, frame, position, eyeDat):
        if not self.show_pupil_overlay:
            return frame
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

    def playVideoFrame(self, frame_position, cap, eyedat):
        if cap is None or not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position))
        ret, frame = cap.read()
        if not ret or frame is None:
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

    def onShowVideosToggled(self, checked):
        self.show_eye_videos = bool(checked)
        if hasattr(self, "leftVideoLabel") and self.leftVideoLabel is not None:
            self.leftVideoLabel.setVisible(self.show_eye_videos)
            if not self.show_eye_videos:
                self.leftVideoLabel.clear()
        if hasattr(self, "rightVideoLabel") and self.rightVideoLabel is not None:
            self.rightVideoLabel.setVisible(self.show_eye_videos)
            if not self.show_eye_videos:
                self.rightVideoLabel.clear()
        if self.show_eye_videos and self.loaded:
            self.updateFrame()

    def onOverlayToggled(self, checked):
        self.show_pupil_overlay = bool(checked)
        if self.show_eye_videos and self.loaded:
            self.updateFrame()

    def onMotionSmoothingChanged(self, value):
        try:
            self.motion_smoothing_secs = float(value)
        except Exception:
            self.motion_smoothing_secs = 0.5
        if self.loaded:
            self.plotTraces()

    def updateFrame(self):
        if not self.loaded:
            return

        start_total = time.perf_counter()
        timings = []
        last = start_total

        t = self._current_time()
        now = time.perf_counter()
        timings.append(("time_lookup", now - last))
        last = now

        if self.show_eye_videos:
            if self.eye_frame_times is not None and self.eye_frame_times.size > 0:
                idx_frame = int(np.argmin(np.abs(self.eye_frame_times - t)))
            else:
                idx_frame = 0

            frame_left = self.playVideoFrame(
                idx_frame,
                self.left_video_cap,
                self.left_eyedat,
            )
            if frame_left is not None:
                image_left = QtGui.QImage(
                    frame_left.data, frame_left.shape[1], frame_left.shape[0],
                    frame_left.strides[0], QtGui.QImage.Format_RGB888
                )
                pixmap_left = QtGui.QPixmap.fromImage(image_left)
                self.leftVideoLabel.setPixmap(pixmap_left.scaled(self.leftVideoLabel.size(), Qt.KeepAspectRatio))

            frame_right = self.playVideoFrame(
                idx_frame,
                self.right_video_cap,
                self.right_eyedat,
            )
            if frame_right is not None:
                image_right = QtGui.QImage(
                    frame_right.data, frame_right.shape[1], frame_right.shape[0],
                    frame_right.strides[0], QtGui.QImage.Format_RGB888
                )
                pixmap_right = QtGui.QPixmap.fromImage(image_right)
                self.rightVideoLabel.setPixmap(pixmap_right.scaled(self.rightVideoLabel.size(), Qt.KeepAspectRatio))

            video_end = time.perf_counter()
            timings.append(("video_render", video_end - last))
            last = video_end
        else:
            if self.leftVideoLabel is not None:
                self.leftVideoLabel.clear()
            if self.rightVideoLabel is not None:
                self.rightVideoLabel.clear()
            video_end = time.perf_counter()
            timings.append(("video_hidden", video_end - last))
            last = video_end

        if self.vlines:
            for vline in self.vlines:
                try:
                    vline.set_xdata([t, t])
                except Exception:
                    pass
            self.canvas.draw_idle()
        vline_end = time.perf_counter()
        timings.append(("vline_update", vline_end - last))
        last = vline_end

        if self.frameJumpEdit.isEnabled():
            self.frameJumpEdit.setText(f"{t:.3f}")
        frame_text_end = time.perf_counter()
        timings.append(("frame_text", frame_text_end - last))
        last = frame_text_end

        total_elapsed = time.perf_counter() - start_total
        if self.debug_frame_timing:
            detail = " | ".join(f"{name}={duration*1000:.1f}ms" for name, duration in timings)
            print(f"[DEBUG] updateFrame times: {detail} | total={total_elapsed*1000:.1f}ms")

    def _close_eye_video_caps(self):
        for attr in ("left_video_cap", "right_video_cap"):
            cap = getattr(self, attr, None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _open_eye_video_caps(self):
        self._close_eye_video_caps()
        self.left_video_cap = cv2.VideoCapture(self.video_path_left)
        self.right_video_cap = cv2.VideoCapture(self.video_path_right)

        if (
            not self.left_video_cap.isOpened()
            or not self.right_video_cap.isOpened()
        ):
            QMessageBox.critical(
                self,
                "Video Error",
                "Unable to open one or both eye videos. Please verify the files.",
            )
            self._close_eye_video_caps()
            return False
        return True

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

    def closeEvent(self, event):
        self._close_eye_video_caps()
        super().closeEvent(event)

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
                if not self.loaded or not self._selection_enabled():
                    return
                self._set_selection_range(xmin, xmax, patch_ax=ax)
            return on_select

        selector_props = dict(alpha=0.0, facecolor='none', edgecolor='none', linewidth=0)

        for ax in self.figure.axes:
            selector = SpanSelector(
                ax,
                make_on_select(ax),
                direction='horizontal',
                useblit=False,
                interactive=False,
                props=selector_props,
            )
            selector.set_active(self._selection_enabled())
            self.span_selectors.append(selector)

    def _normalize_selection_bounds(self, value_a, value_b):
        try:
            a_val = float(value_a)
            b_val = float(value_b)
        except Exception:
            return None
        start = min(a_val, b_val)
        end = max(a_val, b_val)
        if not np.isfinite(start) and np.isfinite(end):
            start = end
        if not np.isfinite(end) and np.isfinite(start):
            end = start
        if not np.isfinite(start):
            return None
        start = self._snap_time_to_epoch_edge(start)
        end = self._snap_time_to_epoch_edge(end)
        if start > end:
            end = start
        return start, end

    def _snap_time_to_epoch_edge(self, value):
        try:
            t = float(value)
        except Exception:
            return 0.0
        epoch_centers = getattr(self, "epoch_t", None)
        if epoch_centers is None or epoch_centers.size == 0:
            return t
        edges = self._compute_epoch_edges(epoch_centers)
        finite_edges = edges[np.isfinite(edges)]
        if finite_edges.size == 0:
            return t
        idx = int(np.argmin(np.abs(finite_edges - t)))
        return float(finite_edges[idx])

    def _create_selection_patch(self, ax, start, end):
        if ax is None:
            return
        self._remove_selection_patch()
        try:
            patch = ax.axvspan(start, end, color='yellow', alpha=0.3)
        except Exception:
            return
        self.selection_patch = patch
        self.selection_patches = [patch]
        axis_index = None
        try:
            axis_index = list(self.figure.axes).index(ax)
        except Exception:
            axis_index = None
        self._selection_patch_axis_index = axis_index

    def _remove_selection_patch(self):
        if self.selection_patches:
            for patch in self.selection_patches:
                try:
                    patch.remove()
                except Exception:
                    pass
        self.selection_patches = []
        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
        self.selection_patch = None
        self._selection_patch_axis_index = None

    def _set_selection_range(self, start, end, patch_ax=None):
        bounds = self._normalize_selection_bounds(start, end)
        if bounds is None:
            return
        start, end = bounds
        self.selection_range = (start, end)
        target_ax = patch_ax
        axes = self.figure.axes
        if target_ax is None and self._selection_patch_axis_index is not None:
            if 0 <= self._selection_patch_axis_index < len(axes):
                target_ax = axes[self._selection_patch_axis_index]
        if target_ax is None and axes:
            target_ax = axes[0]
        if target_ax is not None:
            self._create_selection_patch(target_ax, start, end)
        self._sync_selection_lines_to_range()
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.draw_idle()
        self._move_cursor_to_time(start)

    def _sync_selection_lines_to_range(self):
        if self.selection_range is None:
            base = self._snap_time_to_epoch_edge(self._current_time())
            start = end = base
        else:
            start, end = self.selection_range
        if self._selection_line_start is not None:
            self._selection_line_start.set_x(start, trigger_callback=False)
        if self._selection_line_end is not None:
            self._selection_line_end.set_x(end, trigger_callback=False)

    def _on_selection_line_changed(self, is_start, x):
        if not np.isfinite(x):
            return
        if self.selection_range is None:
            base = self._snap_time_to_epoch_edge(self._current_time())
            current_start = current_end = base
        else:
            current_start, current_end = self.selection_range
        if is_start:
            start = self._snap_time_to_epoch_edge(x)
            end = max(current_end, start)
        else:
            end = self._snap_time_to_epoch_edge(x)
            start = min(current_start, end)
        patch_ax = None
        axes = self.figure.axes
        if self._selection_patch_axis_index is not None and 0 <= self._selection_patch_axis_index < len(axes):
            patch_ax = axes[self._selection_patch_axis_index]
        self._set_selection_range(start, end, patch_ax=patch_ax)

    def _setup_selection_vlines(self):
        self._remove_selection_vlines()
        axes = self.figure.axes
        if not axes:
            return
        if self.selection_range is not None:
            start, end = self.selection_range
        else:
            base = self._snap_time_to_epoch_edge(self._current_time())
            start = end = base
        self._selection_line_start = DraggableVLine(
            axes,
            start,
            color="cyan",
            linestyle="--",
            linewidth=1.5,
            tolerance_px=6,
            on_changed=lambda x: self._on_selection_line_changed(True, x),
            is_enabled_fn=self._selection_enabled,
        )
        self._selection_line_end = DraggableVLine(
            axes,
            end,
            color="magenta",
            linestyle="--",
            linewidth=1.5,
            tolerance_px=6,
            on_changed=lambda x: self._on_selection_line_changed(False, x),
            is_enabled_fn=self._selection_enabled,
        )
        self._sync_selection_lines_to_range()

    def _remove_selection_vlines(self):
        for vline in (self._selection_line_start, self._selection_line_end):
            if vline is not None:
                vline.disconnect()
        self._selection_line_start = None
        self._selection_line_end = None

    def _restore_selection_patch(self):
        if self.selection_range is None:
            return
        start, end = self.selection_range
        axes = self.figure.axes
        target_ax = None
        if self._selection_patch_axis_index is not None and 0 <= self._selection_patch_axis_index < len(axes):
            target_ax = axes[self._selection_patch_axis_index]
        elif axes:
            target_ax = axes[0]
        if target_ax is None:
            return
        self._create_selection_patch(target_ax, start, end)
        self._sync_selection_lines_to_range()
        if hasattr(self, "canvas") and self.canvas is not None:
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
    def _smooth_signal(x, window_samples):
        x = np.asarray(x, dtype=float)
        if window_samples <= 1 or x.size == 0:
            return x.copy()
        kernel = int(window_samples)
        if kernel <= 0:
            return x.copy()
        if kernel % 2 == 0:
            kernel += 1
        try:
            return signal.medfilt(x, kernel_size=kernel)
        except Exception:
            return x.copy()

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
        if self.available_state_values and int(state_value) not in self.available_state_values:
            return
        tmin, tmax = self.selection_range

        mask_epoch = (self.state_epoch_t >= tmin) & (self.state_epoch_t <= tmax)
        self.state_epoch[mask_epoch] = int(state_value)

        mask_10 = (self.state_10hz_t >= tmin) & (self.state_10hz_t <= tmax)
        self.state_10hz[mask_10] = int(state_value)

        self._state_dirty = True
        self._update_save_button_state()

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
            state_epoch, _ = score_from_epoch_features(
                self.epoch_features_data,
                thresholds=thresholds,
                auto_thresholds=False,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Rescore error", f"Error during rescoring:\n{exc}")
            return

        self.state_epoch = state_epoch.astype(np.int8)

        interp = interp1d(
            self.epoch_t,
            self.state_epoch,
            kind="nearest",
            bounds_error=False,
            fill_value=(self.state_epoch[0], self.state_epoch[-1]),
        )
        self.state_10hz = interp(self.state_10hz_t).astype(np.int8)

        self._state_dirty = True
        self._sync_threshold_widgets_to_values()
        self._update_state_axis_only()
        self._update_save_button_state()

    def onRescoreClicked(self):
        if not self.loaded:
            return
        band_values = self._parse_band_entries()
        if band_values is None:
            return
        delta_band_vals, theta_band_vals = band_values
        if self._bands_changed(delta_band_vals, theta_band_vals):
            self._run_full_rescore_with_new_bands(delta_band_vals, theta_band_vals)
        else:
            self.rerun_classification()

    def saveSleepScoring(self):
        if not self.loaded:
            return
        if not self.sleep_state_path:
            QMessageBox.warning(self, "Save error", "Sleep scoring path is unknown; load data first.")
            return

        sleep_state = dict(self.loaded_sleep_state) if isinstance(self.loaded_sleep_state, dict) else {}

        sleep_state["state_epoch"] = np.asarray(self.state_epoch, dtype=np.int8)
        sleep_state["state_epoch_t"] = np.asarray(self.state_epoch_t, dtype=np.float64)
        sleep_state["state_10hz"] = np.asarray(self.state_10hz, dtype=np.int8)
        sleep_state["state_10hz_t"] = np.asarray(self.state_10hz_t, dtype=np.float64)

        sleep_state["emg_rms_threshold"] = float(self.emg_rms_threshold)
        sleep_state["wheel_speed_threshold"] = float(self.wheel_speed_threshold)
        sleep_state["theta_delta_ratio_threshold"] = float(self.theta_delta_ratio_threshold)
        sleep_state["delta_power_threshold"] = float(self.delta_power_threshold)

        if self.epoch_features_data is not None:
            sleep_state["epoch_features"] = {
                key: np.asarray(val)
                for key, val in self.epoch_features_data.items()
            }

        try:
            with open(self.sleep_state_path, "wb") as f:
                pickle.dump(sleep_state, f)
        except Exception as e:
            base_name = self._sleep_state_basename()
            QMessageBox.critical(self, "Save error", f"Failed to save {base_name}:\n{e}")
            return

        self.loaded_sleep_state = sleep_state
        self._state_dirty = False
        self._update_save_button_state()
        QMessageBox.information(self, "Sleep scoring", "Updated sleep scoring data saved successfully.")

    # -----------------------------
    # Threshold widgets (drag + slider sync)
    # -----------------------------
    def _threshold_edit_enabled(self):
        if not self.modeThresholdButton.isEnabled():
            return False
        if not self.modeThresholdButton.isChecked():
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
            and self._threshold_edit_enabled()
        )

    def _handle_threshold_change(self):
        self._state_dirty = True
        self._update_save_button_state()
        if self._auto_rescore_enabled():
            self.rerun_classification()

    def _update_save_button_state(self):
        if hasattr(self, "saveButton") and self.saveButton is not None:
            self.saveButton.setEnabled(self.loaded and self._state_dirty)

    def _selection_enabled(self):
        return self.loaded and self.modeSelectionButton.isChecked()

    def onModeToggled(self, *_):
        selection_active = self._selection_enabled()
        for selector in self.span_selectors:
            try:
                selector.set_active(selection_active)
            except Exception:
                pass
        if not selection_active:
            self._clear_selection_visuals()
        slider_active = self._threshold_edit_enabled()
        for slider in [self._slider_emg, self._slider_wheel, self._slider_ratio]:
            if slider is not None:
                try:
                    slider.set_active(slider_active)
                except Exception:
                    pass

    def _clear_selection_visuals(self):
        self._remove_selection_patch()
        self.selection_range = None
        self._sync_selection_lines_to_range()
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.draw_idle()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _refresh_state_controls(self):
        self._clear_layout(self.stateButtonsLayout)
        self.state_buttons = []
        for sc in self.state_shortcuts:
            try:
                sc.activated.disconnect()
            except Exception:
                pass
            sc.setParent(None)
        self.state_shortcuts = []
        self.state_shortcut_map = {}
        self.available_state_values = set()

        if not self.loaded:
            return

        if self.state_label_map:
            keys = sorted(int(k) for k in self.state_label_map.keys())
            state_items = []
            for key in keys:
                label = self.state_label_map.get(key)
                if label is None:
                    label = self.state_label_map.get(str(key), f"State {key}")
                state_items.append((key, label))
        else:
            unique_states = np.unique(self.state_epoch) if hasattr(self, "state_epoch") else []
            state_items = [(int(val), f"State {int(val)}") for val in unique_states]
            if not state_items:
                state_items = [(0, "State 0")]

        self.available_state_values = {state_id for state_id, _ in state_items}

        for idx, (state_id, label) in enumerate(state_items):
            btn = QPushButton(f"{idx}: {label}")
            btn.clicked.connect(lambda _, sid=state_id: self.assign_state_key(sid))
            self.stateButtonsLayout.addWidget(btn)
            self.state_buttons.append(btn)

            shortcut_key = str(idx)
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(shortcut_key), self)
            shortcut.activated.connect(lambda sid=state_id: self.assign_state_key(sid))
            self.state_shortcuts.append(shortcut)
            self.state_shortcut_map[shortcut_key] = state_id

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
        fig.subplots_adjust(bottom=0.08, top=0.88, left=0.07, right=0.98, hspace=0.12)

        # Slider axes (normalized figure coords)
        ax_s_emg = fig.add_axes([0.08, 0.91, 0.28, 0.03])
        ax_s_wheel = fig.add_axes([0.40, 0.91, 0.28, 0.03])
        ax_s_ratio = fig.add_axes([0.72, 0.91, 0.20, 0.03])

        # Data-driven slider ranges
        emg_vals = self.emg_rms_mean_by_epoch
        if emg_vals is None:
            emg_vals = np.asarray(self.emg_rms_10hz, dtype=float)
        emg_finite = emg_vals[np.isfinite(emg_vals)]
        if emg_finite.size > 0:
            emg_max = float(np.nanmax(emg_finite))
        else:
            emg_max = max(1.0, float(self.emg_rms_threshold))
        emg_min = 0.0
        emg_max = max(emg_max * 1.05, emg_min + 1e-6)

        wheel_vals = self.wheel_speed_mean_by_epoch
        if wheel_vals is None:
            wheel_vals = np.asarray(self.wheel_10hz, dtype=float)
        wheel_finite = wheel_vals[np.isfinite(wheel_vals)]
        if wheel_finite.size > 0:
            wheel_max = float(np.nanmax(np.abs(wheel_finite)))
        else:
            wheel_max = max(0.1, abs(float(self.wheel_speed_threshold)))
        wheel_max = max(wheel_max * 1.05, 0.1)

        ratio_vals = np.asarray(self.theta_delta_ratio_smoothed, dtype=float)
        ratio_finite = ratio_vals[np.isfinite(ratio_vals)]
        if ratio_finite.size > 0:
            ratio_max = float(np.nanmax(ratio_finite))
        else:
            ratio_max = max(1.0, float(self.theta_delta_ratio_threshold))
        ratio_min = 0.0
        ratio_max = max(ratio_max * 1.05, ratio_min + 1e-6)

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
            if not self._threshold_edit_enabled():
                self._syncing_slider = True
                try:
                    if self._slider_emg is not None:
                        self._slider_emg.set_val(float(self.emg_rms_threshold))
                finally:
                    self._syncing_slider = False
                return
            self._syncing_slider = True
            try:
                new_val = float(val)
                self.emg_rms_threshold = new_val
                if self._thr_drag_emg is not None:
                    self._thr_drag_emg.set_y(new_val, trigger_callback=False)
            finally:
                self._syncing_slider = False
            self._handle_threshold_change()

        def on_wheel_slider(val):
            if self._syncing_slider:
                return
            if not self._threshold_edit_enabled():
                self._syncing_slider = True
                try:
                    if self._slider_wheel is not None:
                        self._slider_wheel.set_val(float(self.wheel_speed_threshold))
                finally:
                    self._syncing_slider = False
                return
            self._syncing_slider = True
            try:
                new_val = float(val)
                self.wheel_speed_threshold = new_val
                if self._thr_drag_wheel is not None:
                    self._thr_drag_wheel.set_y(new_val, trigger_callback=False)
            finally:
                self._syncing_slider = False
            self._handle_threshold_change()

        def on_ratio_slider(val):
            if self._syncing_slider:
                return
            if not self._threshold_edit_enabled():
                self._syncing_slider = True
                try:
                    if self._slider_ratio is not None:
                        self._slider_ratio.set_val(float(self.theta_delta_ratio_threshold))
                finally:
                    self._syncing_slider = False
                return
            self._syncing_slider = True
            try:
                new_val = float(val)
                self.theta_delta_ratio_threshold = new_val
                if self._thr_drag_ratio is not None:
                    self._thr_drag_ratio.set_y(new_val, trigger_callback=False)
            finally:
                self._syncing_slider = False
            self._handle_threshold_change()

        # Create sliders
        self._slider_emg = Slider(ax_s_emg, "", emg_min, emg_max, valinit=float(self.emg_rms_threshold))
        ax_s_emg.set_title("EMG thr", fontsize=9)
        self._slider_wheel = Slider(ax_s_wheel, "", 0.0, wheel_max, valinit=float(self.wheel_speed_threshold))
        ax_s_wheel.set_title("Wheel thr", fontsize=9)
        self._slider_ratio = Slider(ax_s_ratio, "", ratio_min, ratio_max, valinit=float(self.theta_delta_ratio_threshold))
        ax_s_ratio.set_title("θ/δ thr", fontsize=9)

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

        delta_thr_value = None
        if self.delta_power_threshold is not None and np.isfinite(self.delta_power_threshold):
            delta_thr_value = float(self.delta_power_threshold)

        # Full rebuild
        self.figure.clear()

        axs = self.figure.subplots(
            7, 1, sharex=True,
            gridspec_kw={'height_ratios': [1, 1, 1, 3, 1, 1, 1]}
        )
        self._axs = axs

        # Colors
        emg_color = 'tab:blue'
        wheel_color = 'tab:orange'

        # 1) EMG RMS
        ax1 = axs[0]
        ax1.plot(self.emg_rms_10hz_t, self.emg_rms_10hz, color=emg_color, label="EMG RMS")
        ax1.set_ylabel("EMG RMS", color=emg_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='y', labelcolor=emg_color)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.spines['left'].set_color(emg_color)

        # 2) Wheel
        ax2 = axs[1]
        ax2.plot(self.wheel_10hz_t, self.wheel_10hz, color=wheel_color, label="Wheel")
        ax2.set_ylabel("Wheel", color=wheel_color)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='y', labelcolor=wheel_color)
        ax2.spines['left'].set_color(wheel_color)
        ax2.tick_params(axis='x', labelbottom=False)
        if (
            isinstance(self.face_motion_10hz, np.ndarray)
            and isinstance(self.face_motion_10hz_t, np.ndarray)
            and self.face_motion_10hz.size > 0
        ):
            motion_ax = ax2.twinx()
            motion_signal = self.face_motion_10hz
            window_samples = max(1, int(round(self.motion_smoothing_secs * 10)))
            if window_samples > 1:
                motion_signal = self._smooth_signal(motion_signal, window_samples)
            motion_vals = self._zscore_safe(motion_signal)
            motion_ax.plot(
                self.face_motion_10hz_t,
                motion_vals,
                color="0.4",
                linewidth=1.0,
                label="Motion"
            )
            motion_ax.set_ylabel("Motion", color="0.4")
            motion_ax.tick_params(axis='y', labelcolor="0.4")
            motion_ax.spines['top'].set_visible(False)
            motion_ax.spines['right'].set_color("0.4")

        # 3) EEG 10 Hz
        ax3 = axs[2]
        ax3.plot(self.eeg_10hz_t, self.eeg_10hz)
        ax3.set_ylabel("EEG")
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(axis='x', labelbottom=False)

        # 4) EEG spectrogram, 0–20 Hz, saturation, pixelated
        ax4 = axs[3]
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

                ax4.imshow(spec_band, **im_args)
                ax4.set_ylim(0, max_freq)

        delta_low, delta_high = self.delta_band
        theta_low, theta_high = self.theta_band
        for y, color in [
            (delta_low, 'white'),
            (delta_high, 'white'),
            (theta_low, 'cyan'),
            (theta_high, 'cyan'),
        ]:
            ax4.axhline(y=y, color=color, linestyle=':', linewidth=1)

        ax4.set_ylabel("Freq (Hz)")
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(axis='x', labelbottom=False)

        # 5) Theta + Delta (raw power)
        ax5 = axs[4]
        ax5.plot(self.epoch_t, theta_masked, label="Theta")
        ax5.plot(self.epoch_t, delta_masked, label="Delta")
        ax5.set_ylabel("Power")
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.tick_params(axis='x', labelbottom=False)
        ax5.legend(loc="upper right")
        if delta_thr_value is not None:
            ax5.axhline(delta_thr_value, color='r', linestyle='--', linewidth=1)

        # 6) Ratio plot
        ax6 = axs[5]
        ax6.plot(self.epoch_t, ratio_masked)
        ax6.set_ylabel("Δ/Θ")
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.tick_params(axis='x', labelbottom=False)

        # 7) State axis (epoch-level)
        ax7 = axs[6]
        ax7.set_ylabel("State")
        ax7.set_xlabel("Time (s)")
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)

        self._ax_emg = ax1
        self._ax_wheel = ax2
        self._ax_ratio = ax6
        self._ax_state = ax7

        # Draw state once (from current arrays)
        self._draw_state_axis(ax7)

        self._setup_selection_vlines()

        # Vertical lines (current playhead)
        self.vlines = []
        current_time = self._current_time()
        for ax in axs:
            vline = ax.axvline(x=current_time, color='k', linestyle='--')
            self.vlines.append(vline)

        self._setup_span_selectors()
        self._restore_selection_patch()

        # Threshold widgets (draggable lines + sliders)
        self._setup_threshold_widgets()
        self.onModeToggled()

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
