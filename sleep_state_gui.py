import numpy as np


class DraggableHLine:
    """
    Robust draggable horizontal line using explicit proximity detection.
    No picking, no widgetlock, no toolbar checks.
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
    ):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.on_changed = on_changed
        self.tolerance_px = tolerance_px

        # create line internally (IMPORTANT)
        self.line = ax.axhline(
            y0,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        self._dragging = False

        # connect events ONCE
        self.cid_press = self.canvas.mpl_connect(
            "button_press_event", self._on_press
        )
        self.cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self._on_motion
        )
        self.cid_release = self.canvas.mpl_connect(
            "button_release_event", self._on_release
        )

    def get_y(self):
        return float(self.line.get_ydata()[0])

    def set_y(self, y):
        self.line.set_ydata([y, y])
        if self.on_changed is not None:
            self.on_changed(y)
        self.canvas.draw_idle()

    def _is_near_line(self, event):
        if event.inaxes is not self.ax:
            return False
        if event.ydata is None:
            return False

        # explicit pixel-distance test
        y_line = self.get_y()
        x_dummy = 0.0

        p_event = self.ax.transData.transform((x_dummy, event.ydata))
        p_line = self.ax.transData.transform((x_dummy, y_line))

        dy = abs(p_event[1] - p_line[1])
        return dy <= self.tolerance_px

    def _on_press(self, event):
        if event.button != 1:
            return
        if self._is_near_line(event):
            self._dragging = True

    def _on_motion(self, event):
        if not self._dragging:
            return
        if event.inaxes is not self.ax:
            return
        if event.ydata is None:
            return

        self.set_y(event.ydata)

    def _on_release(self, event):
        self._dragging = False
