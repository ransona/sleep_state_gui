import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class DraggableHLine:
    """
    Click near the horizontal line, then drag to move it up/down.

    With toolbar support:
      - If Pan/Zoom is active, dragging is disabled so it doesn't fight the nav tools.
    """
    def __init__(self, ax, y0=0.0, pick_tol_pixels=8, on_change=None, clamp=None, toolbar=None):
        self.ax = ax
        self.on_change = on_change
        self.clamp = clamp          # (ymin, ymax) or None
        self.toolbar = toolbar      # NavigationToolbar2Tk or None

        # Draw the threshold line (owned by this class, like your working version)
        self.line = ax.axhline(y0, linewidth=2, picker=False)

        self._dragging = False
        self._pick_tol_pixels = pick_tol_pixels

        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def y(self):
        return float(self.line.get_ydata()[0])

    def set_y(self, y):
        y = float(y)
        if self.clamp is not None:
            lo, hi = self.clamp
            if lo is not None:
                y = max(lo, y)
            if hi is not None:
                y = min(hi, y)

        self.line.set_ydata([y, y])
        if self.on_change is not None:
            self.on_change(y)
        self.canvas.draw_idle()

    def _blocked_by_nav_tools(self):
        # If toolbar pan/zoom is active OR Matplotlib widgetlock is held, don't drag.
        if self.toolbar is not None and bool(getattr(self.toolbar, "mode", "")):
            return True
        if hasattr(self.canvas, "widgetlock") and self.canvas.widgetlock.locked():
            return True
        return False

    def _near_line(self, event):
        if event.inaxes != self.ax:
            return False
        if event.x is None or event.y is None:
            return False

        y_line = self.y()
        x0 = self.ax.get_xlim()[0]
        _, ydisp = self.ax.transData.transform((x0, y_line))
        return abs(event.y - ydisp) <= self._pick_tol_pixels

    def _on_press(self, event):
        if event.button != 1:
            return
        if self._blocked_by_nav_tools():
            return
        if self._near_line(event):
            self._dragging = True

    def _on_move(self, event):
        if not self._dragging:
            return
        if event.inaxes != self.ax:
            return
        if event.ydata is None:
            return
        self.set_y(event.ydata)

    def _on_release(self, event):
        if event.button == 1:
            self._dragging = False


def main():
    root = tk.Tk()
    root.title("Two plots + draggable thresholds + sliders + zoom/pan")

    top = tk.Frame(root)
    top.pack(fill="both", expand=True)

    controls = tk.Frame(root)
    controls.pack(fill="x")

    # ---- Figure / Axes ----
    fig = Figure(figsize=(9, 5), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    x = np.linspace(0, 2*np.pi, 800)
    ax1.plot(x, np.sin(x), linewidth=2)
    ax2.plot(x, np.sin(x + 0.75), linewidth=2)

    for ax, title in [(ax1, "Plot 1"), (ax2, "Plot 2")]:
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("sin(x)")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)

    # ---- Status ----
    status = tk.StringVar(value="t1: 0.000    t2: 0.000")
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=8, pady=(8, 0))

    # ---- Canvas + Toolbar (zoom/pan controls) ----
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)

    toolbar = NavigationToolbar2Tk(canvas, top)
    toolbar.update()

    # ---- Sliders ----
    v1 = tk.DoubleVar(value=0.0)
    v2 = tk.DoubleVar(value=0.0)
    syncing = {"s1": False, "s2": False}

    def update_status():
        status.set(f"t1: {v1.get():.3f}    t2: {v2.get():.3f}")

    tk.Label(controls, text="Threshold 1").grid(row=0, column=0, sticky="w", padx=8, pady=6)
    s1 = tk.Scale(
        controls, from_=-1.5, to=1.5, resolution=0.01,
        orient="horizontal", length=420, variable=v1
    )
    s1.grid(row=0, column=1, sticky="ew", padx=8, pady=6)

    tk.Label(controls, text="Threshold 2").grid(row=1, column=0, sticky="w", padx=8, pady=6)
    s2 = tk.Scale(
        controls, from_=-1.5, to=1.5, resolution=0.01,
        orient="horizontal", length=420, variable=v2
    )
    s2.grid(row=1, column=1, sticky="ew", padx=8, pady=6)

    controls.grid_columnconfigure(1, weight=1)

    # ---- Draggable lines (same logic as your working version), now toolbar-aware ----
    def on_drag_1(y):
        syncing["s1"] = True
        try:
            v1.set(y)
            update_status()
        finally:
            syncing["s1"] = False

    def on_drag_2(y):
        syncing["s2"] = True
        try:
            v2.set(y)
            update_status()
        finally:
            syncing["s2"] = False

    d1 = DraggableHLine(ax1, y0=0.0, pick_tol_pixels=10, on_change=on_drag_1,
                        clamp=(-1.5, 1.5), toolbar=toolbar)
    d2 = DraggableHLine(ax2, y0=0.0, pick_tol_pixels=10, on_change=on_drag_2,
                        clamp=(-1.5, 1.5), toolbar=toolbar)

    # Slider -> line (avoid feedback loops)
    def slider1_changed(*_):
        if syncing["s1"]:
            return
        d1.set_y(v1.get())
        update_status()

    def slider2_changed(*_):
        if syncing["s2"]:
            return
        d2.set_y(v2.get())
        update_status()

    v1.trace_add("write", slider1_changed)
    v2.trace_add("write", slider2_changed)

    canvas.draw()
    update_status()
    root.mainloop()


if __name__ == "__main__":
    main()
