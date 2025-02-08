import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QFileDialog
import csv
from itertools import zip_longest


class ZPlaneCanvas(FigureCanvas):
    from PyQt5.QtCore import pyqtSignal

    transfer_function_updated = pyqtSignal(list, list)

    def __init__(self):
        self.selected_conjugate = None
        self.selected = None
        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.figure.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1
        )  # Adjust these values as needed
        super().__init__(self.figure)

        self.zeros = []
        self.poles = []
        self.undo_stack = []
        self.redo_stack = []
        self.plot_z_plane(self.zeros, self.poles)

        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("motion_notify_event", self.on_drag)
        self.mpl_connect("button_release_event", self.on_release)

    def plot_z_plane(self, zeros, poles):
        self.ax.clear()
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(np.cos(theta), np.sin(theta), "b--", label="Unit Circle")

        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect("equal", adjustable="box")

        if zeros:
            self.ax.plot(
                [z.real for z in zeros], [z.imag for z in zeros], "go", label="Zeros"
            )
        if self.poles:
            self.ax.plot(
                [p.real for p in poles], [p.imag for p in poles], "rx", label="Poles"
            )

        self.draw()
        self.transfer_function_updated.emit(self.zeros, self.poles)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        elif event.button == 3:
            self.delete_point(event.xdata, event.ydata)
            self.plot_z_plane(self.zeros, self.poles)

        for i, (zero, pole) in enumerate(
            zip_longest(self.zeros, self.poles, fillvalue=None)
        ):
            if zero is not None:
                distance_zero = np.sqrt(
                    (event.xdata - zero.real) ** 2 + (event.ydata - zero.imag) ** 2
                )
            else:
                distance_zero = float("inf")

            if pole is not None:
                distance_pole = np.sqrt(
                    (event.xdata - pole.real) ** 2 + (event.ydata - pole.imag) ** 2
                )
            else:
                distance_pole = float("inf")

            if distance_zero < 0.1 and distance_zero < distance_pole:
                self.selection_flag = "zero"
                self.selected = zero
                self.selected_conjugate = zero
                self.selected_index = i
                self.save_state()  # Save state before dragging
                self.update_plot()
                return
            elif distance_pole < 0.1 and distance_pole < distance_zero:
                self.selection_flag = "pole"
                self.selected = pole
                self.selected_conjugate = pole
                self.selected_index = i
                self.save_state()  # Save state before dragging
                self.update_plot()
                return

    def delete_point(self, x, y):
        self.save_state()
        all_points = self.zeros + self.poles
        if not all_points:
            return

        distances = [abs(complex(x, y) - p) for p in all_points]
        if min(distances) < 0.05:
            closest_idx = np.argmin(distances)
            if closest_idx < len(self.zeros):
                self.zeros.pop(closest_idx)
            else:
                self.poles.pop(closest_idx - len(self.zeros))
        else:
            return

    def on_drag(self, event):

        if event.inaxes != self.ax or self.selected is None or event.button != 1:
            return

        if self.selection_flag == "zero":
            self.selected = complex(event.xdata, event.ydata)
            self.selected_conjugate = self.selected
            self.zeros[self.selected_index] = self.selected

        elif self.selection_flag == "pole":
            self.selected = complex(event.xdata, event.ydata)
            self.selected_conjugate = self.selected
            self.poles[self.selected_index] = self.selected

        self.update_plot()

    def on_release(self, event):
        if self.selected is not None:
            self.save_state()  # Save state after dragging
        self.selected = None

    def save_state(self):
        self.undo_stack.append((self.zeros.copy(), self.poles.copy()))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append((self.zeros.copy(), self.poles.copy()))
            self.zeros, self.poles = self.undo_stack.pop()
            self.plot_z_plane(self.zeros, self.poles)

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.zeros.copy(), self.poles.copy()))
            self.zeros, self.poles = self.redo_stack.pop()
            self.plot_z_plane(self.zeros, self.poles)

    def switch_zeros_poles(self):
        self.save_state()
        self.zeros, self.poles = self.poles, self.zeros
        self.plot_z_plane(self.zeros, self.poles)

    def add_conjugate(self):
        self.save_state()
        if self.selection_flag == "zero":
            zero = self.selected_conjugate
            self.zeros.append(complex(zero.real, -zero.imag))
            self.selected_conjugate = None

        elif self.selection_flag == "pole":
            pole = self.selected_conjugate
            self.poles.append(complex(pole.real, -pole.imag))
            self.selected_conjugate = None
        self.plot_z_plane(self.zeros, self.poles)

    def add_zero(self, x, y):
        self.save_state()
        self.zeros.append(complex(x, y))
        self.plot_z_plane(self.zeros, self.poles)

    def add_pole(self, x, y):
        self.save_state()
        self.poles.append(complex(x, y))
        self.plot_z_plane(self.zeros, self.poles)

    def clear_zeros(self):
        self.save_state()
        self.zeros = []
        self.plot_z_plane(self.zeros, self.poles)

    def clear_poles(self):
        self.save_state()
        self.poles = []
        self.plot_z_plane(self.zeros, self.poles)

    def clear_all(self):
        self.save_state()
        self.zeros = []
        self.poles = []
        self.plot_z_plane(self.zeros, self.poles)

    def save_state_to_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save State as CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if file_path:
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Type", "Real", "Imaginary"])
                for zero in self.zeros:
                    writer.writerow(["Zero", zero.real, zero.imag])
                for pole in self.poles:
                    writer.writerow(["Pole", pole.real, pole.imag])

    def load_state_from_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load State from CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if file_path:
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                next(reader)
                self.zeros = []
                self.poles = []
                for row in reader:
                    if row[0] == "Zero":
                        self.zeros.append(complex(float(row[1]), float(row[2])))
                    elif row[0] == "Pole":
                        self.poles.append(complex(float(row[1]), float(row[2])))
                self.plot_z_plane(self.zeros, self.poles)

    def update_plot(self):
        self.ax.clear()
        self.plot_z_plane(self.zeros, self.poles)
        parent_widget = self.parentWidget()
        while parent_widget and not hasattr(
            parent_widget, "update_add_conjugate_button"
        ):
            parent_widget = parent_widget.parentWidget()
        if parent_widget:
            parent_widget.update_add_conjugate_button()

        if self.selection_flag == "zero":
            self.ax.plot(
                self.selected.real,
                self.selected.imag,
                "o",
                color="red",
                markersize=12,
                alpha=0.6,
                zorder=1,
            )
            self.ax.plot(
                self.selected.real,
                self.selected.imag,
                "go",
                markersize=8,
                zorder=2,
            )

        elif self.selection_flag == "pole":
            self.ax.plot(
                self.selected.real,
                self.selected.imag,
                "o",
                color="orange",
                markersize=12,
                alpha=0.6,
                zorder=1,
            )
            self.ax.plot(
                self.selected.real,
                self.selected.imag,
                "rx",
                markersize=8,
                zorder=2,
            )

        self.draw()
