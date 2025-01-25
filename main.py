import sys
from itertools import zip_longest
import numpy as np
import csv
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLineEdit, QLabel, QPushButton, QSplitter, QSlider, QRadioButton
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg


class ZPlanePlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graphs_window = None
        self.setWindowTitle("Z-Plane Plot and Transfer Function")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)

        self.z_plane_canvas = ZPlaneCanvas()
        self.selected_conjugate = self.z_plane_canvas.selected_conjugate
        left_layout.addWidget(NavigationToolbar(self.z_plane_canvas, self))
        left_layout.addWidget(self.z_plane_canvas)

        self.coord_layout = QHBoxLayout()
        self.coord_label = QLabel("Enter Coordinates (Real, Imaginary):")
        self.coord_layout.addWidget(self.coord_label)

        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Real")
        self.coord_layout.addWidget(self.x_input)

        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("Imaginary")
        self.coord_layout.addWidget(self.y_input)

        left_layout.addLayout(self.coord_layout)

        self.button_layout = QHBoxLayout()

        self.add_zero_button = QPushButton("Add Zero")
        self.add_zero_button.clicked.connect(self.add_zero)
        self.button_layout.addWidget(self.add_zero_button)

        self.add_pole_button = QPushButton("Add Pole")
        self.add_pole_button.clicked.connect(self.add_pole)
        self.button_layout.addWidget(self.add_pole_button)

        self.add_conjugate_button = QPushButton("Add Conjugate")
        self.add_conjugate_button.clicked.connect(self.z_plane_canvas.add_conjugate)
        self.button_layout.addWidget(self.add_conjugate_button)
        self.add_conjugate_button.setDisabled(True)

        self.clear_zeros_button = QPushButton("Clear Zeros")
        self.clear_zeros_button.clicked.connect(self.z_plane_canvas.clear_zeros)
        self.button_layout.addWidget(self.clear_zeros_button)

        self.clear_poles_button = QPushButton("Clear Poles")
        self.clear_poles_button.clicked.connect(self.z_plane_canvas.clear_poles)
        self.button_layout.addWidget(self.clear_poles_button)

        self.button_layout_2 = QHBoxLayout()

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.z_plane_canvas.clear_all)
        self.button_layout_2.addWidget(self.clear_all_button)

        self.switch_button = QPushButton("Switch Zeros and Poles")
        self.switch_button.clicked.connect(self.switch_zeros_poles)
        self.button_layout_2.addWidget(self.switch_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.z_plane_canvas.undo)
        self.button_layout_2.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.z_plane_canvas.redo)
        self.button_layout_2.addWidget(self.redo_button)

        self.save_csv_button = QPushButton("Save as CSV")
        self.save_csv_button.clicked.connect(self.z_plane_canvas.save_state_to_csv)
        self.button_layout_2.addWidget(self.save_csv_button)

        self.load_csv_button = QPushButton("Load from CSV")
        self.load_csv_button.clicked.connect(self.z_plane_canvas.load_state_from_csv)
        self.button_layout_2.addWidget(self.load_csv_button)

        left_layout.addLayout(self.button_layout)
        left_layout.addLayout(self.button_layout_2)
        splitter.addWidget(left_pane)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)

        self.transfer_function_canvas = TransferFunctionCanvas()
        right_layout.addWidget(NavigationToolbar(self.transfer_function_canvas, self))
        right_layout.addWidget(self.transfer_function_canvas)

        splitter.addWidget(right_pane)
        splitter.setSizes([400, 8000])

        self.z_plane_canvas.transfer_function_updated.connect(
            self.transfer_function_canvas.update_transfer_function
        )

    def update_add_conjugate_button(self):
        self.selected_conjugate = self.z_plane_canvas.selected_conjugate
        if self.selected_conjugate is None:
            self.add_conjugate_button.setDisabled(True)
        else:
            self.add_conjugate_button.setDisabled(False)

    def switch_zeros_poles(self):
        self.z_plane_canvas.switch_zeros_poles()

    def add_zero(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            self.z_plane_canvas.add_zero(x, y)
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    def add_pole(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            self.z_plane_canvas.add_pole(x, y)
        except ValueError:
            print("Invalid input. Please enter numeric values.")


class ZPlaneCanvas(FigureCanvas):
    from PyQt5.QtCore import pyqtSignal

    transfer_function_updated = pyqtSignal(list, list)

    def __init__(self):
        self.selected_conjugate = None
        self.selected = None
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        super().__init__(self.figure)

        self.zeros = []
        self.poles = []
        self.undo_stack = []
        self.redo_stack = []

        self.plot_z_plane()

        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("motion_notify_event", self.on_drag)
        self.mpl_connect("button_release_event", self.on_release)

    def plot_z_plane(self):
        self.ax.clear()

        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(np.cos(theta), np.sin(theta), "b--", label="Unit Circle")

        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect("equal", adjustable="box")

        if self.zeros:
            self.ax.plot([z.real for z in self.zeros], [z.imag for z in self.zeros], "go", label="Zeros")
        if self.poles:
            self.ax.plot([p.real for p in self.poles], [p.imag for p in self.poles], "rx", label="Poles")

        self.ax.legend()
        print("Current existing zeroes:")
        print(self.zeros)
        print("Current existing poles:")
        print(self.poles)
        self.draw()
        self.transfer_function_updated.emit(self.zeros, self.poles)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        elif event.button == 3:
            self.delete_point(event.xdata, event.ydata)
            self.plot_z_plane()

        for i, (zero, pole) in enumerate(zip_longest(self.zeros, self.poles, fillvalue=None)):
            if zero is not None:
                distance_zero = np.sqrt((event.xdata - zero.real) ** 2 + (event.ydata - zero.imag) ** 2)
            else:
                distance_zero = float('inf')

            if pole is not None:
                distance_pole = np.sqrt((event.xdata - pole.real) ** 2 + (event.ydata - pole.imag) ** 2)
            else:
                distance_pole = float('inf')

            if distance_zero < 0.1 and distance_zero < distance_pole:
                self.selection_flag = 'zero'
                self.selected = zero
                self.selected_conjugate = zero
                self.selected_index = i
                self.update_plot()
                return
            elif distance_pole < 0.1 and distance_pole < distance_zero:
                self.selection_flag = 'pole'
                self.selected = pole
                self.selected_conjugate = pole
                self.selected_index = i
                self.update_plot()
                return

    def delete_point(self, x, y):
        all_points = self.zeros + self.poles
        if not all_points:
            return

        distances = [abs(complex(x, y) - p) for p in all_points]
        print(distances)
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

        if self.selection_flag == 'zero':
            self.selected = complex(event.xdata, event.ydata)
            self.selected_conjugate = self.selected
            self.zeros[self.selected_index] = self.selected

        elif self.selection_flag == 'pole':
            self.selected = complex(event.xdata, event.ydata)
            self.selected_conjugate = self.selected
            self.poles[self.selected_index] = self.selected

        self.update_plot()

    def on_release(self, event):
        self.selected = None

    def save_state(self):
        self.undo_stack.append((self.zeros.copy(), self.poles.copy()))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append((self.zeros.copy(), self.poles.copy()))
            self.zeros, self.poles = self.undo_stack.pop()
            self.plot_z_plane()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.zeros.copy(), self.poles.copy()))
            self.zeros, self.poles = self.redo_stack.pop()
            self.plot_z_plane()

    def switch_zeros_poles(self):
        self.save_state()
        self.zeros, self.poles = self.poles, self.zeros
        self.plot_z_plane()

    def add_conjugate(self):
        self.save_state()
        if self.selection_flag == 'zero':
            zero = self.selected_conjugate
            self.zeros.append(complex(zero.real, -zero.imag))
            self.selected_conjugate = None

        elif self.selection_flag == 'pole':
            pole = self.selected_conjugate
            self.poles.append(complex(pole.real, -pole.imag))
            self.selected_conjugate = None
        self.plot_z_plane()

    def add_zero(self, x, y):
        self.save_state()
        self.zeros.append(complex(x, y))
        self.plot_z_plane()

    def add_pole(self, x, y):
        self.save_state()
        self.poles.append(complex(x, y))
        self.plot_z_plane()

    def clear_zeros(self):
        self.save_state()
        self.zeros = []
        self.plot_z_plane()

    def clear_poles(self):
        self.save_state()
        self.poles = []
        self.plot_z_plane()

    def clear_all(self):
        self.save_state()
        self.zeros = []
        self.poles = []
        self.plot_z_plane()

    def save_state_to_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save State as CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Type', 'Real', 'Imaginary'])
                for zero in self.zeros:
                    writer.writerow(['Zero', zero.real, zero.imag])
                for pole in self.poles:
                    writer.writerow(['Pole', pole.real, pole.imag])

    def load_state_from_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load State from CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                self.zeros = []
                self.poles = []
                for row in reader:
                    if row[0] == 'Zero':
                        self.zeros.append(complex(float(row[1]), float(row[2])))
                    elif row[0] == 'Pole':
                        self.poles.append(complex(float(row[1]), float(row[2])))
                self.plot_z_plane()

    def update_plot(self):

        self.ax.clear()
        self.plot_z_plane()
        self.parent().parent().parent().parent().update_add_conjugate_button()

        if self.selection_flag == 'zero':
            self.ax.plot(
                self.selected.real, self.selected.imag,
                "o", color="red", markersize=12, alpha=0.6, zorder=1,
            )
            self.ax.plot(
                self.selected.real, self.selected.imag,
                "go", markersize=8, zorder=2,
            )

        elif self.selection_flag == 'pole':
            self.ax.plot(
                self.selected.real, self.selected.imag,
                "o", color="green", markersize=12, alpha=0.6, zorder=1,
            )
            self.ax.plot(
                self.selected.real, self.selected.imag,
                "rx", markersize=8, zorder=2,
            )

        self.ax.legend()
        self.draw()


class TransferFunctionCanvas(FigureCanvas):
    def __init__(self):
        self.figure, (self.ax_mag, self.ax_phase) = plt.subplots(2, 1, figsize=(6, 6))
        super().__init__(self.figure)
        self.plot_initial()
        self.z_plane_canvas = ZPlaneCanvas()

    def plot_initial(self):
        self.ax_mag.set_title("Magnitude of Transfer Function")
        self.ax_mag.set_xlabel("Frequency (rad/s)")
        self.ax_mag.set_ylabel("Magnitude")
        self.ax_mag.grid(True)

        self.ax_phase.set_title("Phase of Transfer Function")
        self.ax_phase.set_xlabel("Frequency (rad/s)")
        self.ax_phase.set_ylabel("Phase (radians)")
        self.ax_phase.grid(True)

        self.draw()

    def compute_transfer_function(self, zeros, poles):
        omega = np.linspace(0, 2 * np.pi, 500)
        z = np.exp(1j * omega)

        Y_Z = np.ones_like(z, dtype=complex)
        X_Z = np.ones_like(z, dtype=complex)

        for zero in zeros:
            Y_Z *= (z - zero)
        for pole in poles:
            X_Z *= (z - pole)

        H = Y_Z / X_Z
        return H, omega

    def update_transfer_function(self, zeros, poles):
        H, omega = self.compute_transfer_function(zeros, poles)
        magnitude = np.abs(H)
        phase = np.angle(H)

        self.ax_mag.clear()
        self.ax_phase.clear()

        self.ax_mag.plot(omega, magnitude, label="|H(z)|", color="blue")
        self.ax_mag.set_title("Magnitude of Transfer Function")
        self.ax_mag.set_xlabel("Frequency (rad/s)")
        self.ax_mag.set_ylabel("Magnitude")
        self.ax_mag.grid(True)

        self.ax_phase.plot(omega, phase, label="∠H(z)", color="orange")
        self.ax_phase.set_title("Phase of Transfer Function")
        self.ax_phase.set_xlabel("Frequency (rad/s)")
        self.ax_phase.set_ylabel("Phase (radians)")
        self.ax_phase.grid(True)

        self.draw()


class GraphsWindow(QWidget):
    def __init__(self, z_plane_canvas, parent=None):
        super().__init__(parent)
        self.z_plane_canvas = z_plane_canvas

        self.data = None
        self.setWindowTitle("Graphs")
        self.resize(1000, 600)
        self.transfer_function_canvas = TransferFunctionCanvas()
        self.input_current_index = 0
        self.filtered_current_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)

        self.mouse_timer = QTimer()
        self.mouse_timer.timeout.connect(self.update_mouse_signal)
        self.mouse_signal = []

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.radio_csv = QRadioButton("Upload CSV")
        self.radio_mouse = QRadioButton("Draw using Mouse")
        self.radio_csv.setChecked(True)
        self.radio_csv.toggled.connect(self.toggle_input_mode)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.radio_csv)
        button_layout.addWidget(self.radio_mouse)
        left_layout.addLayout(button_layout)

        self.input_button = QPushButton("Load Signal")
        self.input_button.clicked.connect(self.load_signal)
        left_layout.addWidget(self.input_button)

        self.temporal_resolution = QSlider(Qt.Horizontal)
        self.temporal_resolution.setMinimum(1)
        self.temporal_resolution.setMaximum(100)
        self.temporal_resolution.setValue(50)
        left_layout.addWidget(self.temporal_resolution)

        self.input_plot = pg.PlotWidget(title="Input Signal")
        self.input_plot.setLabel("bottom", "Time")
        self.input_plot.setLabel("left", "Amplitude")
        left_layout.addWidget(self.input_plot)

        self.filtered_plot = pg.PlotWidget(title="Filtered Signal")
        self.filtered_plot.setLabel("bottom", "Time")
        self.filtered_plot.setLabel("left", "Amplitude")
        left_layout.addWidget(self.filtered_plot)

        self.mouse_input_area = QWidget()
        self.mouse_input_area.setFixedSize(400, 300)
        self.mouse_input_area.setStyleSheet("background-color: lightgray;")
        self.mouse_input_area.setMouseTracking(True)
        self.mouse_input_area.mouseMoveEvent = self.mouse_move_event

        self.third_plot = pg.PlotWidget(title="TBD")
        self.third_plot.setLabel("bottom", "Time")
        self.third_plot.setLabel("left", "Amplitude")
        left_layout.addWidget(self.third_plot)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.mouse_input_area)

        self.setLayout(main_layout)

#########################################################################################################################################################################
#Mouse Plotting
    def toggle_input_mode(self):
        """Enable or disable the upload button and reset the input mode."""
        if self.radio_csv.isChecked():
            self.input_button.setEnabled(True)
            self.input_plot.clear()
            self.filtered_plot.clear()
            self.mouse_signal = []
        else:
            self.input_button.setEnabled(False)
            self.input_plot.clear()
            self.filtered_plot.clear()
            self.mouse_signal = []

    def mouse_move_event(self, event):
        """Handle mouse movement to draw the input signal and update the filtered plot."""
        if self.radio_mouse.isChecked():

            pos = event.pos()
            if 0 <= pos.x() < self.mouse_input_area.width() and 0 <= pos.y() < self.mouse_input_area.height():
                normalized_y = (self.mouse_input_area.height() - pos.y()) / self.mouse_input_area.height()
                amplitude = normalized_y * 2 - 1

                self.mouse_signal.append(amplitude)

                self.input_plot.clear()
                self.input_plot.plot(range(len(self.mouse_signal)), self.mouse_signal, pen="b", clear=True)

                self.update_filtered_plot_mouse_event()

    def update_filtered_plot_mouse_event(self):
        """Update the filtered plot in real-time based on mouse-drawn signal."""
        if self.mouse_signal:
            amplitude = np.array(self.mouse_signal)

            time = np.arange(len(amplitude))

            transfer_function_time = self.update_H(self.z_plane_canvas.zeros, self.z_plane_canvas.poles)

            filtered_signal = np.convolve(amplitude, transfer_function_time, mode="same")

            filtered_signal = filtered_signal[:len(time)]

            self.filtered_plot.clear()
            self.filtered_plot.plot(time, filtered_signal, pen="r", clear=True)

    def update_mouse_signal(self):
        """Update the input plot with the mouse-drawn signal."""
        if self.radio_mouse.isChecked():
            time = np.linspace(0, len(self.mouse_signal) / 30, len(self.mouse_signal))

            self.input_plot.clear()
            self.input_plot.plot(time, self.mouse_signal, pen="g")

#########################################################################################################################################################################
    def load_signal(self):
        """Load a signal from a CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)

        if file_path:
            self.data = pd.read_csv(file_path, header=None)
            self.input_plot.clear()
            self.timer.start(100)

    def update_plots(self):
        """Update both input and filtered plots."""
        self.update_input_plot()
        self.update_filtered_plot()

    def update_input_plot(self):
        """Update the input plot with data from the CSV file."""
        if self.data is not None and self.input_current_index < len(self.data[0]):
            input_end_index = min(self.input_current_index + 50, len(self.data[0]))
            time = self.data[0][self.input_current_index:input_end_index].to_numpy()
            amplitude = self.data[1][self.input_current_index:input_end_index].to_numpy()
            self.input_plot.plot(time, amplitude, pen="b", clear=False)
            self.input_current_index = input_end_index
        else:
            self.timer.stop()

    def update_filtered_plot(self):
        """Update the filtered plot."""
        if self.data is not None and self.filtered_current_index < len(self.data[0]):
            filtered_end_index = min(self.filtered_current_index + self.temporal_resolution.value(), len(self.data[0]))
            time = self.data[0][self.filtered_current_index:filtered_end_index].to_numpy()
            amplitude = self.data[1].to_numpy()
            transfer_function_time = self.update_H(self.z_plane_canvas.zeros,
                                                   self.z_plane_canvas.poles)

            filtered_signal = np.convolve(amplitude, transfer_function_time, mode="same")
            self.filtered_plot.plot(time, filtered_signal[self.filtered_current_index:filtered_end_index], pen="r",
                                    clear=False)
            self.filtered_current_index = filtered_end_index
        else:
            self.timer.stop()

    def update_H(self, zeros, poles):
        """Compute the transfer function."""
        transfer_function_freq = self.transfer_function_canvas.compute_transfer_function(zeros, poles)[0]
        transfer_function_time = np.fft.ifft(transfer_function_freq).real
        return transfer_function_time


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ZPlanePlotApp()
    graphs_window = GraphsWindow(main_window.z_plane_canvas)
    main_window.z_plane_canvas.transfer_function_updated.connect(
        graphs_window.update_filtered_plot
    )
    main_window.show()
    graphs_window.show()

    sys.exit(app.exec_())
