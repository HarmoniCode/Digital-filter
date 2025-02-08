import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QRadioButton,
    QPushButton,
    QSlider,
    QSpacerItem,
    QSizePolicy,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer
from TransferFunctionCanvas import TransferFunctionCanvas


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

        main_layout = QVBoxLayout()
        main_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter
        )

        self.radio_csv = QRadioButton("Upload CSV")
        self.radio_mouse = QRadioButton("Draw using Mouse")
        self.radio_csv.setChecked(True)
        self.radio_csv.toggled.connect(self.toggle_input_mode)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.radio_csv)
        button_layout.addWidget(self.radio_mouse)
        main_layout.addLayout(button_layout)

        H_button_layout = QHBoxLayout()

        self.input_button = QPushButton("Load Signal")
        self.input_button.clicked.connect(self.load_signal)
        H_button_layout.addWidget(self.input_button)

        self.clear_button = QPushButton("Clear Graphs")
        self.clear_button.clicked.connect(self.clear_graphs)
        H_button_layout.addWidget(self.clear_button)

        main_layout.addLayout(H_button_layout)

        self.temporal_resolution = QSlider(Qt.Horizontal)
        self.temporal_resolution.setMinimum(1)
        self.temporal_resolution.setMaximum(100)
        self.temporal_resolution.setValue(50)
        main_layout.addWidget(self.temporal_resolution)

        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)

        self.input_plot = pg.PlotWidget(title="Input Signal")
        self.input_plot.setLabel("bottom", "Time")
        self.input_plot.setLabel("left", "Amplitude")
        plot_layout.addWidget(self.input_plot)

        self.filtered_plot = pg.PlotWidget(title="Filtered Signal")
        self.filtered_plot.setLabel("bottom", "Time")
        self.filtered_plot.setLabel("left", "Amplitude")
        plot_layout.addWidget(self.filtered_plot)

        main_layout.addWidget(plot_frame)

        H_layout = QHBoxLayout()
        H_layout.addSpacerItem(
            QSpacerItem(
                0,
                0,
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Expanding,
            )
        )
        self.mouse_input_area = QWidget()
        H_layout.addWidget(self.mouse_input_area)
        H_layout.addSpacerItem(
            QSpacerItem(
                0,
                0,
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Expanding,
            )
        )
        self.mouse_input_area.setFixedSize(400, 300)
        self.mouse_input_area.setStyleSheet("background-color: lightgray;")
        self.mouse_input_area.setMouseTracking(True)
        self.mouse_input_area.mouseMoveEvent = self.mouse_move_event

        main_layout.addLayout(main_layout)
        main_layout.addLayout(H_layout)

        self.setLayout(main_layout)

    def clear_graphs(self):
        """Clear the input and filtered plots."""
        self.input_plot.clear()
        self.filtered_plot.clear()
        self.mouse_signal = []
        self.data = None
        self.input_current_index = 0
        self.filtered_current_index = 0

    #########################################################################################################################################################################
    # Mouse Plotting
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
            if (
                0 <= pos.x() < self.mouse_input_area.width()
                and 0 <= pos.y() < self.mouse_input_area.height()
            ):
                normalized_y = (
                    self.mouse_input_area.height() - pos.y()
                ) / self.mouse_input_area.height()
                amplitude = normalized_y * 2 - 1

                self.mouse_signal.append(amplitude)

                self.input_plot.clear()
                self.input_plot.plot(
                    range(len(self.mouse_signal)),
                    self.mouse_signal,
                    pen="b",
                    clear=True,
                )

                self.update_filtered_plot_mouse_event()

    def update_filtered_plot_mouse_event(self):
        """Update the filtered plot in real-time based on mouse-drawn signal."""
        if self.mouse_signal:
            amplitude = np.array(self.mouse_signal)

            time = np.arange(len(amplitude))

            transfer_function_time = self.update_H(
                self.z_plane_canvas.zeros, self.z_plane_canvas.poles
            )

            filtered_signal = np.convolve(
                amplitude, transfer_function_time, mode="full"
            )

            filtered_signal = filtered_signal[: len(time)]

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
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )

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
            input_end_index = min(
                self.input_current_index + self.temporal_resolution.value(),
                len(self.data[0]),
            )
            time = self.data[0][self.input_current_index : input_end_index].to_numpy()
            amplitude = self.data[1][
                self.input_current_index : input_end_index
            ].to_numpy()
            self.input_plot.plot(time, amplitude, pen="b", clear=False)
            self.input_current_index = input_end_index
        else:
            self.timer.stop()

    def update_filtered_plot(self):
        """Update the filtered plot."""
        if self.data is not None and self.filtered_current_index < len(self.data[0]):
            filtered_end_index = min(
                self.filtered_current_index + self.temporal_resolution.value(),
                len(self.data[0]),
            )
            time = self.data[0][
                self.filtered_current_index : filtered_end_index
            ].to_numpy()
            amplitude = self.data[1].to_numpy()
            transfer_function_time = self.update_H(
                self.z_plane_canvas.zeros, self.z_plane_canvas.poles
            )

            filtered_signal = np.convolve(
                amplitude, transfer_function_time, mode="full"
            )
            self.filtered_plot.plot(
                time,
                filtered_signal[self.filtered_current_index : filtered_end_index],
                pen="r",
                clear=False,
            )
            self.filtered_current_index = filtered_end_index
        else:
            self.timer.stop()

    def update_H(self, zeros, poles):
        """Compute the transfer function."""
        transfer_function_freq = (
            self.transfer_function_canvas.compute_transfer_function(
                False, zeros, poles
            )[0]
        )
        transfer_function_time = np.fft.ifft(transfer_function_freq).real
        return transfer_function_time
