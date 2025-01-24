import sys
from itertools import zip_longest
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QLabel, QPushButton, QHBoxLayout
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT)
import matplotlib.pyplot as plt


class ZPlanePlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Plane Plot")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.canvas = ZPlaneCanvas(self.main_widget)
        self.layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.layout.addWidget(self.canvas)

        self.coord_layout = QHBoxLayout()
        self.coord_label = QLabel("Enter Coordinates (Real, imaginary):")
        self.coord_layout.addWidget(self.coord_label)

        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("real")
        self.coord_layout.addWidget(self.x_input)

        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("imaginary")
        self.coord_layout.addWidget(self.y_input)

        self.layout.addLayout(self.coord_layout)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.add_zero_button = QPushButton("Add Zero")
        self.add_zero_button.clicked.connect(self.add_zero)
        self.button_layout.addWidget(self.add_zero_button)

        self.add_pole_button = QPushButton("Add Pole")
        self.add_pole_button.clicked.connect(self.add_pole)
        self.button_layout.addWidget(self.add_pole_button)
        
        self.add_conjugate_button = QPushButton("Add Conjugate")
        self.add_conjugate_button.clicked.connect(self.canvas.add_conjugate)
        self.button_layout.addWidget(self.add_conjugate_button)

        self.clear_zeros_button = QPushButton("Clear Zeros")
        self.clear_zeros_button.clicked.connect(self.canvas.clear_zeros)
        self.button_layout.addWidget(self.clear_zeros_button)

        self.clear_poles_button = QPushButton("Clear Poles")
        self.clear_poles_button.clicked.connect(self.canvas.clear_poles)
        self.button_layout.addWidget(self.clear_poles_button)

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.canvas.clear_all)
        self.button_layout.addWidget(self.clear_all_button)

    def add_zero(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            self.canvas.add_zero(x, y)
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    def add_pole(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            self.canvas.add_pole(x, y)
        except ValueError:
            print("Invalid input. Please enter numeric values.")


class ZPlaneCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.selected_index = None
        self.selected = None
        self.selection_flag = None
        self.selected_conjugate = None
        self.figure, self.ax = plt.subplots()
        super().__init__(self.figure)
        self.setParent(parent)
        self.zeros = []
        self.poles = []
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
            self.ax.plot(
                [z.real for z in self.zeros],
                [z.imag for z in self.zeros],
                "go",
                label="Zeros",
            )
        if self.poles:
            self.ax.plot(
                [p.real for p in self.poles],
                [p.imag for p in self.poles],
                "rx",
                label="Poles",
            )

        self.ax.legend()
        print("Current existing zeroes:")
        print(self.zeros)
        print("Current existing poles:")
        print(self.poles)
        self.draw()

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
                return
            elif distance_pole < 0.1 and distance_pole < distance_zero:
                self.selection_flag = 'pole'
                self.selected = pole
                self.selected_conjugate = pole
                self.selected_index = i
                return
        # if event.button == 1:
        #     self.zeros.append(complex(event.xdata, event.ydata))
        #     self.plot_z_plane()

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
            self.zeros[self.selected_index] = self.selected

        elif self.selection_flag == 'pole':
            self.selected = complex(event.xdata, event.ydata)
            self.poles[self.selected_index] = self.selected

        self.update_plot()

    def on_release(self, event):
        self.selected = None

    def add_zero(self, x, y):
        self.zeros.append(complex(x, y))
        self.plot_z_plane()
    

    def add_pole(self, x, y):
        self.poles.append(complex(x, y))
        self.plot_z_plane()
        
    def add_conjugate(self):
        if self.selection_flag == 'zero':
            zero = self.selected_conjugate
            self.zeros.append(complex(zero.real, -zero.imag))

        elif self.selection_flag == 'pole':
            pole = self.selected_conjugate
            self.poles.append(complex(pole.real, -pole.imag))

        self.plot_z_plane()

    def clear_zeros(self):
        self.zeros = []
        self.plot_z_plane()

    def clear_poles(self):
        self.poles = []
        self.plot_z_plane()

    def clear_all(self):
        self.zeros = []
        self.poles = []
        self.plot_z_plane()

    def update_plot(self):

        self.ax.clear()
        self.plot_z_plane()

        if self.selection_flag == 'zero':
            self.ax.plot(self.selected.real, self.selected.imag, "bo")
        elif self.selection_flag == 'pole':
            self.ax.plot(self.selected.real, self.selected.imag, "bx")

        self.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ZPlanePlotApp()
    main_window.show()
    sys.exit(app.exec_())
