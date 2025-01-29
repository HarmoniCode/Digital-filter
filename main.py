import sys
from itertools import zip_longest
from jinja2 import Template
import numpy as np
import csv

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLineEdit, QLabel, QPushButton, QSplitter, QSlider, QRadioButton, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg
from scipy.signal import butter, cheby1, bessel, ellip, tf2zpk, zpk2tf


class ZPlanePlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graphs_window = None
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.axis('off')

        self.standard_filters = {
            "Butterworth LPF": None,
            "Butterworth HPF": None,
            "Butterworth BPF": None,
            "Chebyshev LPF": None,
            "Chebyshev HPF": None,
            "Chebyshev BPF": None,
            "Bessel LPF": None,
            "Bessel HPF": None,
            "Bessel BPF": None,
            "Elliptic LPF": None,
            "Elliptic HPF": None,
            "Elliptic BPF": None
        }
        self.all_pass_filters = {
            "Custom APF": None,
            "All-Pass Filter 1": None,
            "All-Pass Filter 2": None,
            "All-Pass Filter 3": None,
            "All-Pass Filter 4": None
        }
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
##################
        self.direct_form_ii_widget = QWidget()
        self.direct_form_ii_layout = QVBoxLayout(self.direct_form_ii_widget)
        self.form_canvas = FigureCanvas(self.fig)
        self.direct_form_ii_layout.addWidget(self.form_canvas)
        left_layout.addWidget(self.direct_form_ii_widget)
###############
        self.coord_layout = QHBoxLayout()
        self.coord_label = QLabel("Enter Coordinates (Real, Imaginary):")
        self.coord_layout.addWidget(self.coord_label)

        validator = QDoubleValidator(-1.5, 1.5, 2, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Real")
        self.x_input.setValidator(validator)
        self.x_input.setToolTip("Value must be between -1.5 and 1.5")
        self.coord_layout.addWidget(self.x_input)

        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("Imaginary")
        self.y_input.setValidator(validator)
        self.y_input.setToolTip("Value must be between -1.5 and 1.5")
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

        self.generate_code_button = QPushButton("Generate C Code")
        self.generate_code_button.clicked.connect(self.generate_c_code)
        self.button_layout_2.addWidget(self.generate_code_button)

        self.filter_dropdown = QComboBox()
        self.filter_dropdown.insertItem(0, "Choose Standard Filter")
        self.filter_dropdown.addItems([
            "Butterworth LPF", "Butterworth HPF", "Butterworth BPF",
            "Chebyshev LPF", "Chebyshev HPF", "Chebyshev BPF",
            "Bessel LPF", "Bessel HPF", "Bessel BPF",
            "Elliptic LPF", "Elliptic HPF", "Elliptic BPF"
        ])
        self.filter_dropdown.setCurrentIndex(0)
        self.filter_dropdown.currentIndexChanged.connect(self.select_filter)
        left_layout.addWidget(self.filter_dropdown)
        
        self.form_dropdown = QComboBox()
        self.form_dropdown.addItems([
             "Cascade Form", "Direct Form II"
        ])
        self.form_dropdown.currentIndexChanged.connect(self.select_form)
        left_layout.addWidget(self.form_dropdown) 

        self.apf_dropdown = QComboBox()
        self.apf_dropdown.insertItem(0, "Choose All-Pass Filter")
        self.apf_dropdown.addItems([
            "Custom APF", "All-Pass Filter 1", "All-Pass Filter 2", "All-Pass Filter 3", "All-Pass Filter 4"
        ])
        self.apf_dropdown.setCurrentIndex(0)
        self.apf_dropdown.currentIndexChanged.connect(self.update_chosen_apf)
        self.apf_dropdown.currentIndexChanged.connect(self.toggle_a_spinbox)
        left_layout.addWidget(self.apf_dropdown)

        self.a_spinbox = QDoubleSpinBox()
        self.a_spinbox.setDisabled(True)
        self.a_spinbox.setRange(0.1, 0.9)  # Set the range of values (min, max)
        self.a_spinbox.setValue(0.1)  # Set the initial value
        self.a_spinbox.setSingleStep(0.1)
        self.a_spinbox.valueChanged.connect(self.update_custom_apf)  # Connect signal to slot
        left_layout.addWidget(self.a_spinbox)
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
        self.create_standard_filter_library()
        self.create_all_pass_filter_library()
        
    def create_standard_filter_library(self):

        for filter_type in self.standard_filters.keys():
            if "Butterworth" in filter_type:
                if "LPF" in filter_type:
                    b, a, k = butter(N=4, Wn=0.5, btype='low', analog=False, output='zpk')
                elif "HPF" in filter_type:
                    b, a, k = butter(N=4, Wn=0.5, btype='high', analog=False, output='zpk')
                elif "BPF" in filter_type:
                    b, a, k = butter(N=4, Wn=[0.3, 0.7], btype='band', analog=False, output='zpk')
            elif "Chebyshev" in filter_type:
                if "LPF" in filter_type:
                    b, a, k = cheby1(N=4, rp=1, Wn=0.5, btype='low', analog=False, output='zpk')
                elif "HPF" in filter_type:
                    b, a, k = cheby1(N=4, rp=1, Wn=0.5, btype='high', analog=False, output='zpk')
                elif "BPF" in filter_type:
                    b, a, k = cheby1(N=4, rp=1, Wn=[0.3, 0.7], btype='band', analog=False, output='zpk')
            elif "Bessel" in filter_type:
                if "LPF" in filter_type:
                    b, a, k = bessel(N=4, Wn=0.5, btype='low', analog=False, output='zpk')
                elif "HPF" in filter_type:
                    b, a, k = bessel(N=4, Wn=0.5, btype='high', analog=False, output='zpk')
                elif "BPF" in filter_type:
                    b, a, k = bessel(N=4, Wn=[0.3, 0.7], btype='band', analog=False, output='zpk')
            elif "Elliptic" in filter_type:
                if "LPF" in filter_type:
                    b, a, k = ellip(N=4, rp=1, rs=40, Wn=0.5, btype='low', analog=False, output='zpk')
                elif "HPF" in filter_type:
                    b, a, k = ellip(N=4, rp=1, rs=40, Wn=0.5, btype='high', analog=False, output='zpk')
                elif "BPF" in filter_type:
                    b, a, k = ellip(N=4, rp=1, rs=40, Wn=[0.3, 0.7], btype='band', analog=False, output='zpk')
            else:
                pass
            self.standard_filters[filter_type] = (b, a)
    def update_custom_apf(self):
            a = self.a_spinbox.value()
            b, a_coeff = self.first_order_all_pass(a)
            self.all_pass_filters["Custom APF"] = (b, a_coeff)
        
    def toggle_a_spinbox(self):
        if self.apf_dropdown.currentText() == "Custom APF":
            self.a_spinbox.setDisabled(False)
        else:
            self.a_spinbox.setDisabled(True)

    def create_all_pass_filter_library(self):
        for APF_filter_type in self.all_pass_filters.keys():
            if "Custom APF" in APF_filter_type:  # First-Order All-Pass Filter
                a = 0.1
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)
            
            elif "All-Pass Filter 1" in APF_filter_type:  # Second-Order All-Pass Filter
                a = 0.2
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)
            
            elif "All-Pass Filter 2" in APF_filter_type:  # Lattice All-Pass Filter
                a = 0.4
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)

            elif "All-Pass Filter 3" in APF_filter_type:  # Third-Order All-Pass Filter
                a = 0.6
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)
            
            elif "All-Pass Filter 4" in APF_filter_type:  # Third-Order All-Pass Filter
                a = 0.8
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)
            
            # elif "Second Order APF" in APF_filter_type:  # Second-Order All-Pass Filter
            #     a = 0.5
            #     b, a_coeff = self.second_order_all_pass(a)
            #     self.all_pass_filters[APF_filter_type] = (b, a_coeff)
            
            # elif "Lattice APF" in APF_filter_type:  # Lattice All-Pass Filter
            #     a_coeffs = [0.7, 0.5]  
            #     b, a_coeff = self.lattice_all_pass(a_coeffs)
            #     self.all_pass_filters[APF_filter_type] = (b, a_coeff)

            # elif "Third Order APF" in APF_filter_type:  # Third-Order All-Pass Filter
            #     a0, a1, a2 = 0.5, 0.7, 0.9 
            #     b, a_coeff = self.third_order_all_pass(a0, a1, a2)
            #     self.all_pass_filters[APF_filter_type] = (b, a_coeff)

    def first_order_all_pass(self, a):
        # First-order All-Pass Filter transfer function: H(z) = (z^-1 - a) / (1 - a * z^-1)
        b = [a, -1]  # Numerator coefficients
        a_coeff = [1, -a]  # Denominator coefficients
        return b, a_coeff

    # def second_order_all_pass(self, a):
    #     # Second-order All-Pass Filter transfer function: H(z) = (z^-2 - 2a * z^-1 + 1) / (z^-2 + 2a * z^-1 + 1)
    #     b = [1, -2*a, 1]
    #     a_coeff = [1, 2*a, 1]
    #     return b, a_coeff

    # def lattice_all_pass(self, a_coeffs):
    #     # Lattice All-Pass Filter: Cascading second-order sections
    #     b_all, a_all = [], []
    #     for a in a_coeffs:
    #         b, a_coeff = self.second_order_all_pass(a)
    #         b_all.extend(b)                 
    #         a_all.extend(a_coeff)
    #     return b_all, a_all

    # def third_order_all_pass(self, a0, a1, a2):
    #     # Transfer function: H(z) = (z^-3 - a2*z^-2 + a1*z^-1 - a0) / (z^-3 + a0*z^-2 + a1*z^-1 + a2)
    #     b = [1, -a2, a1, -a0]
    #     a_coeff = [1, a0, a1, a2]
    #     return b, a_coeff

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

    def select_filter(self):
        filter_type = self.filter_dropdown.currentText()
        if filter_type != "Choose Standard Filter":
            b, a = self.standard_filters[filter_type]
            self.z_plane_canvas.zeros, self.z_plane_canvas.poles = b.tolist(), a.tolist()
            
            self.z_plane_canvas.plot_z_plane()
    def generate_c_code(self):
        b,a = TransferFunctionCanvas.compute_transfer_function(self.transfer_function_canvas, True, self.z_plane_canvas.zeros, self.z_plane_canvas.poles)
        c_template = Template("""
                    #include <stdio.h>

                    #define N {{ num_order }}  // Order of numerator (b coefficients)
                    #define M {{ den_order }}  // Order of denominator (a coefficients)

                    // Filter coefficients
                    double b[N+1] = { {{ b_coeffs }} };  // Numerator coefficients
                    double a[M+1] = { {{ a_coeffs }} };  // Denominator coefficients

                    // Apply filter to input signal
                    void apply_filter(double *input, double *output, int length) {
                        double x[N+1] = {0};  // Delay buffer for input
                        double y[M+1] = {0};  // Delay buffer for output

                        for (int n = 0; n < length; n++) {
                            x[0] = input[n];  // Newest input sample

                            // Compute output using difference equation
                            output[n] = 0;
                            for (int i = 0; i <= N; i++) {
                                output[n] += b[i] * x[i];
                            }
                            for (int j = 1; j <= M; j++) {
                                output[n] -= a[j] * y[j];
                            }

                            // Update delay buffers (shift values)
                            for (int i = N; i > 0; i--) {
                                x[i] = x[i-1];
                            }
                            for (int j = M; j > 0; j--) {
                                y[j] = y[j-1];
                            }

                            y[0] = output[n];  // Store new output sample
                        }
                    }

                    int main() {
                        double input_signal[10] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                        double output_signal[10];

                        apply_filter(input_signal, output_signal, 10);

                        // Print output
                        printf("Filtered Output: ");
                        for (int i = 0; i < 10; i++) {
                            printf("%f ", output_signal[i]);
                        }
                        printf("\\n");

                        return 0;
                    }
                    """)
        c_code = c_template.render(
            num_order=len(b) - 1,
            den_order=len(a) - 1,
            b_coeffs=", ".join(map(str, b)),
            a_coeffs=", ".join(map(str, a))
        )



        print(c_code)

    def update_chosen_apf(self, apf_type):
            apf_type = self.apf_dropdown.currentText()
            if apf_type != "Choose All-Pass Filter":
                b, a = self.all_pass_filters[apf_type]
                apf_zeros, apf_poles, gain = tf2zpk(b, a)
                self.z_plane_canvas.zeros, self.z_plane_canvas.poles = apf_zeros.tolist(), apf_poles.tolist()
                print(f"Gain of this {apf_type} filter is: {gain}")
                self.z_plane_canvas.plot_z_plane()
                
    def select_form(self):
        form_type = self.form_dropdown.currentText()
        if form_type == "Direct Form II":
            self.show_direct_form_II(self.transfer_function_canvas.b_coeffs, self.transfer_function_canvas.a_coeffs)
        else:
            self.show_cascade_form(self.transfer_function_canvas.b_coeffs, self.transfer_function_canvas.a_coeffs)

    def show_direct_form_II(self, b, a):
        b = b.tolist()
        a = a.tolist()        
        if len(b) < len(a):
            for _ in range(len(a) - len(b)):
                b.append(0) 
        elif len(a) < len(b):
            for _ in range(len(b) - len(a)):
                a.append(0)
        order = len(b) - 1 


        for i in range(order):
            if a[i] != 0:
               
                self.ax.arrow(0.6, 0.7 - (i * 0.4), -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                self.ax.text(0.45 , 0.72 - (i * 0.4), f"{a[i]:.2f}", fontsize=12, color="blue")

            if b[i] != 0:
                
                self.ax.arrow(0.6, 0.7 - (i * 0.4), 0.38, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                self.ax.text(0.8 , 0.72 - (i * 0.4), f"{b[i]:.2f}", fontsize=12, color="blue")

            self.ax.arrow(0.6, 0.7 - (i * 0.4), 0, -0.2, head_width=0.02, head_length=0.02, fc="k", ec="k")
            self.ax.text(0.6,  0.5 - (i *  0.4), r"$Z^{-1}$", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="square", facecolor="yellow"))
            self.ax.arrow(0.6, 0.4 - (i * 0.4), 0, -0.08, head_width=0.02, head_length=0.02, fc="k", ec="k")

        if a[order] != 0:
            self.ax.arrow(0.6, 0.7 - (order * 0.4), -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
            self.ax.text(0.45 , 0.72 - (order * 0.4), f"{a[order]:.2f}", fontsize=12, color="blue")

        if b[order] != 0:
            self.ax.arrow(0.6, 0.7 - (order * 0.4), 0.38, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
            self.ax.text(0.8 , 0.72 - (order * 0.4), f"{b[order]:.2f}", fontsize=12, color="blue")

        
        for i in range(order):  
                            
                if a[i + 1] != 0:
                    self.ax.text(0.38 , 0.7 - (i * 0.4), "+", fontsize=10,
                            bbox=dict(boxstyle="circle", facecolor="cyan"))
                    self.ax.arrow(0.4, 0.32 - (i * 0.4), 0, 0.3, head_width=0.02, head_length=0.02, fc="k", ec="k")
                if b[i+1] != 0:
                    self.ax.text(0.99 , 0.7 - (i * 0.4), "+", fontsize=10,
                            bbox=dict(boxstyle="circle", facecolor="cyan"))
                    
                    self.ax.arrow(1, 0.32 - (i * 0.4), 0, 0.3, head_width=0.02, head_length=0.02, fc="k", ec="k")
        self.ax.arrow(0.25, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
        self.ax.arrow(1.24, 0.7, -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
        self.ax.text(0.2, 0.7, "X [n]", fontsize=12, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
        self.ax.text(1.26, 0.7, "Y [n]", fontsize=12, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
        self.ax.set_xlim(-0.1, 1.3)
        self.ax.set_ylim(-1, 1)
        plt.title("Direct Form II Block Diagram")
        self.form_canvas.draw()
       
               
    def show_cascade_form(self, b, a):
        pass




        
    

       

        
        

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
        #print("Current existing zeroes:")
        #print(self.zeros)
        #print("Current existing poles:")
        #print(self.poles)
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
                self.save_state()  # Save state before dragging
                self.update_plot()
                return
            elif distance_pole < 0.1 and distance_pole < distance_zero:
                self.selection_flag = 'pole'
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
        #print(distances)
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
                "o", color="orange", markersize=12, alpha=0.6, zorder=1,
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
        self.b_coeffs = None
        self.a_coeffs = None
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

    def compute_transfer_function(self, c_code, zeros, poles):
        """
        Computes the transfer function H(z) given zeros and poles.
        """
        omega = np.linspace(0, np.pi, 500)  # Frequency range
        z = np.exp(1j * omega)

        # Convert zero-pole representation to transfer function (B(z)/A(z))

        Y_Z = np.ones_like(z, dtype=complex)
        X_Z = np.ones_like(z, dtype=complex)
        for zero in zeros:
            Y_Z *= (z - zero)
        for pole in poles:
            X_Z *= (z - pole)
        H = Y_Z / X_Z
        self.b_coeffs, self.a_coeffs = zpk2tf(zeros, poles, 1)
        print(f"b_coeff: {self.b_coeffs}")
        print(f"a_coeff: {self.a_coeffs}")
        if c_code == True:
           return self.z_plane_canvas.b_coeffs, self.z_plane_canvas.a_coeffs
        else:
           return H, omega

    def update_transfer_function(self, zeros, poles):
        """
        Updates the transfer function plot.
        """
        H, omega = self.compute_transfer_function(False, zeros, poles)
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

            filtered_signal = np.convolve(amplitude, transfer_function_time, mode="full")

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
            input_end_index = min(self.input_current_index + self.temporal_resolution.value(), len(self.data[0]))
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

            filtered_signal = np.convolve(amplitude, transfer_function_time, mode="full")
            self.filtered_plot.plot(time, filtered_signal[self.filtered_current_index:filtered_end_index], pen="r",
                                    clear=False)
            self.filtered_current_index = filtered_end_index
        else:
            self.timer.stop()


    def update_H(self, zeros, poles):
        """Compute the transfer function."""
        transfer_function_freq= self.transfer_function_canvas.compute_transfer_function(False, zeros, poles)[0]
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
