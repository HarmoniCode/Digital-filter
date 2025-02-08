import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QSplitter, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSpacerItem, QSizePolicy, QMessageBox, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QLocale
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import butter, cheby1, bessel, ellip, tf2zpk, zpk2tf, zpk2sos
from jinja2 import Template
import csv
from itertools import zip_longest
from ZPlaneCanvas import ZPlaneCanvas
from TransferFunctionCanvas import TransferFunctionCanvas

class ZPlanePlotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.gain = 1
        self.apf_poles = []
        self.apf_zeros = []
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

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

        self.main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter)

        right_frame = QFrame()
        right_frame.setObjectName("right_frame")
        right_layout = QVBoxLayout(right_frame)

        upper_layout = QHBoxLayout()
        right_layout.addLayout(upper_layout)

        self.z_plane_canvas = ZPlaneCanvas()
        self.selected_conjugate = self.z_plane_canvas.selected_conjugate
        # left_layout.addWidget(NavigationToolbar(self.z_plane_canvas, self))

        self.form_widget = QFrame()
        self.form_layout = QVBoxLayout(self.form_widget)
        self.form_canvas = FigureCanvas(self.fig)
        self.form_layout.addWidget(self.form_canvas)

        bottom_splitter = QSplitter(Qt.Vertical)

        bottom_splitter.addWidget(self.form_widget)

        self.coord_layout = QHBoxLayout()
        self.coord_layout.setSpacing(5)
        self.coord_label = QLabel("Coordinates (Real, Imaginary):")
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

        upper_left_frame = QFrame()
        upper_left_frame.setObjectName("upper_left_frame")
        upper_left_layout = QVBoxLayout(upper_left_frame)
        upper_left_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        upper_left_layout.setSpacing(20)

        upper_layout.addWidget(upper_left_frame)
        upper_layout.addWidget(self.z_plane_canvas)

        upper_left_layout.addLayout(self.coord_layout)

        buttons_frame = QFrame()
        buttons_frame.setObjectName("buttons_frame")

        self.buttons_layout = QVBoxLayout(buttons_frame)
        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.buttons_layout.setSpacing(20)

        # self.button_layout = QHBoxLayout()

        self.add_zero_button = QPushButton("Add Zero")
        self.add_zero_button.clicked.connect(self.add_zero)
        self.buttons_layout.addWidget(self.add_zero_button)

        self.add_pole_button = QPushButton("Add Pole")
        self.add_pole_button.clicked.connect(self.add_pole)
        self.buttons_layout.addWidget(self.add_pole_button)

        self.add_conjugate_button = QPushButton("Add Conjugate")
        self.add_conjugate_button.clicked.connect(self.z_plane_canvas.add_conjugate)
        self.buttons_layout.addWidget(self.add_conjugate_button)
        self.add_conjugate_button.setDisabled(True)

        self.clear_zeros_button = QPushButton("Clear Zeros")
        self.clear_zeros_button.clicked.connect(self.z_plane_canvas.clear_zeros)
        self.buttons_layout.addWidget(self.clear_zeros_button)

        self.clear_poles_button = QPushButton("Clear Poles")
        self.clear_poles_button.clicked.connect(self.z_plane_canvas.clear_poles)
        self.buttons_layout.addWidget(self.clear_poles_button)

        self.button_layout_2 = QHBoxLayout()

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.z_plane_canvas.clear_all)
        self.buttons_layout.addWidget(self.clear_all_button)

        self.switch_button = QPushButton("Switch")
        self.switch_button.clicked.connect(self.switch_zeros_poles)
        self.buttons_layout.addWidget(self.switch_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.z_plane_canvas.undo)
        self.buttons_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.z_plane_canvas.redo)
        self.buttons_layout.addWidget(self.redo_button)

        self.save_csv_button = QPushButton("Save")
        self.save_csv_button.clicked.connect(self.z_plane_canvas.save_state_to_csv)
        self.buttons_layout.addWidget(self.save_csv_button)

        self.load_csv_button = QPushButton("Load ")
        self.load_csv_button.clicked.connect(self.z_plane_canvas.load_state_from_csv)
        self.buttons_layout.addWidget(self.load_csv_button)

        self.generate_code_button = QPushButton("C Code")
        self.generate_code_button.clicked.connect(self.generate_c_code)
        self.buttons_layout.addWidget(self.generate_code_button)

        self.export_form_button = QPushButton("export Form")
        self.export_form_button.clicked.connect(self.export_form)
        self.buttons_layout.addWidget(self.export_form_button)

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
        upper_left_layout.addWidget(self.filter_dropdown)

        apf_frame=QFrame()
        apf_layout=QVBoxLayout(apf_frame)
        apf_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        
        self.apf_checkboxes = []
        custom_apf_checkbox = QCheckBox("Custom APF")
        custom_apf_checkbox.stateChanged.connect(self.toggle_a_spinbox)
        self.apf_checkboxes.append(custom_apf_checkbox)
        apf_layout.addWidget(custom_apf_checkbox)
        for apf_name in list(self.all_pass_filters.keys())[1:]:
            checkbox = QCheckBox(apf_name)
            checkbox.stateChanged.connect(self.update_chosen_apf)
            self.apf_checkboxes.append(checkbox)
            apf_layout.addWidget(checkbox)
            
        upper_left_layout.addWidget(apf_frame)

        self.a_spinbox = QDoubleSpinBox()
        self.a_spinbox.setDisabled(True)
        self.a_spinbox.setLocale(QLocale(QLocale.Language.English))

        self.a_spinbox.setRange(0.1, 0.9)  # Set the range of values (min, max)
        self.a_spinbox.setValue(0.1)  # Set the initial value
        self.a_spinbox.setSingleStep(0.1)
        self.a_spinbox.valueChanged.connect(self.update_custom_apf)  # Connect signal to slot
        upper_left_layout.addWidget(self.a_spinbox)

        self.form_dropdown = QComboBox()
        self.form_dropdown.addItems([
            "Direct Form II", "Cascade Form"
        ])
        self.form_dropdown.currentIndexChanged.connect(self.select_form)

        upper_left_layout.addWidget(self.form_dropdown)

        # right_pane = QWidget()
        # right_layout = QVBoxLayout(right_pane)

        self.transfer_function_canvas = TransferFunctionCanvas()
        # local_left_layout.addWidget(NavigationToolbar(self.transfer_function_canvas, self))
        bottom_splitter.addWidget(self.transfer_function_canvas)
        right_layout.addWidget(bottom_splitter)

        splitter.addWidget(buttons_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([400, 8000])

        self.z_plane_canvas.transfer_function_updated.connect(
            self.transfer_function_canvas.update_transfer_function
        )
        self.create_standard_filter_library()
        self.create_all_pass_filter_library()

    def clear_form_widget(self):
        
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        self.form_canvas = FigureCanvas(self.fig)
        self.form_layout.addWidget(self.form_canvas)
        self.form_canvas.draw()

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
        b_coeff, a_coeff = self.first_order_all_pass(a)
        self.all_pass_filters["Custom APF"] = (b_coeff, a_coeff)
        self.update_chosen_apf()

    def toggle_a_spinbox(self):
        if  self.apf_checkboxes[0].isChecked():
            self.a_spinbox.setDisabled(False)
        else:
            self.a_spinbox.setDisabled(True)
        self.update_chosen_apf()
        
    def create_all_pass_filter_library(self):
        for APF_filter_type in self.all_pass_filters.keys():
            if "Custom APF" in APF_filter_type:  
                a = 0.1
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)

            elif "All-Pass Filter 1" in APF_filter_type:  
                a = 0.2
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)

            elif "All-Pass Filter 2" in APF_filter_type:  
                a = 0.4
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)

            elif "All-Pass Filter 3" in APF_filter_type:  
                a = 0.6
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)

            elif "All-Pass Filter 4" in APF_filter_type:  
                a = 0.8
                b, a_coeff = self.first_order_all_pass(a)
                self.all_pass_filters[APF_filter_type] = (b, a_coeff)

    def first_order_all_pass(self, a):
        # First-order All-Pass Filter transfer function: H(z) = (z^-1 - a) / (1 - a * z^-1)
        b = [a, 1]  # Numerator coefficients
        a_coeff = [1, a]  # Denominator coefficients
        return b, a_coeff

    
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

    def plot_z_plane(self):
        self.z_plane_canvas.plot_z_plane(self.z_plane_canvas.zeros, self.z_plane_canvas.poles)
        self.clear_form_widget()
        self.select_form()

    def select_filter(self):
        ZPlaneCanvas.save_state(self.z_plane_canvas)
        filter_type = self.filter_dropdown.currentText()
        if filter_type != "Choose Standard Filter":
            b, a = self.standard_filters[filter_type]
            self.z_plane_canvas.zeros, self.z_plane_canvas.poles = b.tolist(), a.tolist()
            self.plot_z_plane()

    def generate_c_code(self):
        b, a = TransferFunctionCanvas.compute_transfer_function(
            self.transfer_function_canvas, True, self.z_plane_canvas.zeros, self.z_plane_canvas.poles
        )
        c_template = Template("""
        #include <stdio.h>

        #define N {{ num_order }}  
        #define M {{ den_order }}  

        
        double b[N+1] = { {{ b_coeffs }} };  
        double a[M+1] = { {{ a_coeffs }} };  

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

        # Write the generated code to a .c file
        file_name = "filter_design.c"
        with open(file_name, "w") as c_file:
            c_file.write(c_code)
        print(f"C code has been generated and saved to {file_name}.")

        # Display the code in a pop-up window
        msg_box = QMessageBox()
        msg_box.setSizeGripEnabled(True)
        msg_box.setStyleSheet("QLabel{min-width: 700px;}")
        msg_box.setWindowTitle("Generated C Code")
        msg_box.setText("The C code has been generated successfully.")

        # Create a QTextEdit for detailed text
        text_edit = QTextEdit()
        text_edit.setPlainText(c_code)
        text_edit.setMinimumHeight(300)
        text_edit.setReadOnly(True)

        # Add the QTextEdit to the message box layout
        layout = msg_box.layout()
        layout.addWidget(text_edit, layout.rowCount(), 0, 1, layout.columnCount())

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def update_chosen_apf(self):
        self.apf_zeros = []
        self.apf_poles = []
        self.gain = 1

        for checkbox in self.apf_checkboxes:
            if checkbox.isChecked():
                apf_type = checkbox.text()
                b, a = self.all_pass_filters[apf_type]
                zeros, poles, system_gain = tf2zpk(b, a)
                self.apf_zeros.extend(zeros)
                self.apf_poles.extend(poles)
                self.gain *= system_gain

        combined_zeros = self.z_plane_canvas.zeros + self.apf_zeros
        combined_poles = self.z_plane_canvas.poles + self.apf_poles
        self.z_plane_canvas.plot_z_plane(combined_zeros, combined_poles)
        self.transfer_function_canvas.update_transfer_function(combined_zeros, combined_poles, self.gain)

    def export_form(self):  
        form_type = self.form_dropdown.currentText()
        file_path = f"{form_type} Diagram.png"
        if file_path:
            self.fig.savefig(file_path)
        
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Export Form")
        msg_box.setText(f"The {form_type} diagram has been exported successfully.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        
    def select_form(self):
        self.clear_form_widget()
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
            # arrow segment and coefficient for input part
            if (a[i]) != 0:
                self.ax.arrow(0.6, 0.7 - (i * 0.4), -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                self.ax.text(0.45 , 0.72 - (i * 0.4), f"{a[i]:.2f}", fontsize=10, color="blue")

            # arrow segment and coefficient for output part
            if (b[i]) != 0:
                self.ax.arrow(0.6, 0.7 - (i * 0.4), 0.38, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                self.ax.text(0.8 , 0.72 - (i * 0.4), f"{b[i]:.2f}", fontsize=10, color="blue")
            self.ax.arrow(0.6, 0.7 - (i * 0.4), 0, -0.2, head_width=0.02, head_length=0.02, fc="k", ec="k")
    
            # delay elements
            self.ax.text(0.6,  0.5 - (i *  0.4), r"$Z^{-1}$", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="square", facecolor="yellow"))

            # arrow segment after delay element
            self.ax.arrow(0.6, 0.4 - (i * 0.4), 0, -0.08, head_width=0.02, head_length=0.02, fc="k", ec="k")

        # arrow and coeffiecient for the last element of input part
        if a[order] != 0:
            self.ax.arrow(0.6, 0.7 - (order * 0.4), -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
            self.ax.text(0.45 , 0.72 - (order * 0.4), f"{a[order]:.2f}", fontsize=10, color="blue")

        # arrow and coeffiecient for the last element of output part
        if b[order] != 0:
            self.ax.arrow(0.6, 0.7 - (order * 0.4), 0.38, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
            self.ax.text(0.8 , 0.72 - (order * 0.4), f"{b[order]:.2f}", fontsize=10, color="blue")

        
        for i in range(order):  
            # summation nodes and arrow segments for input part                
                if a[i + 1] != 0:
                    self.ax.text(0.38 , 0.7 - (i * 0.4), "+", fontsize=10,
                            bbox=dict(boxstyle="circle", facecolor="cyan"))
                    self.ax.arrow(0.4, 0.32 - (i * 0.4), 0, 0.3, head_width=0.02, head_length=0.02, fc="k", ec="k")
                # summation nodes and arrow segments for input part                
                if b[i+1] != 0:
                    self.ax.text(0.99 , 0.7 - (i * 0.4), "+", fontsize=10,
                            bbox=dict(boxstyle="circle", facecolor="cyan"))
                    
                    self.ax.arrow(1, 0.32 - (i * 0.4), 0, 0.3, head_width=0.02, head_length=0.02, fc="k", ec="k")
        # input arrow segment 
        self.ax.arrow(0.25, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
        # output arrow segment 
        self.ax.arrow(1, 0.7, 0.13, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")

        self.ax.text(0.18, 0.7, "X [n]", fontsize=10, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))

        self.ax.text(1.22, 0.7, "Y [n]", fontsize=10, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
        # self.ax.set_xlim(-0.1, 1.3)
        self.ax.set_ylim(-1, 1)
        self.ax.set_title("Direct Form II Block Diagram")
        self.form_canvas.draw()
        
    def show_cascade_form(self, b, a):
        order = max(len(b.tolist()), len(a.tolist())) - 1
        sos = zpk2sos(*tf2zpk(b, a))

        for index, section in enumerate(sos):
            if order > 0 :
                for i in range(2):
            # arrow segment and coefficient for input part
                    if section[i+3] != 0:
                        self.ax.arrow(0.6 + (index * 0.68), 0.7 - (i * 0.4), -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                        self.ax.text(0.45 + (index * 0.68) , 0.72 - (i * 0.4), f"{section[i+3]:.2f}", fontsize=10, color="blue")

                    # arrow segment and coefficient for output part
                    if section[i] != 0:
                        
                        self.ax.arrow(0.6 + (index * 0.68), 0.7 - (i * 0.4), 0.38, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                        self.ax.text(0.8 + (index * 0.68) , 0.72 - (i * 0.4), f"{section[i]:.2f}", fontsize=10, color="blue")
                    self.ax.arrow(0.6 + (index * 0.68), 0.7 - (i * 0.4), 0, -0.2, head_width=0.02, head_length=0.02, fc="k", ec="k")
            
                    # delay elements
                    self.ax.text(0.6 + (index * 0.68),  0.5 - (i *  0.4), r"$Z^{-1}$", fontsize=9, ha="center", va="center",
                            bbox=dict(boxstyle="square", facecolor="yellow"))
        
                    # arrow segment after delay element
                    self.ax.arrow(0.6 + (index * 0.68), 0.4 - (i * 0.4), 0, -0.08, head_width=0.02, head_length=0.02, fc="k", ec="k")
        
                # arrow and coeffiecient for the last element of input part
                if section[5] != 0:
                    self.ax.arrow(0.6 + (index * 0.68), 0.7 - (2 * 0.4), -0.18, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                    self.ax.text(0.45 + (index * 0.68) , 0.72 - (2 * 0.4), f"{section[5]:.2f}", fontsize=10, color="blue")

                # arrow and coeffiecient for the last element of output part

                if section[2] != 0:
                    self.ax.arrow(0.6 + (index * 0.68), 0.7 - (2 * 0.4), 0.38, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
                    self.ax.text(0.8 + (index * 0.68) , 0.72 - (2 * 0.4), f"{section[2]:.2f}", fontsize=10, color="blue")

                
                for i in range(2):  
                
                    # summation nodes and arrow segments for input part                
                        if section[i + 4] != 0:
                            self.ax.text(0.38 + (index * 0.68) , 0.7 - (i * 0.4), "+", fontsize=10,
                                    bbox=dict(boxstyle="circle", facecolor="cyan"))
                            self.ax.arrow(0.4 + (index * 0.68), 0.32 - (i * 0.4), 0, 0.3, head_width=0.02, head_length=0.02, fc="k", ec="k")
                    
                    # summation nodes and arrow segments for output part                
                        if section[i+1] != 0:
                            self.ax.text(0.99 + (index * 0.68) , 0.7 - (i * 0.4), "+", fontsize=10,
                                    bbox=dict(boxstyle="circle", facecolor="cyan"))
                            
                            self.ax.arrow(1 + (index * 0.68), 0.32 - (i * 0.4), 0, 0.3, head_width=0.02, head_length=0.02, fc="k", ec="k")
                order -= 2
                self.ax.arrow( 1 + (index * 0.68), 0.7, 0.05, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")
        
        # input arrow
        self.ax.text(0.2, 0.7, "X [n]", fontsize=10, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
        self.ax.arrow(0.25, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, fc="k", ec="k")

        self.ax.text(1 + ((sos.shape[0] - 1) * 0.8), 0.7, "Y [n]", fontsize=10, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
        self.ax.set_ylim(-1, 1)
        self.ax.set_title("Cascade Form Block Diagram")
        self.form_canvas.draw()