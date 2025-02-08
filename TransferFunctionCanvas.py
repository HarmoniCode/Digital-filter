import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import zpk2tf
from ZPlaneCanvas import ZPlaneCanvas


class TransferFunctionCanvas(FigureCanvas):
    def __init__(self):
        self.figure, (self.ax_mag, self.ax_phase) = plt.subplots(2, 1, figsize=(6, 6))
        self.figure.subplots_adjust(
            hspace=0.5
        )  # Adjust this value to increase vertical space
        super().__init__(self.figure)
        self.plot_initial()
        self.z_plane_canvas = ZPlaneCanvas()
        self.b_coeffs = None
        self.a_coeffs = None

    def plot_initial(self):
        self.ax_mag.set_title("Magnitude of Transfer Function")

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
            Y_Z *= z - zero
        for pole in poles:
            X_Z *= z - pole
        H = Y_Z / X_Z
        self.b_coeffs, self.a_coeffs = zpk2tf(zeros, poles, 1)
        if c_code == True:
            return self.b_coeffs, self.a_coeffs
        else:
            return H, omega

    def update_transfer_function(self, zeros, poles, gain=1):
        """
        Updates the transfer function plot.
        """
        H, omega = self.compute_transfer_function(False, zeros, poles)
        magnitude = np.abs(gain) * np.abs(H)
        phase = np.angle(H)

        self.ax_mag.clear()
        self.ax_phase.clear()

        self.ax_mag.plot(omega, magnitude, label="|H(z)|", color="blue")
        self.ax_mag.set_title("Magnitude of Transfer Function")

        self.ax_mag.set_ylabel("Magnitude")
        self.ax_mag.grid(True)

        self.ax_phase.plot(omega, phase, label="∠H(z)", color="orange")
        self.ax_phase.set_title("Phase of Transfer Function")
        self.ax_phase.set_xlabel("Frequency (rad/s)")
        self.ax_phase.set_ylabel("Phase (radians)")
        self.ax_phase.grid(True)
        self.draw()
