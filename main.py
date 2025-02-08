import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter
)
from PyQt5.QtCore import Qt


from GraphsWindow import GraphsWindow
from ZPlanePlotApp import ZPlanePlotApp

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Filter Designer")
        self.setGeometry(100, 100, 1600, 900)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.z_plane_plot_app = ZPlanePlotApp()
        self.graphs_window = GraphsWindow(self.z_plane_plot_app.z_plane_canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("splitter")
        splitter.addWidget(self.z_plane_plot_app)
        splitter.addWidget(self.graphs_window)

        self.main_layout.addWidget(splitter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open("styles/index.qss", "r") as file:
        app.setStyleSheet(file.read())
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())