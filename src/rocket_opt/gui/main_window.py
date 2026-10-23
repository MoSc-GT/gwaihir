import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Rocket Opt")
    window.setCentralWidget(QLabel("Raketen-Optimalsteuerung"))
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
