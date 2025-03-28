import sys
import math
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget

class JoystickWidget(QWidget):
    commandChanged = pyqtSignal(float, float, float)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(300, 300)
        self.outer_radius = 100
        self.inner_radius = 30
        self.dragging = False
        self.command_x = 0.0
        self.command_z = 0.0
        self.center = QPoint(self.width() // 2, self.height() // 2)
        self.joystick_pos = self.center

    def resizeEvent(self, event):
        self.center = QPoint(self.width() // 2, self.height() // 2)
        if not self.dragging:
            self.joystick_pos = self.center
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(Qt.lightGray))
        painter.drawEllipse(self.center, self.outer_radius, self.outer_radius)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(Qt.darkGray))
        painter.drawEllipse(self.joystick_pos, self.inner_radius, self.inner_radius)

    def mousePressEvent(self, event):
        if (event.pos() - self.center).manhattanLength() <= self.outer_radius:
            self.dragging = True
            self.updateJoystick(event.pos())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.updateJoystick(event.pos())

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.joystick_pos = self.center
        self.command_x = 0.0
        self.command_z = 0.0
        self.commandChanged.emit(self.command_x, 0.0, self.command_z)
        self.update()

    def updateJoystick(self, pos):
        delta = pos - self.center
        distance = math.hypot(delta.x(), delta.y())
        if distance > self.outer_radius:
            factor = self.outer_radius / distance
            delta = QPoint(int(delta.x() * factor), int(delta.y() * factor))
        self.joystick_pos = self.center + delta
        self.command_x = -delta.y() / self.outer_radius * 1.5
        self.command_z = delta.x() / self.outer_radius * -1.0
        self.commandChanged.emit(self.command_x, 0.0, self.command_z)
        self.update()

class CommandVelocityUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Command Velocity Control")
        self.resize(600, 500)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self.text_tab = QWidget()
        self.tabs.addTab(self.text_tab, "Text Mode")
        text_layout = QVBoxLayout(self.text_tab)

        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Enter X velocity")
        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("Enter Y velocity")
        self.z_input = QLineEdit()
        self.z_input.setPlaceholderText("Enter Angular Z velocity")

        text_layout.addWidget(QLabel("X Velocity:"))
        text_layout.addWidget(self.x_input)
        text_layout.addWidget(QLabel("Y Velocity:"))
        text_layout.addWidget(self.y_input)
        text_layout.addWidget(QLabel("Angular Z Velocity:"))
        text_layout.addWidget(self.z_input)

        self.set_button = QPushButton("Set Command Velocity")
        self.set_button.setStyleSheet("font-weight: bold;")

        text_layout.addWidget(self.set_button)
        self.text_command_label = QLabel("Published Command: (0, 0, 0)")
        self.text_command_label.setAlignment(Qt.AlignCenter)
        text_layout.addWidget(self.text_command_label)
        self.joystick_tab = QWidget()

        self.tabs.addTab(self.joystick_tab, "Joystick Mode")
        joystick_layout = QVBoxLayout(self.joystick_tab)
        self.joystick = JoystickWidget()
        joystick_layout.addWidget(self.joystick)
        
        self.joystick_command_label = QLabel("Joystick Command: (0.00, 0.00, 0.00)")
        self.joystick_command_label.setAlignment(Qt.AlignCenter)
        joystick_layout.addWidget(self.joystick_command_label)
        self.joystick.commandChanged.connect(self.updateJoystickCommandLabel)

        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QLabel { font-size: 14px; font-weight: bold; }
            QLineEdit { font-size: 14px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; }
            QPushButton { font-size: 14px; padding: 6px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; }
            QTabWidget::pane { border: 1px solid #ccc; border-radius: 4px; }
            QTabBar::tab { background: #ddd; padding: 10px 25px; font-weight: bold; }
            QTabBar::tab:selected { background: #bbb; }
        """)
    def updateJoystickCommandLabel(self, x, y, z):
        self.joystick_command_label.setText(f"Joystick Command: ({x:.2f}, {y:.2f}, {z:.2f})")

def main():
    app = QApplication(sys.argv)
    window = CommandVelocityUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
