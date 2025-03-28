import sys
import rclpy
from rclpy.node import Node
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from geometry_msgs.msg import PoseStamped

from .submodule.robot_UI import CommandVelocityUI


class UIController(Node):
    def __init__(self):
        # ROS2 Node 초기화
        super().__init__('velocity_command_ui')
        # UI 디자인 클래스 인스턴스 생성
        self.ui = CommandVelocityUI()
        
        # /robot_velocity_command 토픽으로 PoseStamped 메시지를 발행하는 publisher 생성
        self.velocity_pub = self.create_publisher(PoseStamped, '/robot_velocity_command', 10)
        
        # UI 디자인에서 set_button과 조이스틱의 commandChanged 시그널을 연결합니다.
        self.ui.set_button.clicked.connect(self.publishTextCommand)
        self.ui.joystick.commandChanged.connect(self.publishJoystickCommand)
        self.ui.joystick.commandChanged.connect(self.updateJoystickCommandLabel)
        
        # ROS spin과 UI 이벤트를 동시에 처리하기 위한 QTimer 설정
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: rclpy.spin_once(self, timeout_sec=0.1))
        self.timer.start(100)

    def publishTextCommand(self):
        # 텍스트 입력란에서 x, y, z 값을 읽어 PoseStamped 메시지로 발행합니다.
        try:
            x = float(self.ui.x_input.text())
            y = float(self.ui.y_input.text())
            z = float(self.ui.z_input.text())
        except ValueError:
            self.ui.text_command_label.setText("Invalid input!")
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        self.velocity_pub.publish(msg)
        self.ui.text_command_label.setText(f"Published Command: ({x}, {y}, {z})")

    def publishJoystickCommand(self, x, y, z):
        # 조이스틱 입력값을 바로 PoseStamped 메시지로 발행합니다.
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        self.velocity_pub.publish(msg)

    def updateJoystickCommandLabel(self, x, y, z):
        # 조이스틱에서 발생한 입력값을 UI의 레이블에 업데이트합니다.
        self.ui.joystick_command_label.setText(f"Joystick Command: ({x:.2f}, {y:.2f}, {z:.2f})")


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    ui_controller = UIController()
    ui_controller.ui.show()  # robot_UI.py에 정의된 UI 창 표시
    exit_code = app.exec_()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
