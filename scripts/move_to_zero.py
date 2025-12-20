"""
move_actuator.py

让电机在运控模式下追踪 sine 曲线

注意：电机会旋转！随时使用 Ctrl + C 结束程序
"""

import time
import argparse

import numpy as np
from loop_rate_limiters import RateLimiter

from robstride_dynamics import RobstrideBus, Motor, ParameterType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", "-c", type=str, default="can1", help="CAN 通道")
    parser.add_argument("--device_id", "-i", type=int, default=0x02, help="电机 CAN ID")
    args = parser.parse_args()

    return args.channel, args.device_id  # CAN 通道, 电机 ID


channel, device_id = parse_args()
print(f"Using CAN channel: {channel}, device ID: {device_id}")

motor_name = "motor"

bus = RobstrideBus(channel=channel, motors={
    motor_name: Motor(id=device_id, model="rs-02"),
})
bus.connect()

rate = RateLimiter(frequency=20.0)

kp = 10.0
kd = 1.0
torque_limit = 0.5

frequency = 1.0  # 频率为 1 Hz
amplitude = 0.1  # 振幅为 1 rad


bus.write(motor_name, ParameterType.TORQUE_LIMIT, torque_limit)
print("torque limit: ", bus.read(motor_name, ParameterType.TORQUE_LIMIT))

# 使能电机
bus.enable(motor_name)

try:
    while True:
        target_angle = np.sin(2 * np.pi * frequency * time.time()) * amplitude

        # 更新电机的目标角度
        bus.write_operation_frame(
            motor_name,
            target_angle,
            kp,
            kd,
        )
        position, velocity, torque, temperature = bus.read_operation_frame(motor_name)
        print(f"position: {position:.3f} \tvelocity: {velocity:.3f} \ttorque: {torque:.3f} \ttemperature: {temperature:.1f}")

        rate.sleep()

except KeyboardInterrupt:
    pass

# 停止电机
bus.disable(motor_name)

bus.disconnect()

print("Program terminated.")
