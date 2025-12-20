"""
read_parameters.py

读取电机的参数
"""

import argparse

from robstride_dynamics import ParameterType, RobstrideBus, Motor


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


response = bus.read(motor_name, ParameterType.VBUS)
print(f"VBUS: {response}")

response = bus.read(motor_name, ParameterType.POSITION_KP)
print(f"Position KP: {response}")

response = bus.read(motor_name, ParameterType.CURRENT_LIMIT)
print(f"Current Limit: {response}")

response = bus.read(motor_name, ParameterType.ZERO_STATE)
print(f"Zero mode: {response}")

bus.read_id(motor_name)

bus.disconnect()

print("Program terminated.")
