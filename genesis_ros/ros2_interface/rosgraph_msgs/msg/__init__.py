from dataclasses import dataclass
from pycdr2 import IdlStruct

from genesis_ros.ros2_interface.builtin_interfaces.msg import Time


@dataclass
class Clock(IdlStruct, typename="rosgraph_msgs/msg/Clock"):  # type: ignore
    clock: Time
