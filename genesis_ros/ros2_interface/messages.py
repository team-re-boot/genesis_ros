from dataclasses import dataclass
from pycdr2 import IdlStruct
from pycdr2.types import int8, int32, uint32, float64


@dataclass
class Time(IdlStruct, typename="Time"):  # type: ignore
    sec: int32
    nanosec: uint32


@dataclass
class Clock(IdlStruct, typename="rosgraph_msgs/msg/Clock"):  # type: ignore
    clock: Time
