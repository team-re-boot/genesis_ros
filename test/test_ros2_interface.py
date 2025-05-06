from genesis_ros.ros2_interface import builtin_interfaces, rosgraph_msgs, torch_msgs
import torch

from genesis_ros.topic_interfaces import NopInterface
from genesis_ros.ros2_interface import ROS2Interface
import zenoh
import pytest
import os


def test_nop_interface():
    interface = NopInterface()
    interface.add_publisher("clock", rosgraph_msgs.msg.Clock)
    interface.publish(
        "clock",
        rosgraph_msgs.msg.Clock(clock=builtin_interfaces.msg.Time(sec=1, nanosec=1)),
    )


@pytest.mark.skipif(
    os.environ.get("ROS_DISTRO") is None, reason="ROS 2 was not installed"
)
def test_ros2_interface():
    interface = ROS2Interface(zenoh_config=zenoh.Config())
    interface.add_publisher("clock", rosgraph_msgs.msg.Clock)
    interface.publish(
        "clock",
        rosgraph_msgs.msg.Clock(clock=builtin_interfaces.msg.Time(sec=1, nanosec=1)),
    )
    with pytest.raises(Exception) as excinfo:
        interface.add_publisher("clock", int)
    assert str(excinfo.value) == "Invalid message type, message type is <class 'int'>"


def test_builtin_interfaces_time():
    msg = builtin_interfaces.msg.Time(sec=1, nanosec=1)
    msg.serialize()


def test_rosgraph_msgs_clock():
    msg = rosgraph_msgs.msg.Clock(clock=builtin_interfaces.msg.Time(sec=1, nanosec=1))
    msg.serialize()


def test_torch_msgs_fp32_tensor():
    msg = torch_msgs.msg.FP32Tensor(is_cuda=False, data=[0.0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.float32)
        ),
        torch_msgs.msg.FP32Tensor,
    )


def test_torch_msgs_fp64_tensor():
    msg = torch_msgs.msg.FP64Tensor(is_cuda=False, data=[0.0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.float64)
        ),
        torch_msgs.msg.FP64Tensor,
    )


def test_torch_msgs_int16_tensor():
    msg = torch_msgs.msg.INT16Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.int16)
        ),
        torch_msgs.msg.INT16Tensor,
    )


def test_torch_msgs_int32_tensor():
    msg = torch_msgs.msg.INT32Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.int32)
        ),
        torch_msgs.msg.INT32Tensor,
    )


def test_torch_msgs_int64_tensor():
    msg = torch_msgs.msg.INT64Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.int64)
        ),
        torch_msgs.msg.INT64Tensor,
    )


def test_torch_msgs_int8_tensor():
    msg = torch_msgs.msg.INT8Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.int8)
        ),
        torch_msgs.msg.INT8Tensor,
    )


def test_torch_msgs_uint8_tensor():
    msg = torch_msgs.msg.UINT8Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()
    assert isinstance(
        torch_msgs.msg.from_torch_tensor(
            torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.uint8)
        ),
        torch_msgs.msg.UINT8Tensor,
    )
