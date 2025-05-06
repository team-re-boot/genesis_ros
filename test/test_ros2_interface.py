from genesis_ros.ros2_interface import builtin_interfaces, rosgraph_msgs, torch_msgs


def test_builtin_interfaces_time():
    msg = builtin_interfaces.msg.Time(sec=1, nanosec=1)
    msg.serialize()


def test_rosgraph_msgs_clock():
    msg = rosgraph_msgs.msg.Clock(clock=builtin_interfaces.msg.Time(sec=1, nanosec=1))
    msg.serialize()


def test_torch_msgs_fp32_tensor():
    msg = torch_msgs.msg.FP32Tensor(is_cuda=False, data=[0.0], shape=[1])
    msg.serialize()


def test_torch_msgs_fp64_tensor():
    msg = torch_msgs.msg.FP64Tensor(is_cuda=False, data=[0.0], shape=[1])
    msg.serialize()


def test_torch_msgs_int16_tensor():
    msg = torch_msgs.msg.INT16Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()


def test_torch_msgs_int32_tensor():
    msg = torch_msgs.msg.INT32Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()


def test_torch_msgs_int64_tensor():
    msg = torch_msgs.msg.INT64Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()


def test_torch_msgs_int8_tensor():
    msg = torch_msgs.msg.INT8Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()


def test_torch_msgs_uint8_tensor():
    msg = torch_msgs.msg.UINT8Tensor(is_cuda=False, data=[0], shape=[1])
    msg.serialize()
