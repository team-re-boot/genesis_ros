from genesis_ros.ros2_interface import builtin_interfaces


def test_builtin_interfaces_time():
    msg = builtin_interfaces.msg.Time(sec=1, nanosec=1)
    msg.serialize()
