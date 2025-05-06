from . import builtin_interfaces
from . import rosgraph_msgs
from . import torch_msgs

import zenoh
import pycdr2
from pycdr2 import IdlStruct
from typing import Dict, Any

from genesis_ros.topic_interfaces import TopicInterface


class ROS2Interface(TopicInterface):
    def __init__(self, zenoh_config: zenoh.Config):
        zenoh.init_log_from_env_or("error")
        self.session = zenoh.open(zenoh_config)
        self.publishers: Dict[str, zenoh.Publisher] = {}

    def add_publisher(self, topic_name: str, message_type: Any):
        if not issubclass(message_type, IdlStruct):
            raise Exception(
                "Invalid message type, message type is " + str(message_type)
            )
        self.publishers[topic_name] = self.session.declare_publisher(topic_name)

    def publish(self, topic_name: str, message: Any):
        if not topic_name in self.publishers:
            self.add_publisher(topic_name=topic_name, message_type=type(message))
        self.publishers[topic_name].put(message.serialize())

    def __del__(self):
        self.session.close()
