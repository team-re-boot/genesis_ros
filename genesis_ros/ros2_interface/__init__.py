from . import builtin_interfaces
from . import rosgraph_msgs
from . import torch_msgs

import zenoh
import pycdr2
import time
from pycdr2 import IdlStruct
from typing import Dict, Any, Callable, Optional

from genesis_ros.topic_interfaces import TopicInterface


class ROS2Interface(TopicInterface):
    def __init__(self, zenoh_config: zenoh.Config):
        zenoh.init_log_from_env_or("error")
        self.session = zenoh.open(zenoh_config)
        self.publishers: Dict[str, zenoh.Publisher] = {}
        self.subscribers: Dict[str, zenoh.subscriber] = {}
        self.subscribed_data: Dict[str, Optional[bytes]] = {}
        self.subscribed_data_type: Dict[str, Any] = {}

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

    def subscribe(self, topic_name: str, message_type: Any):
        if not issubclass(message_type, IdlStruct):
            raise Exception(
                "Invalid message type, message type is " + str(message_type)
            )

        def callback(data: Any):
            self.subscribed_data[topic_name] = data.payload.to_bytes()

        self.subscribed_data[topic_name] = None
        self.subscribed_data_type[topic_name] = message_type
        self.subscribers[topic_name] = self.session.declare_subscriber(
            topic_name, callback
        )

    def get_subscribed_data(self, topic_name: str) -> Optional[Any]:
        if self.subscribed_data[topic_name]:
            return self.subscribed_data_type[topic_name].deserialize(
                self.subscribed_data[topic_name]
            )
        else:
            return None

    def spin(self, timeout: float = 0.02) -> None:
        time.sleep(timeout)

    def __del__(self):
        self.session.close()
