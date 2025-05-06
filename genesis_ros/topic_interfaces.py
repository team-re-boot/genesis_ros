from abc import ABCMeta, abstractmethod
from typing import Any


class TopicInterface(metaclass=ABCMeta):
    @abstractmethod
    def add_publisher(self, topic_name: str, message_type: Any):
        raise NotImplementedError()

    @abstractmethod
    def publish(self, topic_name: str, message: Any):
        raise NotImplementedError()


class NopInterface(TopicInterface):
    def add_publisher(self, topic_name: str, message_type: Any):
        pass

    def publish(self, topic_name: str, message: Any):
        pass
