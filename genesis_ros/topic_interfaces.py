from abc import ABCMeta, abstractmethod
from typing import Any, Optional


class TopicInterface(metaclass=ABCMeta):
    @abstractmethod
    def add_publisher(self, topic_name: str, message_type: Any):
        raise NotImplementedError()

    @abstractmethod
    def publish(self, topic_name: str, message: Any):
        raise NotImplementedError()

    @abstractmethod
    def subscribe(self, topic_name: str, message_type: Any):
        raise NotImplementedError()

    @abstractmethod
    def get_subscribed_data(self, topic_name: str) -> Optional[Any]:
        raise NotImplementedError()

    @abstractmethod
    def spin(self, timeout: float = 0.02) -> None:
        raise NotImplementedError()

    @abstractmethod
    def spin_until_subscribe_new_data(self, topic_name: str) -> Any:
        raise NotImplementedError()


class NopInterface(TopicInterface):
    def add_publisher(self, topic_name: str, message_type: Any):
        pass

    def publish(self, topic_name: str, message: Any):
        pass

    def subscribe(self, topic_name: str, message_type: Any):
        pass

    def get_subscribed_data(self, topic_name: str) -> Optional[Any]:
        pass

    def spin(self, timeout: float = 0.02) -> None:
        pass

    def spin_until_subscribe_new_data(self, topic_name: str) -> Any:
        pass
