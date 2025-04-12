from amber_mcap.dataset.schema import ImageMessageSchema # type: ignore
from amber_mcap.dataset.schema import TFMessageSchema
from amber_mcap.dataset.conversion import build_message_from_tf # type: ignore
from amber_mcap.dataset.conversion import build_message_from_image
from amber_mcap.tf2_amber import TransformStamped, Header # type: ignore
from amber_mcap.unit.time import Time, TimeUnit # type: ignore
from mcap_ros2.writer import Writer as McapWriter
import math
import numpy as np
from typing import Callable


class RosbagWriter:
    def __init__(self, mcap_path="output.mcap"):
        self.mcap_path = mcap_path
        self.file = open(self.mcap_path, "wb")
        self.writer = McapWriter(self.file)
        self.tf_schema = self.writer.register_msgdef(
            TFMessageSchema.name, TFMessageSchema.schema_text
        )
        self.image_schema = self.writer.register_msgdef(
            ImageMessageSchema.name, ImageMessageSchema.schema_text
        )
        self.tf_index = 0
        self.image_index = 0

    # self.writer.finish throw error when we use it in destructor.
    def finish(self):
        self.writer.finish()
        self.file.close()

    def write_image(self, topic: str, image: np.ndarray, header: Header):
        nanoseconds = int(header.stamp.sec * math.pow(10, 9) + header.stamp.nanosec)
        self.writer.write_message(
            topic=topic,
            schema=self.image_schema,
            message=build_message_from_image(
                image,
                header.frame_id,
                Time(
                    nanoseconds,
                    TimeUnit.NANOSECOND,
                ),
                "rgb8",
            ),
            log_time=nanoseconds,
            publish_time=nanoseconds,
            sequence=self.image_index,
        )
        self.image_index = self.image_index + 1

    def write_tf(self, transform: TransformStamped):
        nanoseconds = int(
            transform.header.stamp.sec * math.pow(10, 9)
            + transform.header.stamp.nanosec
        )
        self.writer.write_message(
            topic="/tf",
            schema=self.tf_schema,
            message=build_message_from_tf([transform]),
            log_time=nanoseconds,
            publish_time=nanoseconds,
            sequence=self.tf_index,
        )
        self.tf_index = self.tf_index + 1
