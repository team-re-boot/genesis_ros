from dataclasses import dataclass
from pycdr2 import IdlStruct
from pycdr2.types import uint8, int8, int16, int32, int64, float32, float64
from typing import List


@dataclass
class FP32Tensor(IdlStruct, typename="torch_msgs/msg/FP32Tensor"):  # type: ignore
    is_cuda: bool
    data: List[float32]
    shape: List[int64]


@dataclass
class FP64Tensor(IdlStruct, typename="torch_msgs/msg/FP64Tensor"):  # type: ignore
    is_cuda: bool
    data: List[float64]
    shape: List[int64]


@dataclass
class INT16Tensor(IdlStruct, typename="torch_msgs/msg/INT16Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int16]
    shape: List[int64]


@dataclass
class INT32Tensor(IdlStruct, typename="torch_msgs/msg/INT32Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int32]
    shape: List[int64]


@dataclass
class INT64Tensor(IdlStruct, typename="torch_msgs/msg/INT64Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int64]
    shape: List[int64]


@dataclass
class INT8Tensor(IdlStruct, typename="torch_msgs/msg/INT8Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int8]
    shape: List[int64]


@dataclass
class UINT8Tensor(IdlStruct, typename="torch_msgs/msg/UINT8Tensor"):  # type: ignore
    is_cuda: bool
    data: List[uint8]
    shape: List[int64]
