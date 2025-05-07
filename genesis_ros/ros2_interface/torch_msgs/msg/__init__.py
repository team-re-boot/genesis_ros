from dataclasses import dataclass
from pycdr2 import IdlStruct
from pycdr2.types import uint8, int8, int16, int32, int64, float32, float64
from typing import List, Union
import torch


@dataclass
class FP32Tensor(IdlStruct, typename="torch_msgs/msg/FP32Tensor"):  # type: ignore
    is_cuda: bool
    data: List[float32]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return (
                torch.tensor(self.data, dtype=torch.float32).reshape(*self.shape).cuda()
            )
        else:
            return (
                torch.tensor(self.data, dtype=torch.float32).reshape(*self.shape).cpu()
            )


@dataclass
class FP64Tensor(IdlStruct, typename="torch_msgs/msg/FP64Tensor"):  # type: ignore
    is_cuda: bool
    data: List[float64]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return (
                torch.tensor(self.data, dtype=torch.float64).reshape(*self.shape).cuda()
            )
        else:
            return (
                torch.tensor(self.data, dtype=torch.float64).reshape(*self.shape).cpu()
            )


@dataclass
class INT16Tensor(IdlStruct, typename="torch_msgs/msg/INT16Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int16]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return (
                torch.tensor(self.data, dtype=torch.int16).reshape(*self.shape).cuda()
            )
        else:
            return torch.tensor(self.data, dtype=torch.int16).reshape(*self.shape).cpu()


@dataclass
class INT32Tensor(IdlStruct, typename="torch_msgs/msg/INT32Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int32]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return (
                torch.tensor(self.data, dtype=torch.int32).reshape(*self.shape).cuda()
            )
        else:
            return torch.tensor(self.data, dtype=torch.int32).reshape(*self.shape).cpu()


@dataclass
class INT64Tensor(IdlStruct, typename="torch_msgs/msg/INT64Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int64]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return (
                torch.tensor(self.data, dtype=torch.int64).reshape(*self.shape).cuda()
            )
        else:
            return torch.tensor(self.data, dtype=torch.int64).reshape(*self.shape).cpu()


@dataclass
class INT8Tensor(IdlStruct, typename="torch_msgs/msg/INT8Tensor"):  # type: ignore
    is_cuda: bool
    data: List[int8]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return torch.tensor(self.data, dtype=torch.int8).reshape(*self.shape).cuda()
        else:
            return torch.tensor(self.data, dtype=torch.int8).reshape(*self.shape).cpu()


@dataclass
class UINT8Tensor(IdlStruct, typename="torch_msgs/msg/UINT8Tensor"):  # type: ignore
    is_cuda: bool
    data: List[uint8]
    shape: List[int64]

    def to_torch_tensor(self) -> torch.Tensor:
        if self.is_cuda:
            return (
                torch.tensor(self.data, dtype=torch.uint8).reshape(*self.shape).cuda()
            )
        else:
            return torch.tensor(self.data, dtype=torch.uint8).reshape(*self.shape).cpu()


def from_torch_tensor(
    tensor: torch.tensor,
) -> Union[
    FP32Tensor,
    FP64Tensor,
    INT16Tensor,
    INT32Tensor,
    INT64Tensor,
    INT8Tensor,
    UINT8Tensor,
]:
    def is_cuda(tensor: torch.tensor):
        if tensor.device.type == "cpu":
            return False
        if tensor.device.type == "cuda":
            return True
        raise ValueError(
            "Unsupported device type, deice type must be CPU/CUDA. Specified : "
            + tensor.device.type
        )

    if tensor.dtype == torch.float32:
        return FP32Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[float32(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    if tensor.dtype == torch.float64:
        return FP64Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[float64(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    if tensor.dtype == torch.int16:
        return INT16Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[int16(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    if tensor.dtype == torch.int32:
        return INT32Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[int32(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    if tensor.dtype == torch.int64:
        return INT64Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[int64(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    if tensor.dtype == torch.int8:
        return INT8Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[int8(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    if tensor.dtype == torch.uint8:
        return UINT8Tensor(
            is_cuda=is_cuda(tensor=tensor),
            data=[uint8(x) for x in tensor.reshape(tensor.numel()).tolist()],
            shape=list(tensor.shape),
        )
    raise ValueError(
        "Unsupported data type. Supported data type was float32/float64/int16/int32/int64/int8/uint8. Specified: "
        + str(tensor.dtype)
    )
