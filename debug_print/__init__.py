from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
from ._kernels import print_tensor as _print_tensor_kernel


class _Buffer:
    def __init__(self, device_index: int):
        self._tensor = torch.zeros((10_000_000,), dtype=torch.uint8, device=f"cuda:{device_index}")
        self._used_len = 0

    def allocate(self, size: int):
        output = self._tensor[self._used_len: self._used_len + size]
        self._used_len += size
        assert self._used_len <= len(self._tensor)
        return output


@dataclass
class _CopyTask:
    src: torch.Tensor
    dst: torch.Tensor

    def execute(self):
        self.dst.copy_(self.src)


class _DebugPrinter:
    def __init__(self, device_id: Optional[int]):
        if device_id is None:
            device_id = torch.cuda.current_device()

        # Can be optimized
        self._buffers: Dict[int, _Buffer] = {device_id: _Buffer(device_index=device_id)}
        self._pending_copy_tasks: List[_CopyTask] = []

    def post_initialize(self):
        for copy_task in self._pending_copy_tasks:
            copy_task.execute()
        self._pending_copy_tasks.clear()

    def __call__(self, x: torch.Tensor, name: str, print_ptr: bool, print_shape: bool):
        assert x.is_cuda, f"{x.device} must be on cuda"
        name_buffer_gpu = self._compute_name_buffer_gpu(name=name, device_index=x.device.index)
        _print_tensor_kernel(x, name_buffer_gpu, print_ptr, print_shape)

    def _compute_name_buffer_gpu(self, name: str, device_index: int):
        if len(name) == 0:
            return None

        name_bytes = name.encode("utf-8")
        name_buffer_gpu = self._buffers[device_index].allocate(len(name_bytes) + 1)
        name_cpu = torch.tensor(list(name_bytes) + [0], dtype=torch.uint8, device="cpu")
        copy_task = _CopyTask(src=name_cpu, dst=name_buffer_gpu)

        if torch.cuda.is_current_stream_capturing():
            self._pending_copy_tasks.append(copy_task)
        else:
            copy_task.execute()

        return name_buffer_gpu


_printer: Optional[_DebugPrinter] = None


def initialize(device_id: int):
    global _printer
    if _printer is not None:
        print("debug_print.initialize skip since already initialized")
        return
    _printer = _DebugPrinter(device_id=device_id)


def post_initialize():
    _printer.post_initialize()


def print_tensor(x: torch.Tensor, name: str = "", print_ptr: bool = True, print_shape: bool = True):
    _printer(x=x, name=name, print_ptr=print_ptr, print_shape=print_shape)
