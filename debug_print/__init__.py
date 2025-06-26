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
    def __init__(self):
        # Can be optimized
        self._buffers: Dict[int, _Buffer] = {
            device_index: _Buffer(device_index=device_index)
            for device_index in range(torch.cuda.device_count())
        }
        self._pending_copy_tasks: List[_CopyTask] = []

    def __call__(self, x: torch.Tensor, name: str = "", print_ptr: bool = False):
        name_buffer_gpu = self._compute_name_buffer_gpu(name=name)
        _print_tensor_kernel(x, name_buffer_gpu, print_ptr)

    def _compute_name_buffer_gpu(self, name: str):
        if len(name) == 0:
            return None

        name_bytes = name.encode("utf-8")
        name_buffer_gpu = self._buffers[x.device.index].allocate(len(name_bytes) + 1)
        name_cpu = torch.tensor(list(name_bytes) + [0], dtype=torch.uint8, device="cpu")
        copy_task = _CopyTask(src=name_cpu, dst=name_buffer_gpu)

        if torch.cuda.is_current_stream_capturing():
            self._pending_copy_tasks.append(copy_task)
        else:
            copy_task.execute()


_printer: Optional[_DebugPrinter] = None


def initialize():
    global _printer
    assert _printer is None
    _printer = _DebugPrinter()


def print_tensor(x: torch.Tensor, name: str = "", print_ptr: bool = False):
    _printer(x=x, name=name, print_ptr=print_ptr)
