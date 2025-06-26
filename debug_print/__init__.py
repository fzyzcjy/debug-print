from typing import Dict, Optional

import torch
from ._kernels import print_tensor as print_tensor_kernel


class _Buffer:
    def __init__(self, device_index: int):
        self._tensor = torch.zeros((10_000_000,), dtype=torch.char, device=f"cuda:{device_index}")
        self._used_len = 0

    def allocate(self, size: int):
        output = self._tensor[self._used_len: self._used_len + size]
        self._used_len += size
        assert self._used_len <= len(self._tensor)
        return output


class _DebugPrinter:
    def __init__(self):
        # Can be optimized
        self._buffers: Dict[int, _Buffer] = {
            device_index: _Buffer(device_index=device_index)
            for device_index in range(torch.cuda.device_count())
        }

    def __call__(self, x: torch.Tensor, name: str = "", print_ptr: bool = False):
        if len(name) > 0:
            name_bytes = TODO
            name_buffer = self._buffers[x.device].allocate(len(name_bytes) + 1)
            name_buffer.copy_(TODO)


_printer: Optional[_DebugPrinter] = None


def initialize():
    global _printer
    assert _printer is None
    _printer = _DebugPrinter()


def print_tensor(x: torch.Tensor, name: str = "", print_ptr: bool = False):
    _printer(x=x, name=name, print_ptr=print_ptr)
