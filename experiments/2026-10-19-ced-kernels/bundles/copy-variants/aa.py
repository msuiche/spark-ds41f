"""A/A calibration of the paired copy timing: Triton against itself, with separate and shared outputs,
and the candidate variants against Triton with a shared output."""
import math

import torch

import bench
from benchmarks.kernels.ced_reference import _copy_rows
from vllm.models.deepseek_v4_1.tilelang import ced
from vllm.triton_utils import triton

SHAPES = [
    (48, 48, 0, torch.int32, ()),
    (48, 48, 0, torch.bool, ()),
    (4048, 128, 0, torch.bool, ()),
    (4048, 128, 64, torch.int32, ()),
    (4048, 128, 64, torch.bool, ()),
    (4048, 128, 0, torch.int64, ()),
    (4048, 128, 0, torch.bfloat16, (5120,)),
]


def triton_copy(source, indices, output, length, rows, tail, width):
    return lambda: _copy_rows[(triton.cdiv(rows * width, 512),)](
        source, indices, output, length, rows, tail, source.stride(), width, False, 512)


for repeat in range(2):
    for mode in ("aa-separate", "aa-shared", "safe-shared", "ldg-shared", "ca-shared"):
        for length, retained, padding, dtype, tail in SHAPES:
            if tail and mode in ("ldg-shared", "ca-shared"):
                continue
            rows = retained + padding
            source = torch.ones((length, *tail), dtype=dtype, device="cuda")
            indices = torch.cat([torch.arange(length - retained, length, device="cuda"),
                                 torch.full((padding,), -1, device="cuda")])
            outputs = [torch.zeros((rows, *tail), dtype=dtype, device="cuda") for _ in range(2)]
            width = math.prod(tail)
            reference = triton_copy(source, indices, outputs[0], length, rows, tail, width)
            target = outputs[1] if mode == "aa-separate" else outputs[0]
            if mode.startswith("aa"):
                candidate = triton_copy(source, indices, target, length, rows, tail, width)
            else:
                loads = mode.split("-")[0]
                candidate = lambda: ced.copy_rows(source, indices, target, False, 16, loads)
            # Exactness is checked on separate outputs before timing.
            check = torch.zeros_like(outputs[0])
            reference()
            ced.copy_rows(source, indices, check, False, 16, "safe" if mode.startswith("aa") else mode.split("-")[0])
            assert torch.equal(outputs[0], check)
            bench.paired("copy", dict(mode=mode, repeat=repeat, rows=length, retained=retained, padding=padding,
                                      dtype=str(dtype), tail=tail), reference, candidate, lambda: True)
