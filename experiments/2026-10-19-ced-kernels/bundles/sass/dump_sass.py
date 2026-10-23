"""Disassemble the Triton reference and TileLang candidate CED copy and request kernels."""
import glob
import os

import torch

from benchmarks.kernels.ced_reference import _copy_rows, _decoder_requests
from vllm.models.deepseek_v4_1.tilelang import ced
from vllm.triton_utils import triton as _triton

triton_cdiv = _triton.cdiv

CACHE = os.environ["TILELANG_CACHE_DIR"]


def objects():
    return set(glob.glob(os.path.join(CACHE, "**", "*.cubin"), recursive=True))


def sass(path):
    # Triton's bundled cuobjdump knows sm_121a; the image's toolkit one prints no functions.
    from triton.tools.disasm import get_sass
    return get_sass(open(path, "rb").read())


def tilelang(label, build):
    before = objects()
    build()
    torch.cuda.synchronize()
    for path in sorted(objects() - before):
        print(f"===== tilelang {label} {path}")
        print(sass(path))


def triton(label, compiled):
    print(f"===== triton {label}")
    print(compiled.asm["sass"])


dev = torch.device("cuda")
for dtype, tail, rows, scatter in [
    (torch.int32, (), 4048, False),
    (torch.bool, (), 4048, False),
    (torch.bfloat16, (5120,), 1, False),
]:
    source = torch.ones((rows, *tail), dtype=dtype, device=dev)
    idx = torch.arange(rows, dtype=torch.int64, device=dev)
    out = torch.zeros((rows, *tail), dtype=dtype, device=dev)
    width = 1
    for d in tail:
        width *= d
    label = f"copy {dtype} {tail} scatter={scatter}"
    tilelang(label, lambda: ced.copy_rows(source, idx, out, scatter))
    triton(label, _copy_rows[(triton_cdiv(rows * width, 512),)](
        source, idx, out, rows, rows, tail, source.stride(), width, scatter, 512))
