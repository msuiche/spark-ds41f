"""Element gather variants at the gated copy shapes, by the calibrated paired method."""
import functools

import torch

import bench
from vllm.models.deepseek_v4_1.tilelang import ced

VARIANTS = ["current", "split", "safe-split", "conditional"]
LAYOUTS = [torch.bool, torch.int32, torch.int64]

for variant in VARIANTS:
    for kernel, shape, build in bench.copy_cases():
        if not shape["serving"] or shape["scatter"] or shape["tail"] != ():
            continue

        def built(build=build, variant=variant):
            reference, candidate, outputs, equal = build()
            source, indices = candidate.__closure__ if False else (None, None)
            return reference, candidate, outputs, equal

        # Rebind the candidate to the variant: copy_case's candidate calls ced.copy_rows.
        original = ced.copy_rows
        ced.copy_rows = functools.partial(original, variant=variant)
        try:
            bench.paired(kernel, {**shape, "variant": variant}, build)
        finally:
            ced.copy_rows = original
