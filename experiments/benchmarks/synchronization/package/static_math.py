# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modified by mjlab developers and for the bounded Stage12 inference adapter.
# SPDX-License-Identifier: BSD-3-Clause
"""Same quaternion candidates and arithmetic order, with fixed-size selection.

The original positive sqrt uses boolean compaction and the final one-hot index
also has a data-dependent output length. Replace only these selection operations;
retain all four candidates, argmax tie-breaking, floor and quaternion convention.
Inference only: this module does not define a training/autograd replacement.
"""
import torch


def sqrt_positive(x):
    return torch.sqrt(torch.where(x > 0, x, torch.zeros_like(x)))


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = sqrt_positive(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    # Preserve the reviewed source's floor construction and dtype conversion.
    floor = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    index = q_abs.argmax(dim=-1)[..., None, None].expand(batch_dim + (1, 4))
    return torch.gather(candidates, -2, index).squeeze(-2)
