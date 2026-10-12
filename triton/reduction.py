import torch
import triton
import triton.language as tl


@triton.jit
def reduction_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Reduce one block of values into a partial sum."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(values, axis=0)
    tl.store(out_ptr + pid, block_sum)


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Compute the sum of a 1D CUDA tensor using Triton."""
    if x.ndim != 1:
        raise ValueError("Expected a 1D tensor.")
    if not x.is_cuda:
        raise ValueError("Expected a CUDA tensor.")

    n = x.numel()
    if n == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    partial = torch.empty((num_blocks,), device=x.device, dtype=x.dtype)

    reduction_kernel[(num_blocks,)](x, partial, n, BLOCK_SIZE=BLOCK_SIZE)

    if partial.numel() == 1:
        return partial[0]

    return reduce_sum(partial)
