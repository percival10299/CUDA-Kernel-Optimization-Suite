"""
Vector addition implementation using Triton
"""
import triton
import triton.language as tl
import torch


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton vector addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle edge cases
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python wrapper for vector addition"""
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output
