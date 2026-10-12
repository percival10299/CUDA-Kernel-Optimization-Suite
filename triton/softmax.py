"""
Softmax implementation using Triton with numerical stability
"""
import triton
import triton.language as tl
import torch


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, 
                  n_cols, BLOCK_SIZE: tl.constexpr):
    """Triton softmax kernel"""
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Compute max for numerical stability
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    max_row = tl.max(row, axis=0)
    
    # Compute exp(x - max)
    numerator = tl.exp(row - max_row)
    
    # Compute denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax using Triton"""
    n_rows, n_cols = x.shape
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Define grid
    def grid(meta):
        return (n_rows,)
    
    softmax_kernel[grid](
        output, x, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=1024
    )
    
    return output
