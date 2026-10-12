import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Triton matrix multiplication kernel."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        a_mask = (rm[:, None] < M) & (k_start + rk[None, :] < K)
        b_mask = (k_start + rk[:, None] < K) & (rn[None, :] < N)

        a = tl.load(
            A_ptr + rm[:, None] * stride_am + (k_start + rk)[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        b = tl.load(
            B_ptr + (k_start + rk)[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
        )

        acc += tl.dot(a, b)

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two 2D CUDA tensors using Triton."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected 2D tensors.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Expected CUDA tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {a.shape} and {b.shape}.")

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Inner dimensions mismatch: {K} != {K2}.")

    out = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out
