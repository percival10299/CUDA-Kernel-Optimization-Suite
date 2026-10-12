#include <stdio.h>

// Optimized matrix multiplication with prefetching and better memory access
#define TILE_SIZE 32
#define BLOCK_SIZE 32

__global__ void matmul_optimized(float *C, const float *A, const float *B, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // More iterations with smaller tile size per iteration
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles with coalesced memory access
        if (row < n && t * TILE_SIZE + tx < n) {
            As[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        }
        if (t * TILE_SIZE + ty < n && col < n) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        }
        __syncthreads();
        
        // Compute partial products
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}
