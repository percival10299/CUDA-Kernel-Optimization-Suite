#include <stdio.h>

// Tiled matrix multiplication with shared memory
#define TILE_SIZE 16

__global__ void matmul_tiled(float *C, const float *A, const float *B, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < n && t * TILE_SIZE + tx < n) {
            As[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        }
        if (t * TILE_SIZE + ty < n && col < n) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        }
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}
