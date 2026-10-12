#include <stdio.h>

// Shared memory reduction with bank conflict avoidance
__global__ void reduce_shared(float *g_odata, const float *g_idata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Interleaved addressing - better shared memory access patterns
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
