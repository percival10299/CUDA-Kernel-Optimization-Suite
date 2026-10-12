#include <stdio.h>

// Naive reduction kernel - O(n) operations for n threads
__global__ void reduce_naive(float *g_odata, const float *g_idata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    // Sequential addressing - suboptimal shared memory access
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
