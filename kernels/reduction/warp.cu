#include <stdio.h>

// Warp-optimized reduction - uses warp shuffles for efficiency
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_warp(float *g_odata, const float *g_idata, unsigned int n) {
    float sum = 0;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Reduce multiple elements per thread
    for (; i < n; i += gridDim.x * blockDim.x) {
        sum += g_idata[i];
    }
    
    sum = warpReduceSum(sum);
    
    // Block reduction in shared memory
    extern __shared__ float sdata[];
    __shared__ float result;
    
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;
    
    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();
    
    if (wid == 0) {
        sum = warpReduceSum((lane < blockDim.x / warpSize) ? sdata[lane] : 0);
        if (lane == 0) {
            g_odata[blockIdx.x] = sum;
        }
    }
}
