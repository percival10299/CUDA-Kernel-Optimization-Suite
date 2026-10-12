#include <stdio.h>

// Softmax kernel optimized for numerical stability
__global__ void softmax_kernel(float *output, const float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    if (idx < n) {
        // Step 1: Find max for numerical stability
        float maxval = input[idx];
        
        // Parallel reduction to find global max
        sdata[threadIdx.x] = maxval;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            }
            __syncthreads();
        }
        maxval = sdata[0];
        
        // Step 2: Compute exponentials
        float exp_val = expf(input[idx] - maxval);
        
        // Step 3: Sum of exponentials
        sdata[threadIdx.x] = exp_val;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        float sum = sdata[0];
        
        // Step 4: Normalize
        output[idx] = exp_val / sum;
    }
}
