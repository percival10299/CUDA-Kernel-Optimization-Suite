#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Host wrapper function
void vectorAddHost(const float *h_A, const float *h_B, float *h_C, int numElements) {
    size_t size = numElements * sizeof(float);
    
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, numElements
    );

    // Record stop
    cudaEventRecord(stop);

    // Wait for GPU
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    // Clean up event resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
