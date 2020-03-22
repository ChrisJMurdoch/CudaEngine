
#include <iostream>

#include "..\include\math.hpp"

#include ".\kernels.cu"

#define WARPS 8

// VECTOR ADDITION

void cudamath::vectorAdd(int *a, int *b, int *c, int n)
{
    // Host memory -> Device memory
    int *d_a, *d_b, *d_c;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b, (void **)&d_c);
    cudaCheck( cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice) );
    // Run kernel
    int sm; cudaCheck( cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0) );
    kernels::vectorAdd<<<sm, WARPS*32>>>(d_a, d_b, d_c, n);
    // Device memory -> Host memory
    cudaCheck( cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost) );
    multiCudaFree(d_a, d_b, d_c);
}

void cudamath::vectorInAdd(int *a, int *b, int n)
{
    // Host memory -> Device memory
    int *d_a, *d_b;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b);
    cudaCheck( cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice) );
    // Run kernel
    int sm; cudaCheck( cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0) );
    kernels::vectorInAdd<<<sm, WARPS*32>>>(d_a, d_b, n);
    // Device memory -> Host memory
    cudaCheck( cudaMemcpy(a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost) );
    multiCudaFree(d_a, d_b);
}

// VECTOR SUBTRACTION

void cudamath::vectorSub(int *a, int *b, int *c, int n)
{
    // Host memory -> Device memory
    int *d_a, *d_b, *d_c;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b, (void **)&d_c);
    cudaCheck( cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice) );
    // Run kernel
    int sm; cudaCheck( cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0) );
    kernels::vectorSub<<<sm, WARPS*32>>>(d_a, d_b, d_c, n);
    // Device memory -> Host memory
    cudaCheck( cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost) );
    multiCudaFree(d_a, d_b, d_c);
}

void cudamath::vectorInSub(int *a, int *b, int n)
{
    // Host memory -> Device memory
    int *d_a, *d_b;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b);
    cudaCheck( cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice) );
    // Run kernel
    int sm; cudaCheck( cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0) );
    kernels::vectorInSub<<<sm, WARPS*32>>>(d_a, d_b, n);
    // Device memory -> Host memory
    cudaCheck( cudaMemcpy(a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost) );
    multiCudaFree(d_a, d_b);
}

// MACROS

inline void cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        std::cout << "Cuda error: " << cudaGetErrorString(err);
}

inline void multiCudaMalloc(int size, void **a, void **b, void **c)
{
    cudaCheck( cudaMalloc(a, size) );
    if (b != NULL) cudaCheck( cudaMalloc(b, size) );
    if (c != NULL) cudaCheck( cudaMalloc(c, size) );
}

inline void multiCudaFree(void *a, void *b, void *c)
{
    cudaCheck( cudaFree(a) );
    if (b != NULL) cudaCheck( cudaFree(b) );
    if (c != NULL) cudaCheck( cudaFree(c) );
}
