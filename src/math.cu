
#include <iostream>
#include "..\include\math.hpp"

namespace cudamath {

    __global__
    void strideAdd(int *a, int *b, int *c, int size)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<size; i += blockDim.x*gridDim.x)
            c[i] = a[i] + b[i];
    }

    void gpuAdd(int *a, int *b, int *c, int n, int warps)
    {
        // Allocate device memory
        int *d_a, *d_b, *d_c;
        cudaCheck( cudaMalloc(&d_a, n*sizeof(int)) );
        cudaCheck( cudaMalloc(&d_b, n*sizeof(int)) );
        cudaCheck( cudaMalloc(&d_c, n*sizeof(int)) );

        // Send memory to device
        cudaCheck( cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice) );
        cudaCheck( cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice) );

        // Run kernel
        int sms;
        cudaCheck( cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) );
        strideAdd<<<sms, warps*32>>>(d_a, d_b, d_c, n);

        // Fetch memory from device
        cudaCheck( cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost) );

        // Cleanup
        cudaCheck( cudaFree(d_a) );
        cudaCheck( cudaFree(d_b) );
        cudaCheck( cudaFree(d_c) );
    }

    void cpuAdd(int *a, int *b, int *c, int n)
    {
        for (int i=0; i<n; i++)
            c[i] = a[i] + b[i];
    }

    void cudaCheck(cudaError_t err)
    {
        if (err != cudaSuccess)
            std::cout << "Cuda error: " << cudaGetErrorString(err);
    }

}
