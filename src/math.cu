
#include <iostream>

#include "..\include\math.hpp"

#include ".\kernels.cu"

#define WARPS 16
#define STREAMS 8

// DEVICE SETUP

int cudamath::sm;
void cudamath::initDevice()
{
    cudaCheck( cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0) );
}

// VECTOR ADDITION

void cudamath::vectorAdd(int *a, int *b, int *c, int n)
{
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b, (void **)&d_c);
    // Clip streams for small inputs
    int realStreams = STREAMS<=n ? STREAMS : 1;
    // Create stream array
    cudaStream_t *streams = new cudaStream_t[realStreams];
    int streamLength = (n + (n%realStreams)) / realStreams;
    // For each stream
    for (int i=0; i < realStreams; ++i)
    {
        int offset = i*streamLength;
        int realsize = offset+streamLength>n ? n%streamLength : streamLength;
        cudaCheck( cudaStreamCreate(&streams[i]) );
        // Send data on stream
        cudaCheck( cudaMemcpyAsync(&d_a[offset], &a[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        cudaCheck( cudaMemcpyAsync(&d_b[offset], &b[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        // Run kernel on stream
        kernels::vectorAdd<<<sm, WARPS*32, 0, streams[i]>>>( &d_a[offset], &d_b[offset], &d_c[offset], realsize);
        // Retrieve data on stream
        cudaCheck( cudaMemcpyAsync(&c[offset], &d_c[offset], realsize*sizeof(int), cudaMemcpyDeviceToHost, streams[i]) );
    }
    cudaDeviceSynchronize();
    multiCudaFree(d_a, d_b, d_c);
}

void cudamath::vectorInAdd(int *a, int *b, int n)
{
    // Allocate device memory
    int *d_a, *d_b;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b);
    // Clip streams for small inputs
    int realStreams = STREAMS<=n ? STREAMS : 1;
    // Create stream array
    cudaStream_t *streams = new cudaStream_t[realStreams];
    int streamLength = (n + (n%realStreams)) / realStreams;
    // For each stream
    for (int i=0; i < realStreams; ++i)
    {
        int offset = i*streamLength;
        int realsize = offset+streamLength>n ? n%streamLength : streamLength;
        cudaCheck( cudaStreamCreate(&streams[i]) );
        // Send data on stream
        cudaCheck( cudaMemcpyAsync(&d_a[offset], &a[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        cudaCheck( cudaMemcpyAsync(&d_b[offset], &b[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        // Run kernel on stream
        kernels::vectorInAdd<<<sm, WARPS*32, 0, streams[i]>>>( &d_a[offset], &d_b[offset], realsize);
        // Retrieve data on stream
        cudaCheck( cudaMemcpyAsync(&a[offset], &d_a[offset], realsize*sizeof(int), cudaMemcpyDeviceToHost, streams[i]) );
    }
    cudaDeviceSynchronize();
    multiCudaFree(d_a, d_b);
}

// VECTOR SUBTRACTION

void cudamath::vectorSub(int *a, int *b, int *c, int n)
{
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b, (void **)&d_c);
    // Clip streams for small inputs
    int realStreams = STREAMS<=n ? STREAMS : 1;
    // Create stream array
    cudaStream_t *streams = new cudaStream_t[realStreams];
    int streamLength = (n + (n%realStreams)) / realStreams;
    // For each stream
    for (int i=0; i < realStreams; ++i)
    {
        int offset = i*streamLength;
        int realsize = offset+streamLength>n ? n%streamLength : streamLength;
        cudaCheck( cudaStreamCreate(&streams[i]) );
        // Send data on stream
        cudaCheck( cudaMemcpyAsync(&d_a[offset], &a[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        cudaCheck( cudaMemcpyAsync(&d_b[offset], &b[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        // Run kernel on stream
        kernels::vectorSub<<<sm, WARPS*32, 0, streams[i]>>>( &d_a[offset], &d_b[offset], &d_c[offset], realsize);
        // Retrieve data on stream
        cudaCheck( cudaMemcpyAsync(&c[offset], &d_c[offset], realsize*sizeof(int), cudaMemcpyDeviceToHost, streams[i]) );
    }
    cudaDeviceSynchronize();
    multiCudaFree(d_a, d_b, d_c);
}

void cudamath::vectorInSub(int *a, int *b, int n)
{
    // Allocate device memory
    int *d_a, *d_b;
    multiCudaMalloc(n*sizeof(int), (void **)&d_a, (void **)&d_b);
    // Clip streams for small inputs
    int realStreams = STREAMS<=n ? STREAMS : 1;
    // Create stream array
    cudaStream_t *streams = new cudaStream_t[realStreams];
    int streamLength = (n + (n%realStreams)) / realStreams;
    // For each stream
    for (int i=0; i < realStreams; ++i)
    {
        int offset = i*streamLength;
        int realsize = offset+streamLength>n ? n%streamLength : streamLength;
        cudaCheck( cudaStreamCreate(&streams[i]) );
        // Send data on stream
        cudaCheck( cudaMemcpyAsync(&d_a[offset], &a[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        cudaCheck( cudaMemcpyAsync(&d_b[offset], &b[offset], realsize*sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        // Run kernel on stream
        kernels::vectorInSub<<<sm, WARPS*32, 0, streams[i]>>>( &d_a[offset], &d_b[offset], realsize);
        // Retrieve data on stream
        cudaCheck( cudaMemcpyAsync(&a[offset], &d_a[offset], realsize*sizeof(int), cudaMemcpyDeviceToHost, streams[i]) );
    }
    cudaDeviceSynchronize();
    multiCudaFree(d_a, d_b);
}

void cudamath::transpose(int *in, int *out, int height, int width)
{
    // Allocate device memory
    int *d_in, *d_out;
    multiCudaMalloc(height*width*sizeof(int), (void **)&d_in, (void **)&d_out);
    cudaCheck( cudaMemcpy(d_in, in, height*width*sizeof(int), cudaMemcpyHostToDevice) );
    kernels::transpose<<<1, height*width>>>(d_in, d_out, height, width);
    cudaCheck( cudaMemcpy(out, d_out, height*width*sizeof(int), cudaMemcpyDeviceToHost) );
    multiCudaFree(d_in, d_out);
}

// MACROS

inline void cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        std::cout << "Cuda error: " << cudaGetErrorString(err) << std::endl;
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
