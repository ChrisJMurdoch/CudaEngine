
#include <functional>
#include <iostream>
#include <cmath>

// GLM - Cuda interoperability
#include "cuda.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "..\..\include\math\gpuMathEngine.hpp"

#define STREAMS 12
#define WARPS2D 4
#define WARPS WARPS2D*WARPS2D
#define M_PI 3.14159265358979323846

// DEVICE SETUP

GPUMathEngine::GPUMathEngine()
{
    cudaCheck( cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0) );
}

// KERNELS

__device__
int k_combine(int x, int y) {
    return (x*12345) + y;
}

__device__
float k_hashFunction(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return ( x % 101 ) / 100.0f;
}

__device__
int k_centHash(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return ( x % 201 ) - 100;
}

__device__
float k_lerp(float a, float b, float x)
{
    return a + x * (b - a);
}

__device__
float k_fade(float x)
{
    return x * x * x * (x * (x * 6 - 15) + 10);
}

__global__
void perlinSample(float *out, int dimension, float min, float max, float period)
{
    
    // Start index
    int startIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride
    int index = startIndex;
    int x, y;
    do
    {
        // Thread calculation
        x = index % dimension;
        y = index / dimension;

        // Square index
        int X = std::floor( x / period );
        int Y = std::floor( y / period );

        // Normal relative position
        float rx = (x/period) - X;
        float ry = (y/period) - Y;

        // Square corner vectors
        glm::vec2 BL = glm::normalize( glm::vec2( k_centHash( k_combine( X , Y ) ), k_centHash( k_combine( X , Y )+1 ) ) );
        glm::vec2 BR = glm::normalize( glm::vec2( k_centHash( k_combine(X+1, Y ) ), k_centHash( k_combine(X+1, Y )+1 ) ) );
        glm::vec2 TL = glm::normalize( glm::vec2( k_centHash( k_combine( X ,Y+1) ), k_centHash( k_combine( X ,Y+1)+1 ) ) );
        glm::vec2 TR = glm::normalize( glm::vec2( k_centHash( k_combine(X+1,Y+1) ), k_centHash( k_combine(X+1,Y+1)+1 ) ) );

        // Relational vectors
        glm::vec2 point = glm::vec2( rx, ry );
        glm::vec2 BLr = glm::vec2( 0, 0 ) - point;
        glm::vec2 BRr = glm::vec2( 1, 0 ) - point;
        glm::vec2 TLr = glm::vec2( 0, 1 ) - point;
        glm::vec2 TRr = glm::vec2( 1, 1 ) - point;

        // Dot products
        float BLd = glm::dot( BL, BLr );
        float BRd = glm::dot( BR, BRr );
        float TLd = glm::dot( TL, TLr );
        float TRd = glm::dot( TR, TRr );

        // Interpolate
        float bottom = k_lerp( BLd, BRd, k_fade(point.x) );
        float top = k_lerp( TLd, TRd, k_fade(point.x) );
        float centre = k_lerp( bottom, top, k_fade(point.y) );

        // Set value
        out[index] = ( ((centre+1) / 2) * (max-min) ) + min;

        // Stride
        index += blockDim.x*gridDim.x;
    }
    while ( y<dimension );
}

__global__
void hashSample(float *out, int dimension, float min, float max, float period)
{
    
    // Start index
    int startIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride
    int index = startIndex;
    int y;
    do
    {
        y = index / dimension;

        // Set value
        out[index] = min + ( k_hashFunction(index) * (max-min) );

        // Stride
        index += blockDim.x*gridDim.x;
    }
    while ( y<dimension );
}

// FUNCTIONS

void GPUMathEngine::generateHeightMap(int dimension, float min, float max, float *out, Sample sample, float period, int octaves)
{
    // Allocate device memory
    float *d_out;
    cudaCheck( cudaMalloc( (void **)&d_out, dimension*dimension*sizeof(float) ) );
    switch ( sample )
    {
    case hash:
        hashSample<<<nSM, WARPS*32>>>(d_out, dimension, min, max, period);
        break;
    case perlin:
        perlinSample<<<nSM, WARPS*32>>>(d_out, dimension, min, max, period);
        break;
    default:
        hashSample<<<nSM, WARPS*32>>>(d_out, dimension, min, max, period);
        break;
    }
    cudaCheck( cudaMemcpy(out, d_out, dimension*dimension*sizeof(float), cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(d_out) );
}

// MACROS

inline void GPUMathEngine::cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        std::cout << "Cuda error: " << cudaGetErrorString(err) << std::endl;
}

inline void GPUMathEngine::multiCudaMalloc(int size, void **a, void **b, void **c)
{
    cudaCheck( cudaMalloc(a, size) );
    if (b != NULL) cudaCheck( cudaMalloc(b, size) );
    if (c != NULL) cudaCheck( cudaMalloc(c, size) );
}

inline void GPUMathEngine::multiCudaFree(void *a, void *b, void *c)
{
    cudaCheck( cudaFree(a) );
    if (b != NULL) cudaCheck( cudaFree(b) );
    if (c != NULL) cudaCheck( cudaFree(c) );
}
