
// Directly included in math.cu to avoid loss of device code optimisation

#include <functional>
#include <cmath>

#include "cuda.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define M_PI 3.14159265358979323846

namespace kernels
{
    __global__
    void vectorAdd(int *a, int *b, int *c, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            c[i] = a[i] + b[i];
    }

    __global__
    void vectorInAdd(int *a, int *b, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            a[i] = a[i] + b[i];
    }

    __global__
    void vectorSub(int *a, int *b, int *c, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            c[i] = a[i] - b[i];
    }

    __global__
    void vectorInSub(int *a, int *b, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            a[i] = a[i] - b[i];
    }

    __global__
    void transpose(int *in, int *out, int height, int width)
    {
        out[threadIdx.x] = in[ (threadIdx.x*width) % (height*width) + ((threadIdx.x)/height) ];
    }

    // NUMBER GENERATION

    __device__
    int centHash(int x)
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return ( x % 201 ) - 100;
    }

    __device__
    int combine(int x, int y) {
        return (x*12345) + y;
    }

    __device__
    float lerp(float a, float b, float x)
    {
        return a + x * (b - a);
    }

    __device__
    float fade(float x)
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
            glm::vec2 BL = glm::normalize( glm::vec2( centHash( combine( X , Y ) ), centHash( combine( X , Y )+1 ) ) );
            glm::vec2 BR = glm::normalize( glm::vec2( centHash( combine(X+1, Y ) ), centHash( combine(X+1, Y )+1 ) ) );
            glm::vec2 TL = glm::normalize( glm::vec2( centHash( combine( X ,Y+1) ), centHash( combine( X ,Y+1)+1 ) ) );
            glm::vec2 TR = glm::normalize( glm::vec2( centHash( combine(X+1,Y+1) ), centHash( combine(X+1,Y+1)+1 ) ) );

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
            float bottom = lerp( BLd, BRd, fade(point.x) );
            float top = lerp( TLd, TRd, fade(point.x) );
            float centre = lerp( bottom, top, fade(point.y) );

            // Set value
            out[index] = ( ((centre+1) / 2) * (max-min) ) + min;

            // Stride
            index += blockDim.x*gridDim.x;
        }
        while ( y<dimension );
    }
}