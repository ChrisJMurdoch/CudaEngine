
// GLM & CUDA
#include "cuda.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Engine
#include "..\..\include\logger\log.hpp"
#include "..\..\include\math\gpuMathEngine.hpp"

// Host/Device functions
namespace gpucommon
{
    #include "..\..\src\math\common.cu"
}

// HARDWARE SETTINGS

#define STREAMS 12
#define WARPS 16

// DEVICE SETUP

GPUMathEngine::GPUMathEngine()
{
    cudaCheck( cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0) );
}

// HEIGHTMAP GENERATION

__global__
void heightmapKernel(float *out, int dimension, float min, float max, GPUMathEngine::Sample sample, float period, int octaves )
{
    // Start index
    int startIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Grid stride
    int index = startIndex;

    // Thread calculation
    int x = index % dimension;
    int y = index / dimension;
    do
    {
        // Get sample
        float value;
        switch ( sample )
        {
        case GPUMathEngine::mountain:
            value = gpucommon::mountain(x, y, period);
            break;
        default:
            value = gpucommon::fractal(x, y, period, sample, octaves);
            break;
        }
        out[index] = min + ( value * (max-min) );

        // Stride forward
        index += blockDim.x*gridDim.x;
        x = index % dimension;
        y = index / dimension;
    }
    while ( y<dimension );
}

void GPUMathEngine::generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves)
{
    // Allocate device memory
    float *d_out;
    int size = dimension*dimension*sizeof(float);
    cudaCheck( cudaMalloc( (void **)&d_out, size ) );
    heightmapKernel<<<nSM, WARPS*32>>>( d_out, dimension, min, max, sample, period, octaves );
    cudaCheck( cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(d_out) );
}

// MACROS

inline void GPUMathEngine::cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        Log::print( Log::error, "CudaCheck:" );
        Log::print( Log::error, cudaGetErrorString(err) );
    }
}
