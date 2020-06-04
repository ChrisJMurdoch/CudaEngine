
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
#define WARPS2D 4
#define WARPS WARPS2D*WARPS2D

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
        // Octaves
        float height = 0;
        for (int o=0; o<octaves; o++)
        {
            const float lacunarity = 0.5, persistance = 0.4; // Lacunarity inverse for period

            float pmult = pow(lacunarity, o);
            float amplitude = pow(persistance, o);

            float value;
            switch ( sample )
            {
            case GPUMathEngine::hash:
                value = gpucommon::hashSample( x, y, pmult*period );
                break;
            case GPUMathEngine::sin:
                value = gpucommon::sinSample( x, y, pmult*period );
                break;
            case GPUMathEngine::perlin:
                value = gpucommon::perlinSample( x, y, pmult*period );
                break;
            default:
                value = gpucommon::hashSample( x, y, pmult*period );
                break;
            }
            height += ( min + ( (max-min) * value ) ) * amplitude;

            if ( o != 0 ) // Remove half of height if not first octave
                height -= ( min + ( (max-min) * 0.5 ) ) * amplitude;
        }
        //out[index] = height;
        out[index] = height;

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
