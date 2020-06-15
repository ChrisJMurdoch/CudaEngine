
// Engine
#include "..\..\include\logger\log.hpp"
#include "..\..\include\math\gpuMathEngine.hpp"

// GLM & CUDA
#include "cuda.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Device code
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

void GPUMathEngine::brush(float *map, int width, int x, int y, float amount, int radius)
{
    int dim = (radius*2)+1;
    amount /= (float)dim*dim;

    // Sum up eligible cell distribution
    float total = 0;
    for (int xo=-radius; xo<radius+1; xo++) for (int yo=-radius; yo<radius+1; yo++)
    {
        // Out of bounds
        if  ( x+xo<0 || x+xo>=width || y+yo<0 || y+yo>=width )
            continue;
        
        // Fade
        float dist = pow( pow( (xo), 2) + pow( (yo), 2), 0.5);
        float close = (radius - dist) / radius * 2;
        close = close<0 ? 0 : close;

        total += close;
    }

    // Calculate total distribution
    float mult = dim*dim / total;

    // Add values
    for (int xo=-radius; xo<radius+1; xo++) for (int yo=-radius; yo<radius+1; yo++)
    {
        // Out of bounds
        if  ( x+xo<0 || x+xo>=width || y+yo<0 || y+yo>=width )
            continue;

        // Fade
        float dist = pow( pow( (xo), 2) + pow( (yo), 2), 0.5);
        float close = (radius - dist) / radius * 2;
        close = close<0 ? 0 : close;

        // Alter cell
        map[ x+xo + ((y+yo)*width) ] += mult*amount*close;
        total++;
    }
}

float GPUMathEngine::getCellHeight(float *map, int width, int x, int y)
{
    // Simulate walls on edge of map to avoid out of bounds erosion
    return ( x<0 || x>=width || y<0 || y>=width ) ? 9999 : map[ x + (y*width) ]+1000;
}

void GPUMathEngine::erodeCell(float *map, int width, int x, int y, float speed, float sediment, int radius)
{
    // Get cell height
    float height = getCellHeight(map, width, x, y);

    // Find lowest cell in 3x3 (including self)
    int lx=x, ly=y;
    float lh = height;
    for (int xo=-1; xo<2; xo++) for (int yo=-1; yo<2; yo++)
    {
        float h = getCellHeight(map, width, x+xo, y+yo);
        if ( h<lh )
        {
            lx = x+xo;
            ly = y+yo;
            lh = h;
        }
    }

    // Calculate difference
    float delta = height - lh;

    // Base case => Not enough speed to move
    if ( -delta >= speed )
    {
        brush(map, width, x, y, sediment, radius);
        return;
    }

    // Calculate new speed with friction
    const float FRICTION = 0.1f;
    speed += delta - FRICTION;
    
    // Calculate new sediment capacity
    float capacity = speed;
    float deposit = sediment - capacity;

    // Add / Remove sediment from surrounding cells
    brush(map, width, x, y, deposit, radius);

    // Recurse on downhill cell
    erodeCell(map, width, lx, ly, speed, capacity, radius);
}

void GPUMathEngine::erode(float *map, int width, int droplets, int radius)
{
    static float s_i = 0;
    srand( s_i++ );
    for (int i=0; i<droplets; i++)
    {
        int x = rand() % width, y = rand() % width;
        erodeCell(map, width, x, y, 0, 0, radius);
    }
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
