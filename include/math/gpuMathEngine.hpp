
#pragma once

#include <math/mathEngine.hpp>

#include <cuda_runtime_api.h>

class GPUMathEngine : public MathEngine
{
private:

    /** Number of streaming multiprocessors on device */
    int nSM;

public:

    GPUMathEngine();

    /** Create heightmap on gpu */
    void generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves=1) override;

    /** Erode terrain heightmap */
    void erode(float *map, int width, int droplets, int radius) override;

private:

    void brush(float *map, int width, int x, int y, float amount, int radius);
    float getCellHeight(float *map, int width, int x, int y);
    void erodeCell(float *map, int width, int x, int y, float speed, float sediment, int radius);

    inline void GPUMathEngine::cudaCheck(cudaError_t err);
    inline void GPUMathEngine::multiCudaMalloc(int size, void **a, void **b, void **c);
    inline void GPUMathEngine::multiCudaFree(void *a, void *b, void *c);
};
