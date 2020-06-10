
#pragma once

#include <cuda_runtime_api.h>

#include "..\..\include\math\mathEngine.hpp"

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
    void erode(float *map, int width, int droplets) override;

private:

    inline void GPUMathEngine::cudaCheck(cudaError_t err);
    inline void GPUMathEngine::multiCudaMalloc(int size, void **a, void **b, void **c);
    inline void GPUMathEngine::multiCudaFree(void *a, void *b, void *c);
};
