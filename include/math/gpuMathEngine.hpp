
#pragma once

#include <cuda_runtime_api.h>

#include "..\..\include\math\mathEngine.hpp"

class GPUMathEngine : public MathEngine
{
private:

    /** Number of streaming multiprocessors on device - default to GTX1060 */
    int nSM = 10;

public:

    GPUMathEngine();

    /** Create heightmap on gpu */
    void generateHeightMap(int dimension, float min, float max, float *out, Sample sample = sin, float period = 10, int octaves = 1) override;

private:

    inline void GPUMathEngine::cudaCheck(cudaError_t err);
    inline void GPUMathEngine::multiCudaMalloc(int size, void **a, void **b, void **c);
    inline void GPUMathEngine::multiCudaFree(void *a, void *b, void *c);
};
