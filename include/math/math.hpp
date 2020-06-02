
#pragma once

#include<cuda_runtime_api.h>

class cudamath
{
private:

    /** Number of streaming multiprocessors available */
    static int sm;

public:

    /** Initialise device data */
    static void initDevice();

    /** Create heightmap on gpu */
    static void generatePerlinHeightMap(int dimension, float min, float max, float *out, float period);

private:

    /** Error checking function for cuda calls */
    static inline void cudaCheck(cudaError_t err);

    /** Macro for allocating multiple, equally-sized chunks on device memory */
    static inline void multiCudaMalloc(int size, void **a, void **b=NULL, void **c=NULL);

    /** Macro for freeing device memory */
    static inline void multiCudaFree(void *a, void *b=NULL, void *c=NULL);
};
