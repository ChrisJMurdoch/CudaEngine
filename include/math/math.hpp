
#pragma once

#include<cuda_runtime_api.h>

class cudamath
{
    /** Number of streaming multiprocessors available */
    static int sm;

public:

    /** Initialise device data */
    static void initDevice();

    /** Adds two arrays together */
    static void vectorAdd(int *a, int *b, int *c, int n);

    /** Adds two arrays together in-place */
    static void vectorInAdd(int *a, int *b, int n);

    /** Subtracts one vector from another */
    static void vectorSub(int *a, int *b, int *c, int n);

    /** Subtracts one vector from another in-place */
    static void vectorInSub(int *a, int *b, int n);

    /** Transposes a matrix, monolithic */
    static void transpose(int *in, int *out, int height, int width);
};

/** Error checking function for cuda calls */
inline void cudaCheck(cudaError_t err);

/** Macro for allocating multiple, equally-sized chunks on device memory */
inline void multiCudaMalloc(int size, void **a, void **b=NULL, void **c=NULL);

/** Macro for freeing device memory */
inline void multiCudaFree(void *a, void *b=NULL, void *c=NULL);
