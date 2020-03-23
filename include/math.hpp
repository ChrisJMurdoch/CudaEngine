
#pragma once

#include<cuda_runtime_api.h>

namespace cudamath
{
    /** Initialise device data */
    void initDevice();
    extern int sm;

    /** Function for adding two arrays together on the GPU */
    void vectorAdd(int *a, int *b, int *c, int n);

    /** Function for adding two arrays together in-place on the GPU */
    void vectorInAdd(int *a, int *b, int n);

    /** Function for subtracting one vector from another on the GPU */
    void vectorSub(int *a, int *b, int *c, int n);

    /** Function for subtracting one vector from another in-place on the GPU */
    void vectorInSub(int *a, int *b, int n);
}

/** Inline error checking function for cuda calls. */
inline void cudaCheck(cudaError_t err);

/** Inline shortcut for allocating multiple, equally-sized chunks on device memory */
inline void multiCudaMalloc(int size, void **a, void **b=NULL, void **c=NULL);

/** Inline shortcut for freeing device memory */
inline void multiCudaFree(void *a, void *b=NULL, void *c=NULL);
