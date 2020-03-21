
#pragma once

#include<cuda_runtime_api.h>

/** Function for adding two arrays together on the GPU */
void gpuAdd(int *a, int *b, int *c, int n, int warps=4);
/** Function for adding two arrays together on the CPU */
void cpuAdd(int *a, int *b, int *c, int n);
/** Inline error checking function for cuda calls. */
void cudaCheck(cudaError_t err);
