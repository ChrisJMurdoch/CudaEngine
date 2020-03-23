
#include<iostream>

#include "..\..\include\graphic\main.hpp"
#include "..\..\include\math\math.hpp"

#define PINNED_MEMORY true

int main(int argc, char *argv[])
{

    // Initialise math engine
    cudamath::initDevice();

    // Test variables
    int N = 10;

    // Allocate
#if PINNED_MEMORY
    int *a; cudaCheck( cudaMallocHost((void**)&a, N*sizeof(int)) );
    int *b; cudaCheck( cudaMallocHost((void**)&b, N*sizeof(int)) );
    int *c; cudaCheck( cudaMallocHost((void**)&c, N*sizeof(int)) );
#else
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];
#endif
    
    // Run
    cudamath::vectorAdd(a, b, c, N);

    // Cleanup
#if PINNED_MEMORY
    cudaCheck( cudaFreeHost(a) );
    cudaCheck( cudaFreeHost(b) );
    cudaCheck( cudaFreeHost(c) );
#else
    delete[] a;
    delete[] b;
    delete[] c;
#endif

    std::cout << "Exiting.";
    return 0;
}
