
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>

#include "..\include\main.hpp"
#include "..\include\tests.hpp"
#include "..\include\math.hpp"

#define DEBUG_OUTPUT false
#define PINNED_MEMORY true
#define WARPS 8

int main(int argc, char *argv[])
{
    // Get parameters
    int TESTS = 10;
    getArg(argc, argv, 1, &TESTS);

    int power = 6;
    getArg(argc, argv, 2, &power);
    int N = pow(10, power);

    // Display config
    std::cout << "Config: " << TESTS << " tests, 10^" << power << " vector size." << std::endl;

#if DEBUG_OUTPUT
    testOutputs(WARPS);
#endif
    
    // Allocate host memory
#if PINNED_MEMORY
    int *a; cudaCheck( cudaMallocHost((void**)&a, N*sizeof(int)) );
    int *b; cudaCheck( cudaMallocHost((void**)&b, N*sizeof(int)) );
    int *c; cudaCheck( cudaMallocHost((void**)&c, N*sizeof(int)) );
#else
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];
#endif
    
    // Warmup
    cudamath::vectorSub(a, b, c, N, WARPS);

    // Profile engine
    int va = 0; testVectorAdd(a, b, c, TESTS, N, WARPS, &va);
    int via = 0; testVectorInAdd(a, b, TESTS, N, WARPS, &via);
    int vs = 0; testVectorSub(a, b, c, TESTS, N, WARPS, &vs);
    int vis = 0; testVectorInSub(a, b, TESTS, N, WARPS, &vis);

    // Cleanup
#if PINNED_MEMORY
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
#else
    delete[] a;
    delete[] b;
    delete[] c;
#endif

    return 0;
}

void getArg(int argc, char *argv[], int index, int *dest)
{
    if (argc > index)
    {
        try
        {
            *dest = std::stoi(argv[index]);
        }
        catch (std::invalid_argument const &e)
        {
            std::cerr << "Parameter parsing error." << std::endl;
        }
    }
}
