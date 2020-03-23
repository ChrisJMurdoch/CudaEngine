
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>

#include "..\..\include\math\test.hpp"
#include "..\..\include\math\math.hpp"

#define DEBUG_OUTPUT false
#define PINNED_MEMORY true

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

    // Initialise device
    cudamath::initDevice();

#if DEBUG_OUTPUT
    debugOutputs();
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
    cudamath::vectorSub(a, b, c, N);

    // Profile engine
    int va = 0; profileVectorAdd(a, b, c, TESTS, N, &va);
    int via = 0; profileVectorInAdd(a, b, TESTS, N, &via);
    int vs = 0; profileVectorSub(a, b, c, TESTS, N, &vs);
    int vis = 0; profileVectorInSub(a, b, TESTS, N, &vis);

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

    return 0;
}

void profileVectorAdd(int *a, int *b, int *c, int TESTS, int N, int *ms)
{
    populate(a, b, N);
    for (int i=0; i<TESTS; i++)
    {
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorAdd(a, b, c, N);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count());
    }
    std::cout << "vectorAdd:   " << (*ms / 1000000) / TESTS << "ms avg." << std::endl;
}

void profileVectorInAdd(int *a, int *b, int TESTS, int N, int *ms)
{
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorInAdd(a, b, N);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorInAdd: " << *ms / TESTS << "ms avg." << std::endl;
}

void profileVectorSub(int *a, int *b, int *c, int TESTS, int N, int *ms)
{
    populate(a, b, N);
    for (int i=0; i<TESTS; i++)
    {
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorSub(a, b, c, N);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorSub:   " << *ms / TESTS << "ms avg." << std::endl;
}

void profileVectorInSub(int *a, int *b, int TESTS, int N, int *ms)
{
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorInSub(a, b, N);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorInSub: " << *ms / TESTS << "ms avg." << std::endl;
}

void debugOutputs()
{
    // VECTORS

    int size = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {1, 1, 1, 1, 1};
    int c[] = {0, 0, 0, 0, 0};

    std::cout << "a:           " << a[0] << a[1] << a[2] << a[3] << a[4] << std::endl;
    std::cout << "b:           " << b[0] << b[1] << b[2] << b[3] << b[4] << std::endl;
    
    cudamath::vectorAdd(a, b, c, size);
    std::cout << "vectorAdd:   " << c[0] << c[1] << c[2] << c[3] << c[4] << std::endl;

    cudamath::vectorInAdd(a, b, size);
    std::cout << "vectorInAdd: " << a[0] << a[1] << a[2] << a[3] << a[4] << std::endl;
    a[0]=1, a[1]=2, a[2]=3, a[3]=4, a[4]=5; // Reset a

    cudamath::vectorSub(a, b, c, size);
    std::cout << "vectorSub:   " << c[0] << c[1] << c[2] << c[3] << c[4] << std::endl;

    cudamath::vectorInSub(a, b, size);
    std::cout << "vectorInSub: " << a[0] << a[1] << a[2] << a[3] << a[4] << std::endl;
    a[0]=1, a[1]=2, a[2]=3, a[3]=4, a[4]=5; // Reset a

    // MATRICES

    int height = 3;
    int width = 4;
    int ma[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int mb[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0 };

    cudamath::transpose(ma, mb, height, width);
    std::cout << "transpose:   "
        << mb[0] << mb[1] << mb[2] << mb[3]
        << mb[4] << mb[5] << mb[6] << mb[7]
        << mb[8] << mb[9] << mb[10] << mb[11]
        << std::endl;

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

void populate(int *array_a, int *array_b, int n)
{
    for (int i=0; i<n; i++)
    {
        array_a[i] = i;
        array_b[i] = i*2;
    }
}

void validate(int *array_a, int *array_b, int n)
{
    for (int i=0; i<n; i++)
    {
        if (array_a[i] != array_b[i])
        {
            std::cout << "Arrays diverge at index: " << i << "." << std::endl;
            return;
        }
    }
    std::cout << "Arrays match." << std::endl;
}
