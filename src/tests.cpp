
#include <iostream>
#include <chrono>

#include "..\include\tests.hpp"
#include "..\include\math.hpp"

void testVectorAdd(int *a, int *b, int *c, int TESTS, int N, int WARPS, int *ms)
{
    populate(a, b, N);
    for (int i=0; i<TESTS; i++)
    {
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorAdd(a, b, c, N, WARPS);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorAdd:   " << *ms / TESTS << "ms avg." << std::endl;
}

void testVectorInAdd(int *a, int *b, int TESTS, int N, int WARPS, int *ms)
{
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorInAdd(a, b, N, WARPS);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorInAdd: " << *ms / TESTS << "ms avg." << std::endl;
}

void testVectorSub(int *a, int *b, int *c, int TESTS, int N, int WARPS, int *ms)
{
    populate(a, b, N);
    for (int i=0; i<TESTS; i++)
    {
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorSub(a, b, c, N, WARPS);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorSub:   " << *ms / TESTS << "ms avg." << std::endl;
}

void testVectorInSub(int *a, int *b, int TESTS, int N, int WARPS, int *ms)
{
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorInSub(a, b, N, WARPS);
        auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
        *ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorInSub: " << *ms / TESTS << "ms avg." << std::endl;
}

void populate(int *array_a, int *array_b, int n)
{
    for (int i=0; i<n; i++)
    {
        array_a[i] = i;
        array_b[i] = i*2;
    }
}

void validate(int *a, int *b, int n)
{
    for (int i=0; i<n; i++)
    {
        if (a[i] != b[i])
        {
            std::cout << "Arrays diverge at index: " << i << "." << std::endl;
            return;
        }
    }
    std::cout << "Arrays match." << std::endl;
}
