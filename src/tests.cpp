
#include <iostream>
#include <chrono>

#include "..\include\tests.hpp"
#include "..\include\math.hpp"

void testVectorAdd(int *a, int *b, int *c, int TESTS, int N, int *ms)
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

void testVectorInAdd(int *a, int *b, int TESTS, int N, int *ms)
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

void testVectorSub(int *a, int *b, int *c, int TESTS, int N, int *ms)
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

void testVectorInSub(int *a, int *b, int TESTS, int N, int *ms)
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

void testOutputs()
{
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
