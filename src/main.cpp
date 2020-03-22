
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>

#include "..\include\main.hpp"
#include "..\include\math.hpp"

int main(int argc, char *argv[])
{
    // Get TESTS
    int TESTS = 10;
    getArg(argc, argv, 1, &TESTS);
    
    // Get N
    int power = 6;
    getArg(argc, argv, 2, &power);
    int N = pow(10, power);
    
    // Get WARPS
    int WARPS = 4;
    getArg(argc, argv, 3, &WARPS);

    // Display config
    std::cout << "Config: " << TESTS << " tests, 10^" << power << " vector size, " << WARPS << " warps per block." << std::endl;

    // Test engine
    int *a = new int[N], *b = new int[N], *c = new int[N];
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
    int ms;

    // WARMUP

    std::cerr << "Warming up..." << std::endl;
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        cudamath::vectorSub(a, b, c, N, WARPS);
    }

    // ADD

    ms = 0;
    std::cerr << "vectorAdd:   running..." << std::endl;
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorAdd(a, b, c, N, WARPS);
        end = std::chrono::high_resolution_clock::now().time_since_epoch();
        ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorAdd:   " << ms / TESTS << "ms avg." << std::endl;
    
    ms = 0;
    std::cerr << "vectorInAdd: running..." << std::endl;
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorInAdd(a, b, N, WARPS);
        end = std::chrono::high_resolution_clock::now().time_since_epoch();
        ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorInAdd: " << ms / TESTS << "ms avg." << std::endl;

    // SUB

    ms = 0;
    std::cerr << "vectorSub:   running..." << std::endl;
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorSub(a, b, c, N, WARPS);
        end = std::chrono::high_resolution_clock::now().time_since_epoch();
        ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorSub:   " << ms / TESTS << "ms avg." << std::endl;
    
    ms = 0;
    std::cerr << "vectorInSub: running..." << std::endl;
    for (int i=0; i<TESTS; i++)
    {
        populate(a, b, N);
        start = std::chrono::high_resolution_clock::now().time_since_epoch();
        cudamath::vectorInSub(a, b, N, WARPS);
        end = std::chrono::high_resolution_clock::now().time_since_epoch();
        ms += (end.count()-start.count()) / 1000000;
    }
    std::cout << "vectorInSub: " << ms / TESTS << "ms avg.";

    // Cleanup
    delete[] a;
    delete[] b;
    delete[] c;

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
