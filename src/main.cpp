
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>

#include "..\include\main.hpp"
#include "..\include\math.hpp"

int main(int argc, char *argv[])
{
    // Get warps
    int warps = 4;
    if (argc > 1)
    {
        try
        {
            warps = std::stoi(argv[1]);
        }
        catch (std::invalid_argument const &e)
        {
            std::cout << "Parameter parsing error." << std::endl;
            return 0;
        }
    }
    std::cout << std::endl << "Configured to: " << warps << " warps (" << warps*32 << " threads) per thread-block." << std::endl;

    // Test library
    const int N = 3*pow(10, 8);
    int *a = new int[N], *b = new int[N], *c = new int[N];
    populate(a, b, N);
    
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
    cudamath::vectorAdd(a, b, c, N, warps);
    auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
    std::cout << "gpuAdd   time: " << (end.count()-start.count())/1000000 << "ms." << std::endl;
    
    start = std::chrono::high_resolution_clock::now().time_since_epoch();
    cudamath::vectorInAdd(a, b, N, warps);
    end = std::chrono::high_resolution_clock::now().time_since_epoch();
    std::cout << "gpuInAdd time: " << (end.count()-start.count())/1000000 << "ms." << std::endl;
    
    // Validate outputs against each other
    validate(a, c, N);

    // Cleanup
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

void populate(int *array_a, int *array_b, int n)
{
    for (int i=0; i<n; i++)
    {
        array_a[i] = i;
        array_b[i] = i*2;
    }
}

void validate(int *cpu, int *gpu, int n)
{
    for (int i=0; i<n; i++)
    {
        if (cpu[i] != gpu[i])
        {
            std::cout << "Arrays diverge at index: " << i << "." << std::endl;
            return;
        }
    }
    std::cout << "Arrays match." << std::endl;
}
