
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>
#include "..\include\main.h"
#include "..\include\math.cuh"

int main (int argc, char *argv[])
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
    std::cout << "Configured to: " << warps << " warps (" << warps*32 << " threads) per thread-block." << std::endl;

    // Test library
    const int n = pow(10, 6);

    // CPU add
    int *cpu_a = new int[n];
    int *cpu_b = new int[n];
    int *cpu_c = new int[n];
    populate(cpu_a, cpu_b, n);

    std::cout << "cpu_add time: ";
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
    cpu_add(cpu_a, cpu_b, cpu_c, n);
    auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
    std::cout << (end.count()-start.count())/1000000 << "ms" << std::endl;

    delete[] cpu_a;
    delete[] cpu_b;

    // GPU add
    int *gpu_a = new int[n];
    int *gpu_b = new int[n];
    int *gpu_c = new int[n];
    populate(gpu_a, gpu_b, n);

    std::cout << "gpu_add time: ";
    start = std::chrono::high_resolution_clock::now().time_since_epoch();
    gpu_add(gpu_a, gpu_b, gpu_c, n, warps);
    end = std::chrono::high_resolution_clock::now().time_since_epoch();
    std::cout << (end.count()-start.count())/1000000 << "ms" << std::endl;

    delete[] gpu_a;
    delete[] gpu_b;

    // Validate
    for (int i=0; i<n; i++)
    {
        if (cpu_c[i] != gpu_c[i])
        {
            std::cout << "GPU-CPU difference at index: " << i;
            break;
        }
    }

    // Cleanup
    delete[] cpu_c;
    delete[] gpu_c;
    return 0;
}

/** Helper function for testing array math */
void populate (int *array_a, int *array_b, int size)
{
    for (int i=0; i<size; i++)
    {
        array_a[i] = i;
        array_b[i] = i*2;
    }
}
