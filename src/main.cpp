
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>
#include "..\include\main.h"
#include "..\include\math.h"

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
    const int N = pow(10, 8);

    // CPU add
    int *cpu_a = new int[N], *cpu_b = new int[N], *cpu_c = new int[N];
    populate(cpu_a, cpu_b, N);

    std::cout << "CPU...";
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch();
    cpuAdd(cpu_a, cpu_b, cpu_c, N);
    auto end = std::chrono::high_resolution_clock::now().time_since_epoch();
    std::cout << "cpuAdd time: " << (end.count()-start.count())/1000000 << "ms." << std::endl;

    delete[] cpu_a;
    delete[] cpu_b;

    // GPU add
    int *gpu_a = new int[N], *gpu_b = new int[N], *gpu_c = new int[N];
    populate(gpu_a, gpu_b, N);

    std::cout << "GPU...";
    start = std::chrono::high_resolution_clock::now().time_since_epoch();
    gpuAdd(gpu_a, gpu_b, gpu_c, N, warps);
    end = std::chrono::high_resolution_clock::now().time_since_epoch();
    std::cout << "gpuAdd time: " << (end.count()-start.count())/1000000 << "ms." << std::endl;

    delete[] gpu_a;
    delete[] gpu_b;

    // Validate GPU output with CPU output
    validate(cpu_c, gpu_c, N);

    // Cleanup
    delete[] cpu_c;
    delete[] gpu_c;

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
            std::cout << "GPU-CPU diverge at index: " << i << "." << std::endl;
            return;
        }
    }
    std::cout << "GPU-CPU match." << std::endl;
}