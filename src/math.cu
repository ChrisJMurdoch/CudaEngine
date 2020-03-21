
__global__ void addKernel(int *a, int *b, int *c, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) c[index] = a[index] + b[index];
}

/** Function for adding two arrays together on the GPU. */
void gpu_add(int *a, int *b, int *c, int n, int warps=4)
{
    // Copy into device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));
    cudaMalloc(&d_c, n*sizeof(int));
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel
    int threads = warps*32;
    addKernel<<<(n/threads)+1, threads>>>(d_a, d_b, d_c, n);

    // Retrieve device memory
    cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/** Function for adding two arrays together on the CPU */
void cpu_add(int *a, int *b, int *dest, int n)
{
    for (int i=0; i<n; i++)
        dest[i] = a[i] + b[i];
}
