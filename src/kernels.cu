
namespace kernels
{
    __global__
    void vectorAdd(int *a, int *b, int *c, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            c[i] = a[i] + b[i];
    }

    __global__
    void vectorInAdd(int *a, int *b, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            a[i] = a[i] + b[i];
    }

    __global__
    void vectorSub(int *a, int *b, int *c, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            c[i] = a[i] - b[i];
    }

    __global__
    void vectorInSub(int *a, int *b, int n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i=index; i<n; i += blockDim.x*gridDim.x)
            a[i] = a[i] - b[i];
    }
}