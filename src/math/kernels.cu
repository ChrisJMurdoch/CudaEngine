
// Directly included in math.cu to avoid loss of device code optimisation

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

    __global__
    void transpose(int *in, int *out, int height, int width)
    {
        out[threadIdx.x] = in[ (threadIdx.x*width) % (height*width) + ((threadIdx.x)/height) ];
    }
}