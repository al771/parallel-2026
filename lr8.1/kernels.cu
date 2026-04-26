#include <cuda.h>
#include <cuda_runtime.h>

__global__ void VecMulKernel(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
        c[idx] = a[idx] * b[idx];
}

__global__ void VecAddKernel(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

void vec_mul_gpu(float *a, float *b, float *c, int n)
{
    int bytes = n * sizeof(float);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    cudaMalloc((void **)&dev_a, bytes);
    cudaMalloc((void **)&dev_b, bytes);
    cudaMalloc((void **)&dev_c, bytes);

    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(512);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    VecMulKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void vec_add_gpu(float *a, float *b, float *c, int n)
{
    int bytes = n * sizeof(float);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    cudaMalloc((void **)&dev_a, bytes);
    cudaMalloc((void **)&dev_b, bytes);
    cudaMalloc((void **)&dev_c, bytes);

    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(512);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    VecAddKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
