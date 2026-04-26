#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

const int N_DEFAULT = 512;
const int S_DEFAULT = 16;

__global__ void MatMulNaive(float *A, float *B, float *C, int n)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void MatMulCacheRow(float *A, float *B, float *C, int n)
{
    int col = threadIdx.x;
    int row = blockIdx.x;

    extern __shared__ float rowA[];
    rowA[col] = A[row * n + col];
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += rowA[k] * B[k * n + col];

    C[row * n + col] = sum;
}

__global__ void MatMulCacheCol(float *A, float *B, float *C, int n)
{
    int row = threadIdx.x;
    int col = blockIdx.x;

    extern __shared__ float colB[];
    colB[row] = B[row * n + col];
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * colB[k];

    C[row * n + col] = sum;
}

template <int S>
__global__ void MatMulTiled(float *A, float *B, float *C, int n)
{
    __shared__ float tileA[S][S];
    __shared__ float tileB[S][S];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * S + tx;
    int row = blockIdx.y * S + ty;

    float sum = 0.0f;

    for (int t = 0; t < (n + S - 1) / S; t++)
    {
        if (row < n && t * S + tx < n)
            tileA[ty][tx] = A[row * n + t * S + tx];
        else
            tileA[ty][tx] = 0.0f;

        if (col < n && t * S + ty < n)
            tileB[ty][tx] = B[(t * S + ty) * n + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < S; k++)
            sum += tileA[ty][k] * tileB[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

void matMulCPU(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            float s = 0.0f;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

bool matricesEqual(float *A, float *B, int n)
{
    for (int i = 0; i < n * n; i++)
        if (fabs(A[i] - B[i]) > 1e-2f) return false;
    return true;
}

void run_matmul_test()
{
    int n = N_DEFAULT;
    size_t bytes = (size_t)n * n * sizeof(float);

    float *A = new float[n * n];
    float *B = new float[n * n];
    float *Cref = new float[n * n];
    float *Cgpu = new float[n * n];

    for (int i = 0; i < n * n; i++)
    {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    cout << "CPU..." << endl;
    matMulCPU(A, B, Cref, n);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(n / 16, n / 16);

    MatMulNaive<<<grid, block>>>(dA, dB, dC, n);

    cudaMemcpy(Cgpu, dC, bytes, cudaMemcpyDeviceToHost);

    cout << "Проверка: " << (matricesEqual(Cgpu, Cref, n) ? "OK" : "ERROR") << endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] A;
    delete[] B;
    delete[] Cref;
    delete[] Cgpu;
}
