#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

const size_t BLOCK_MB = 100;
const size_t BLOCK_BYTES = BLOCK_MB * 1024 * 1024;
const int REPEATS = 5;

static float measureCopy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(dst, src, bytes, kind);

    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; r++)
        cudaMemcpy(dst, src, bytes, kind);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double totalBytes = static_cast<double>(bytes) * REPEATS;
    double bw = totalBytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);

    return static_cast<float>(bw);
}

void run_bandwidth_test()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << deviceCount << " CUDA device(s) found" << endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "GPU 0: " << prop.name << "\n" << endl;

    cout << "Размер тестового блока: " << BLOCK_MB << " МБ" << endl;
    cout << "Число повторений:       " << REPEATS << "\n" << endl;

    float *h_src = (float*)malloc(BLOCK_BYTES);
    float *h_dst = (float*)malloc(BLOCK_BYTES);

    cout << "RAM allocating... ";
    if (!h_src || !h_dst) { cout << "FAIL" << endl; return; }
    cout << "OK" << endl;

    for (size_t i = 0; i < BLOCK_BYTES / sizeof(float); i++)
        h_src[i] = (float)(i % 1000);

    float *d_buf1 = nullptr;
    float *d_buf2 = nullptr;

    cudaMalloc((void**)&d_buf1, BLOCK_BYTES);
    cudaMalloc((void**)&d_buf2, BLOCK_BYTES);

    cout << "GPU global RAM allocating... ";
    if (!d_buf1 || !d_buf2) { cout << "FAIL" << endl; return; }
    cout << "OK\n" << endl;

    float bw;

    bw = measureCopy(h_dst, h_src, BLOCK_BYTES, cudaMemcpyHostToHost);
    cout << "Copying Host -> Host" << endl;
    cout << "Average bandwidth = " << bw << " GB/s\n" << endl;

    bw = measureCopy(d_buf1, h_src, BLOCK_BYTES, cudaMemcpyHostToDevice);
    cout << "Copying Host -> Device" << endl;
    cout << "Average bandwidth = " << bw << " GB/s\n" << endl;

    bw = measureCopy(h_dst, d_buf1, BLOCK_BYTES, cudaMemcpyDeviceToHost);
    cout << "Copying Device -> Host" << endl;
    cout << "Average bandwidth = " << bw << " GB/s\n" << endl;

    bw = measureCopy(d_buf2, d_buf1, BLOCK_BYTES, cudaMemcpyDeviceToDevice);
    cout << "Copying Device -> Device" << endl;
    cout << "Average bandwidth = " << bw << " GB/s\n" << endl;

    float *h_pinned_src = nullptr;
    float *h_pinned_dst = nullptr;

    cudaMallocHost((void**)&h_pinned_src, BLOCK_BYTES);
    cudaMallocHost((void**)&h_pinned_dst, BLOCK_BYTES);

    cout << "RAM page locked allocating... ";
    if (!h_pinned_src || !h_pinned_dst) { cout << "FAIL" << endl; return; }
    cout << "OK\n" << endl;

    for (size_t i = 0; i < BLOCK_BYTES / sizeof(float); i++)
        h_pinned_src[i] = (float)(i % 1000);

    bw = measureCopy(d_buf1, h_pinned_src, BLOCK_BYTES, cudaMemcpyHostToDevice);
    cout << "Copying Host -> Device page locked" << endl;
    cout << "Average bandwidth = " << bw << " GB/s\n" << endl;

    bw = measureCopy(h_pinned_dst, d_buf1, BLOCK_BYTES, cudaMemcpyDeviceToHost);
    cout << "Copying Device -> Host page locked" << endl;
    cout << "Average bandwidth = " << bw << " GB/s\n" << endl;

    cout << "Done" << endl;

    free(h_src);
    free(h_dst);
    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFreeHost(h_pinned_src);
    cudaFreeHost(h_pinned_dst);
}
