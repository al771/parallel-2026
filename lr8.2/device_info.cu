#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void print_cuda_devices()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    cout << "Количество CUDA-устройств: " << deviceCount << "\n" << endl;

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        cout << "=============================" << endl;
        cout << "Устройство #" << dev << ": " << prop.name << endl;
        cout << "=============================" << endl;

        cout << "Вычислительные возможности:          "
             << prop.major << "." << prop.minor << endl;

        cout << "Объём глобальной памяти:             "
             << prop.totalGlobalMem / (1024 * 1024) << " МБ" << endl;

        cout << "Объём памяти констант:               "
             << prop.totalConstMem << " Б" << endl;

        cout << "Разделяемая память на блок:          "
             << prop.sharedMemPerBlock << " Б" << endl;

        cout << "Число регистров на блок:             "
             << prop.regsPerBlock << endl;

        cout << "Размер WARP:                         "
             << prop.warpSize << endl;

        cout << "Макс. потоков в блоке:               "
             << prop.maxThreadsPerBlock << endl;

        cout << "Макс. размерность блока потоков:     "
             << prop.maxThreadsDim[0] << " x "
             << prop.maxThreadsDim[1] << " x "
             << prop.maxThreadsDim[2] << endl;

        cout << "Макс. размерность сетки блоков:      "
             << prop.maxGridSize[0] << " x "
             << prop.maxGridSize[1] << " x "
             << prop.maxGridSize[2] << endl;

        cout << "Число потоковых мультипроцессоров:   "
             << prop.multiProcessorCount << endl;

        cout << "Асинхронных движков:                 "
             << prop.asyncEngineCount << endl;

        cout << "Параллельное копирование и расчёты:  "
             << (prop.asyncEngineCount > 0 ? 1 : 0) << endl;

        cout << "Ширина шины памяти:                  "
             << prop.memoryBusWidth << " бит" << endl;

        cout << "L2-кэш:                              "
             << prop.l2CacheSize / 1024 << " КБ" << endl;

        cout << endl;
    }
}
