#include <iostream>
#include <intrin.h>
#include <string>
#include <windows.h>

using namespace std;

bool getBit(int val, int bit) {
    return (val >> bit) & 1;
}

int getField(int val, int from, int count) {
    return (val >> from) & ((1 << count) - 1);
}

string cacheTypeName(int t) {
    if (t == 1) return "Data";
    if (t == 2) return "Instruction";
    if (t == 3) return "Unified";
    return "Unknown";
}

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    int info[4];

    // EAX=0: производитель и макс. лист
    __cpuid(info, 0);
    int maxLeaf = info[0];

    char vendor[13] = {};
    memcpy(vendor + 0, &info[1], 4);
    memcpy(vendor + 4, &info[3], 4);
    memcpy(vendor + 8, &info[2], 4);

    cout << "=== Производитель ===" << endl;
    cout << "Vendor ID: " << vendor << endl;
    cout << "Макс. базовый лист: " << hex << maxLeaf << dec << endl;

    // EAX=80000000h: макс. расширенный лист
    __cpuid(info, 0x80000000);
    int maxExtLeaf = info[0];

    // CPU brand string
    if (maxExtLeaf >= 0x80000004) {
        char brand[49] = {};
        __cpuid((int*)&brand[0],  0x80000002);
        __cpuid((int*)&brand[16], 0x80000003);
        __cpuid((int*)&brand[32], 0x80000004);
        cout << "Название: " << brand << endl;
    }

    // EAX=16h: тактовые частоты
    cout << "\n=== Тактовые частоты ===" << endl;
    if (maxLeaf >= 0x16) {
        __cpuid(info, 0x16);
        int baseFreq = info[0] & 0xFFFF;
        int maxFreq  = info[1] & 0xFFFF;
        int busFreq  = info[2] & 0xFFFF;
        cout << "Базовая частота: " << baseFreq << " МГц" << endl;
        cout << "Максимальная частота (Boost): " << maxFreq << " МГц" << endl;
        cout << "Частота шины: " << busFreq << " МГц" << endl;
    } else {
        cout << "Лист 0x16 не поддерживается" << endl;
    }

    // EAX=1: версия и базовые расширения
    cout << "\n=== Версия процессора ===" << endl;
    if (maxLeaf >= 1) {
        __cpuid(info, 1);
        int stepping = getField(info[0], 0, 4);
        int model    = getField(info[0], 4, 4);
        int family   = getField(info[0], 8, 4);
        int extModel  = getField(info[0], 16, 4);
        int extFamily = getField(info[0], 20, 8);
        cout << "Family: "         << family    << endl;
        cout << "Model: "          << model     << endl;
        cout << "Stepping: "       << stepping  << endl;
        cout << "Extended Model: " << extModel  << endl;
        cout << "Extended Family: "<< extFamily << endl;

        int maxLogical = getField(info[1], 16, 8);
        cout << "\n=== Ядра и потоки ===" << endl;
        cout << "Макс. логических процессоров: " << maxLogical << endl;
        cout << "Hyper-Threading: " << (getBit(info[3], 28) ? "ДА" : "НЕТ") << endl;

        cout << "\n=== Расширения (EAX=1, EDX) ===" << endl;
        cout << "FPU:  " << (getBit(info[3], 0)  ? "ДА" : "НЕТ") << endl;
        cout << "TSC:  " << (getBit(info[3], 4)  ? "ДА" : "НЕТ") << endl;
        cout << "MMX:  " << (getBit(info[3], 23) ? "ДА" : "НЕТ") << endl;
        cout << "SSE:  " << (getBit(info[3], 25) ? "ДА" : "НЕТ") << endl;
        cout << "SSE2: " << (getBit(info[3], 26) ? "ДА" : "НЕТ") << endl;

        cout << "\n=== Расширения (EAX=1, ECX) ===" << endl;
        cout << "SSE3:   " << (getBit(info[2], 0)  ? "ДА" : "НЕТ") << endl;
        cout << "SSSE3:  " << (getBit(info[2], 9)  ? "ДА" : "НЕТ") << endl;
        cout << "FMA3:   " << (getBit(info[2], 12) ? "ДА" : "НЕТ") << endl;
        cout << "SSE4.1: " << (getBit(info[2], 19) ? "ДА" : "НЕТ") << endl;
        cout << "SSE4.2: " << (getBit(info[2], 20) ? "ДА" : "НЕТ") << endl;
        cout << "AVX:    " << (getBit(info[2], 28) ? "ДА" : "НЕТ") << endl;
    }

    // EAX=7, ECX=0: расширенные расширения
    if (maxLeaf >= 7) {
        __cpuidex(info, 7, 0);
        cout << "\n=== Расширения (EAX=7, EBX) ===" << endl;
        cout << "AVX2:    " << (getBit(info[1], 5)  ? "ДА" : "НЕТ") << endl;
        cout << "RTM/TSX: " << (getBit(info[1], 11) ? "ДА" : "НЕТ") << endl;
        cout << "AVX512F: " << (getBit(info[1], 16) ? "ДА" : "НЕТ") << endl;
        cout << "SHA:     " << (getBit(info[1], 29) ? "ДА" : "НЕТ") << endl;

        cout << "\n=== Расширения (EAX=7, ECX) ===" << endl;
        cout << "GFNI: " << (getBit(info[2], 8) ? "ДА" : "НЕТ") << endl;
    }

    // EAX=80000001h: AMD-специфичные расширения
    if (maxExtLeaf >= 0x80000001) {
        __cpuid(info, 0x80000001);
        cout << "\n=== Расширения AMD (EAX=80000001h) ===" << endl;
        cout << "SSE4a:  " << (getBit(info[2], 6)  ? "ДА" : "НЕТ") << endl;
        cout << "FMA4:   " << (getBit(info[2], 16) ? "ДА" : "НЕТ") << endl;
        cout << "3DNow!: " << (getBit(info[3], 31) ? "ДА" : "НЕТ") << endl;
        cout << "3DNow!+:" << (getBit(info[3], 30) ? "ДА" : "НЕТ") << endl;
    }

    // EAX=4: кэш-память
    cout << "\n=== Кэш-память (EAX=4) ===" << endl;
    for (int sub = 0; ; sub++) {
        __cpuidex(info, 4, sub);
        int cacheType = getField(info[0], 0, 5);
        if (cacheType == 0) break;

        int level     = getField(info[0], 5, 3);
        int threads   = getField(info[0], 14, 12) + 1;
        int lineSize  = getField(info[1], 0, 12) + 1;
        int partitions= getField(info[1], 12, 10) + 1;
        int ways      = getField(info[1], 22, 10) + 1;
        int sets      = info[2] + 1;
        bool inclusive= getBit(info[3], 1);

        int sizeKB = (lineSize * partitions * ways * sets) / 1024;

        cout << "\nL" << level << " " << cacheTypeName(cacheType) << ":" << endl;
        cout << "  Размер: " << sizeKB << " KB" << endl;
        cout << "  Строка кэша: " << lineSize << " байт" << endl;
        cout << "  Ассоциативность: " << ways << "-way" << endl;
        cout << "  Разделяют ядер: " << threads << endl;
        cout << "  Организация: " << (inclusive ? "Инклюзивный" : "Эксклюзивный") << endl;
    }

    cout << endl;
    system("pause");
    return 0;
}
