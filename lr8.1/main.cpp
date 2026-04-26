#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

void vec_mul_gpu(float *a, float *b, float *c, int n);
void vec_add_gpu(float *a, float *b, float *c, int n);

void vec_add_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

const int N = 2048;

int main()
{
    float a[N], b[N], c_gpu[N], c_cpu[N], c_mul[N];

    srand(42);

    for (int i = 0; i < N; i++)
    {
        a[i] = static_cast<float>(rand() % 100) / 10.0f;
        b[i] = static_cast<float>(rand() % 100) / 10.0f;
        c_gpu[i] = 0.0f;
        c_cpu[i] = 0.0f;
        c_mul[i] = 0.0f;
    }

    vec_mul_gpu(a, b, c_mul, N);

    cout << "=== Поэлементное умножение GPU, первые 20 элементов ===" << endl;
    for (int i = 0; i < 20; i++)
        cout << c_mul[i] << " ";
    cout << endl;

    vec_add_gpu(a, b, c_gpu, N);
    vec_add_cpu(a, b, c_cpu, N);

    cout << "\n=== Поэлементное сложение GPU, первые 20 элементов ===" << endl;
    for (int i = 0; i < 20; i++)
        cout << c_gpu[i] << " ";
    cout << endl;

    float maxErr = 0.0f;

    for (int i = 0; i < N; i++)
    {
        float err = fabs(c_gpu[i] - c_cpu[i]);
        if (err > maxErr)
            maxErr = err;
    }

    cout << "\n=== Сравнение GPU и CPU результатов сложения ===" << endl;
    cout << "Максимальное отклонение: " << maxErr << endl;

    if (maxErr < 1e-4f)
        cout << "Результаты совпадают. Вычисления корректны." << endl;
    else
        cout << "ОШИБКА: результаты расходятся!" << endl;

    return 0;
}
