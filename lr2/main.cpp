#include <iostream>
#include <windows.h>
#include <vector>
using namespace std;

double getTime() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return cnt.QuadPart / (double)freq.QuadPart;
}

void fillMatrix(vector<vector<float>>& A, vector<vector<float>>& B, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10 + 1;
            B[i][j] = rand() % 10 + 1;
        }
}

// 1. Классическое умножение
void classicMul(vector<vector<float>>& A, vector<vector<float>>& B, 
                vector<vector<float>>& C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++)
                s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
}

// 2. Умножение с транспонированием
void transposeMul(vector<vector<float>>& A, vector<vector<float>>& B, 
                  vector<vector<float>>& C, int N) {
    vector<vector<float>> Bt(N, vector<float>(N));
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Bt[i][j] = B[j][i];
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++)
                s += A[i][k] * Bt[j][k];
            C[i][j] = s;
        }
}

// 3. Умножение с буферизацией
void bufferMul(vector<vector<float>>& A, vector<vector<float>>& B, 
               vector<vector<float>>& C, int N, int M) {
    vector<float> buf(N);
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++)
            buf[k] = B[k][j];
        
        for (int i = 0; i < N; i++) {
            float s = 0;
            int k = 0;
            
            if (M >= 2)
                for (; k <= N - 2; k += 2)
                    s += A[i][k] * buf[k] + A[i][k+1] * buf[k+1];
            
            for (; k < N; k++)
                s += A[i][k] * buf[k];
            
            C[i][j] = s;
        }
    }
}

// 4. Блочное умножение
void blockMul(vector<vector<float>>& A, vector<vector<float>>& B, 
              vector<vector<float>>& C, int N, int S, int M) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0;
    
    for (int ii = 0; ii < N; ii += S)
        for (int jj = 0; jj < N; jj += S)
            for (int kk = 0; kk < N; kk += S)
                for (int i = ii; i < min(ii + S, N); i++)
                    for (int j = jj; j < min(jj + S, N); j++) {
                        float s = 0;
                        int k = kk;
                        
                        if (M >= 2)
                            for (; k <= min(kk + S, N) - 2; k += 2)
                                s += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j];
                        
                        for (; k < min(kk + S, N); k++)
                            s += A[i][k] * B[k][j];
                        
                        C[i][j] += s;
                    }
}

int main() {
    int N = 512;
    vector<vector<float>> A(N, vector<float>(N));
    vector<vector<float>> B(N, vector<float>(N));
    vector<vector<float>> C(N, vector<float>(N));
    
    fillMatrix(A, B, N);
    
    double t = getTime();
    classicMul(A, B, C, N);
    t = getTime() - t;
    
    cout << "Классическое: " << t * 1000 << " мс\n";
    
    return 0;
}
