#include <iostream>
#include <windows.h>
#include <intrin.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace std;

const int SIZE = 300;
const int RUNS = 10;

void matMul(vector<vector<double>>& A, vector<vector<double>>& B, vector<vector<double>>& C) {
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++) {
			C[i][j] = 0;
			for (int k = 0; k < SIZE; k++)
				C[i][j] += A[i][k] * B[k][j];
		}
}

void fillRandom(vector<vector<double>>& A, vector<vector<double>>& B, vector<vector<double>>& C) {
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++) {
			A[i][j] = rand() % 100;
			B[i][j] = rand() % 100;
			C[i][j] = 0;
		}
}

void printStats(vector<double>& data, const string& name) {
	cout << "\n=== " << name << " ===" << endl;

	cout << "Замеры: ";
	for (double v : data) cout << v << " ";
	cout << endl;

	double mn = *min_element(data.begin(), data.end());

	double s = 0;
	for (double v : data) s += v;
	double avg = s / data.size();

	double sq = 0;
	for (double v : data) sq += (v - avg) * (v - avg);
	double sigma = sqrt(sq / data.size());
	double ci = 1.96 * sigma / sqrt((double)data.size());

	cout << "tmin     = " << mn << " ms" << endl;
	cout << "tavg(T)  = " << avg << " ms" << endl;
	cout << "sigma    = " << sigma << endl;
	cout << "CI       = [" << avg - ci << "; " << avg + ci << "]" << endl;

	vector<double> filtered;
	for (double v : data)
		if (abs(v - avg) <= 3 * sigma)
			filtered.push_back(v);

	double s2 = 0;
	for (double v : filtered) s2 += v;
	double avg2 = s2 / filtered.size();

	double sq2 = 0;
	for (double v : filtered) sq2 += (v - avg2) * (v - avg2);
	double sigma2 = sqrt(sq2 / filtered.size());
	double ci2 = 1.96 * sigma2 / sqrt((double)filtered.size());

	cout << "выбросов отброшено: " << data.size() - filtered.size() << endl;
	cout << "tavg(T') = " << avg2 << " ms" << endl;
	cout << "CI'      = [" << avg2 - ci2 << "; " << avg2 + ci2 << "]" << endl;

	vector<double> sorted = data;
	sort(sorted.begin(), sorted.end());
	int w = (int)ceil(data.size() * 0.95);

	double bestD = 1e18;
	int bestI = 0;
	for (int i = 0; i + w - 1 < (int)sorted.size(); i++) {
		double d = sorted[i + w - 1] - sorted[i];
		if (d < bestD) { bestD = d; bestI = i; }
	}

	double s3 = 0;
	for (int i = bestI; i < bestI + w; i++) s3 += sorted[i];
	double avg3 = s3 / w;

	cout << "tavg(T''') = " << avg3 << " ms" << endl;
	cout << "CI'''      = [" << sorted[bestI] << "; " << sorted[bestI + w - 1] << "]" << endl;
}

int main() {
	SetThreadAffinityMask(GetCurrentThread(), 1);
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

	srand(42);

	vector<double> gtc(RUNS), tsc(RUNS), qpc(RUNS);

	unsigned __int64 t1 = __rdtsc();
	Sleep(1000);
	unsigned __int64 t2 = __rdtsc();
	double cpuFreq = (double)(t2 - t1) / 1e6;
	cout << "CPU: " << cpuFreq << " MHz" << endl;

	vector<vector<double>> A(SIZE, vector<double>(SIZE));
	vector<vector<double>> B(SIZE, vector<double>(SIZE));
	vector<vector<double>> C(SIZE, vector<double>(SIZE));

	for (int i = 0; i < RUNS; i++) {
		fillRandom(A, B, C);
		DWORD g1 = GetTickCount();
		matMul(A, B, C);
		DWORD g2 = GetTickCount();
		gtc[i] = (double)(g2 - g1);

		fillRandom(A, B, C);
		unsigned __int64 r1 = __rdtsc();
		matMul(A, B, C);
		unsigned __int64 r2 = __rdtsc();
		tsc[i] = (double)(r2 - r1) / (cpuFreq * 1000.0);

		fillRandom(A, B, C);
		LARGE_INTEGER f, q1, q2;
		QueryPerformanceFrequency(&f);
		QueryPerformanceCounter(&q1);
		matMul(A, B, C);
		QueryPerformanceCounter(&q2);
		qpc[i] = (double)(q2.QuadPart - q1.QuadPart) / f.QuadPart * 1000.0;
	}

	printStats(gtc, "GetTickCount");
	printStats(tsc, "RDTSC");
	printStats(qpc, "QueryPerformanceCounter");

	system("pause");
	return 0;
}
