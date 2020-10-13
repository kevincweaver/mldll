#include <stdlib.h>

#define DLLEXPORT extern "C" __declspec(dllexport)


//  creates an n by m matrix of zeros
DLLEXPORT double * c_zeros(int n, int m) {
	double * results = (double *)malloc(sizeof(double) * n * m);
	for (int i = 0; i < n*m; i += n) {
		for (int j = 0; j < m; j++) {
			results[i + j] = 0;
		}
	}
	return results;
}

// multiplies two matricies
DLLEXPORT double * c_mmult(const double * matrixA, int nA, int mA,
	                       const double * matrixB, int nB, int mB) {
	double * results = (double *)malloc(sizeof(double) * nA * mB);
	for (int i = 0; i < nA*mB; i += nA) {
		for (int j = 0; j < mB; j++) {
			results[i + j] = 0;
			for (int k = 0; k < nA; k++) {
				results[i + j] += matrixA[i + k] * matrixB[k * nA + j];
			}
		}
	}
	return results;
}

// sums rows of a matrix
DLLEXPORT double * c_sum(const double * matrix, int n, int m) {
	double * results = (double *)malloc(sizeof(double) * n);
	int index = 0;
	for (int i = 0; i < n*m; i += n) {
		results[index] = 0;
		for (int j = 0; j < m; j++) {
			results[index] += matrix[i + j];
		}
		index += 1;
	}
	return results;
}