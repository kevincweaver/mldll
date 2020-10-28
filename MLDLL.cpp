/* 
MLDLL.cpp 
windows library for optimized machine learning matrix operations

compile with the option -LD to produce a widows dll
eg. windows key > search "developer command prompt"
> cl MLDLL.cpp /LD 
*/
#include <malloc.h>
#define DLLEXPORT extern "C" __declspec(dllexport)
#define RESTRICT __declspec(restrict)
#define ALLIGNMENT 64 // size of L1 cache line 
#define BLOCKSIZE 8 // size of L1 cache line divided by sizeof(double)


// creates a transpose of a matrix
DLLEXPORT RESTRICT double* c_transp(const double* __restrict matrix, int n, int m) {
	double* results = (double*)_aligned_malloc(sizeof(double) * n * m, ALLIGNMENT);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			results[j*n + i] = matrix[i*m + j];
		}
	}
	return results;
}

// creates a transpose of a matrix via a blocked algorithm
DLLEXPORT RESTRICT double* c_transp2(const double* __restrict matrix, int n, int m) {
	double* results = (double*)_aligned_malloc(sizeof(double) * n * m, ALLIGNMENT);
	for (int jj = 0; jj < n; jj = jj + BLOCKSIZE)
	for (int kk = 0; kk < m; kk = kk + BLOCKSIZE)
		for (int j = jj; (j < jj + BLOCKSIZE) && (j < n); j++) {
			for (int k = kk; (k < kk + BLOCKSIZE) && (k < m); k++) {
				results[k*n + j] = matrix[j*m + k];
			}
		}
	return results;
}

// multiplies two matricies
DLLEXPORT double* c_mmult(const double* __restrict matrixA, int nA, int mA,
	                      const double* __restrict matrixB, int nB, int mB) {
	double* results = (double*)_aligned_malloc(sizeof(double) * nA * mB, ALLIGNMENT);
	for (int i = 0; i < nA; i++) {
		for (int j = 0; j < mB; j++) {
			results[i*mB + j] = 0;
			for (int k = 0; k < nB; k++) {
				results[i*mB + j] += matrixA[i*mA + k] * matrixB[k*mB + j];
			}
		}
	}
	return results;
}

// multiplies two matricies with an intermediate transpose step 
DLLEXPORT double* c_mmult2(const double* __restrict matrixA, int nA, int mA,
	                       const double* __restrict matrixB, int nB, int mB) {
	double* results = (double*)_aligned_malloc(sizeof(double) * nA * mB, ALLIGNMENT);
	double* matrixBT = c_transp2(matrixB, nB, mB);
	for (int i = 0; i < nA; i++) {
		for (int j = 0; j < mB; j++) {
			results[i*mB + j] = 0;
			for (int k = 0; k < nB; k++) {
				results[i*mB + j] += matrixA[i*mA + k] * matrixBT[j*nB + k];
			}
		}
	}
	return results;
}

// multiplies two matricies with a blocked algorithm 
DLLEXPORT double* c_mmult3(const double* __restrict matrixA, int nA, int mA,
	                       const double* __restrict matrixB, int nB, int mB) {
	double* results = (double*)_aligned_malloc(sizeof(double) * nA * mB, ALLIGNMENT);
	double r;

	for (int jj = 0; jj < mB; jj = jj + BLOCKSIZE)
	for (int kk = 0; kk < nB; kk = kk + BLOCKSIZE)
	for (int i = 0; i < nA; i++) {
		for (int j = jj; ((j < jj + BLOCKSIZE) && (j < mB)); j++) {
			r = 0;
			for (int k = kk; ((k < kk + BLOCKSIZE) && (k < nB)); k++) {
				r += matrixA[i*mA + k] * matrixB[k * mB + j];
			}
			results[i*mB + j] += r;
		}
	}
	return results;
}

// multiplies two matricies with a blocked algorithm and an intermediate transpose step 
DLLEXPORT double* c_mmult4(const double* __restrict matrixA, int nA, int mA,
	                       const double* __restrict matrixB, int nB, int mB) {
	double* results = (double*)_aligned_malloc(sizeof(double) * nA * mB, ALLIGNMENT);
	double* matrixBT = c_transp2(matrixB, nB, mB);
	double r;

	for (int jj = 0; jj < mB; jj = jj + BLOCKSIZE)
		for (int kk = 0; kk < nB; kk = kk + BLOCKSIZE)
			for (int i = 0; i < nA; i++) {
				for (int j = jj; ((j < jj + BLOCKSIZE) && (j < mB)); j++) {
					r = 0;
					for (int k = kk; ((k < kk + BLOCKSIZE) && (k < nB)); k++) {
						r += matrixA[i*mA + k] * matrixBT[j*nB + k];
					}
					results[i*mB + j] += r;
				}
			}
	return results;
}

// sums rows of a matrix
DLLEXPORT double* c_sum(const double * matrix, int n, int m) {
	double* results = (double*)_aligned_malloc(sizeof(double) * n, ALLIGNMENT);
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
