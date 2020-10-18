#define DLLEXPORT extern "C" __declspec(dllexport)
#define BLOCKSIZE 8

// creates a transpose of a matrix
DLLEXPORT double* c_transp(const double * matrix, int n, int m) {
	double* results = new double[sizeof(double) * n * m];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			results[j*n + i] = matrix[i*m + j];
		}
	}
	return results;
}

// creates a transpose of a matrix via a blocked algorithm
DLLEXPORT double* c_transp2(const double * matrix, const int n, const int m) {
	double* results = new double[sizeof(double) * n * m];
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
DLLEXPORT double* c_mmult(const double * matrixA, int nA, int mA,
	                       const double * matrixB, int nB, int mB) {
	double* results = new double[sizeof(double) * nA * mB];
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
DLLEXPORT double* c_mmult2(const double * matrixA, int nA, int mA,
	                       const double * matrixB, int nB, int mB) {
	double* results = new double[sizeof(double) * nA * mB];
	double* matrixBT = c_transp(matrixB, nB, mB);
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
DLLEXPORT double* c_mmult3(const double * matrixA, int nA, int mA,
	                       const double * matrixB, int nB, int mB) {
	double* results = new double[sizeof(double) * nA * mB](); 
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
DLLEXPORT double * c_sum(const double * matrix, int n, int m) {
	double* results = new double[sizeof(double) * n];
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
