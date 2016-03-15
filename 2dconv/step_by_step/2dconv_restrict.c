#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

double twodconv(int n, double *restrict A, double *restrict B){

	int i, j;
	struct timeval tTime1, tTime2;
	double excution_time = 0;

	const double c11 = +2.2;  const double c21 = +5.3;  const double c31 = -8.5;
	const double c12 = -3.2;  const double c22 = +6.5;  const double c32 = -9.9;
	const double c13 = +4.8;  const double c23 = +7.2;  const double c33 = +10.3;

#pragma acc data copyin(A[0:n*n]) copyout(B[0:n*n])
	{
	gettimeofday(&tTime1, NULL);
#pragma acc kernels
		for (i = 1; i < n - 1; i++){
			for (j = 1; j < n - 1; j++){
				B[i*n + j] = c11 * A[i*n + j - n - 1] + c12 * A[i*n + j - 1] + c13 * A[i*n + j + n - 1]
					+ c21 * A[i*n + j - n] + c22 * A[i*n + j] + c23 * A[i*n + j + n]
					+ c31 * A[i*n + j - n + 1] + c32 * A[i*n + j + 1] + c33 * A[i*n + j + n + 1];

			}
		}
	gettimeofday(&tTime2, NULL);
	}
	excution_time = (tTime2.tv_sec - tTime1.tv_sec) + (tTime2.tv_usec - tTime1.tv_usec)*1e-6;

	return excution_time;
}

int main(int argc, char **argv) {

	int n = atoi(*(argv + 1));

	int i, j;
	struct timeval tTime3, tTime4;
	double data_time, total_time, excution_time;

	double *restrict A = (double*)malloc(n*n * sizeof(double));
	double *restrict B = (double*)malloc(n*n * sizeof(double));

	if (A == NULL || B == NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		return(-10000);
	}

	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			A[i*n + j] = rand() / (1.0 + RAND_MAX);
		}
	}

	twodconv(16,A,B);
	gettimeofday(&tTime3, NULL);
	excution_time = twodconv(n,A,B);
	gettimeofday(&tTime4, NULL);

	total_time = (tTime4.tv_sec - tTime3.tv_sec) + (tTime4.tv_usec - tTime3.tv_usec)*1e-6;
	data_time = total_time - excution_time;
	printf("---2DCONV---Total Time:%lf s\n", data_time);
	double size = n / 1000;
	double gflop = 17 * size * size / 1000;
	double dIntensity = 1.35;
	double flops = gflop / excution_time;

	printf("---2DCONV---Size:%d\n", n);
	printf("---2DCONV---Total Time:%lf s\n", total_time);
	printf("---2DCONV---Data Transfer Time:%lf s\n", data_time);
	printf("---2DCONV---INTENSITY:%lf\n", dIntensity);
	printf("---2DCONV---FLOPS:%lf GFlops\n", flops);

	/* Free malloc'd memory to prevent leaks */
	free(A);
	free(B);

	return EXIT_SUCCESS;
}
