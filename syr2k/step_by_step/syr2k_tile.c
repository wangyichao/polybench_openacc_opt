#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

double syr2k(int n, int alpha, int beta, double *restrict A, double *restrict B, double *restrict CG)
{

	int i = 0;
	int j = 0;
	int k = 0;
	double tmp = 0;

	struct timeval tTime3, tTime4;

	/* ACCELERATOR */
#pragma acc data copy(CG[0:n*n]) copyin(A[0:n*n],B[0:n*n])
{
	gettimeofday(&tTime3, NULL);
#pragma acc kernels 
#pragma acc loop independent collapse(2)
		for (i = 0; i < n; i++){
			for (j = 0; j < n; j++){
				CG[i*n + j] *= beta;
			}
		}

#pragma acc kernels
#pragma acc loop independent tile(32,32)
		for (i = 0; i < n; i++){
			for (j = 0; j < n; j++){
				tmp = 0;
#pragma acc loop independent
				for (k = 0; k < n; k++){
					tmp += (alpha * A[i*n + k] * B[j*n + k]) + (alpha * B[i*n + k] * A[j*n + k]);
				}
				CG[i*n + j] += tmp;
			}
		}
	gettimeofday(&tTime4, NULL);
	} /* end kernel */
	double excution_time = (tTime4.tv_sec - tTime3.tv_sec) + (tTime4.tv_usec - tTime3.tv_usec)*1e-6;
 
	return excution_time;
}

int main(int argc, char **argv) {

	int n = atoi(*(argv + 1));

	struct timeval tTime1, tTime2;
	double excution_time, data_time;

	/* Array declaration */
	double *restrict A = (double*)malloc(n*n * sizeof(double));
	double *restrict B = (double*)malloc(n*n * sizeof(double));
	double *restrict CG = (double*)malloc(n*n * sizeof(double));

	if (A == NULL || B == NULL || CG == NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		return(-10000);
	}


	const int alpha = 124.35;
	const int beta = 45.46;

	/* Init */
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			A[i*n + j] = rand() / (1.0 + RAND_MAX);
			B[i*n + j] = rand() / (1.0 + RAND_MAX);
			CG[i*n + j] = rand() / (1.0 + RAND_MAX);
		}
	}

	syr2k(32, alpha, beta, A, B, CG);
	gettimeofday(&tTime1, NULL);
	excution_time = syr2k(n, alpha, beta, A, B, CG);
	gettimeofday(&tTime2, NULL);

	data_time = (tTime2.tv_sec - tTime1.tv_sec) + (tTime2.tv_usec - tTime1.tv_usec)*1e-6;
	data_time = data_time - excution_time;
	double short_n = n/1000;
	double nFlop = 6 * short_n * short_n * short_n + short_n * short_n;
	float intensity = 0.7;
	double flops = nFlop / excution_time;

	printf("---SYR2K---n:%d\n", n);
	printf("---SYR2K---Excution Time:%lf s\n", data_time);
	printf("---SYR2K---Arithemtic Intensity:%lf\n", intensity);
	printf("---SYR2K---FLOPS:%lf GFLOPS\n", flops);

	/* Free malloc'd memory to prevent leaks */
	free(A);
	free(B);
	free(CG);

	return EXIT_SUCCESS;
}

