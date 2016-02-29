#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
/*
 * Default problem limits.
 */
#ifndef TOL
#define TOL 1e-14
#define FDTOL 1e-12 /* See fdtd2d for justification. */
#endif
#ifndef T_MAX
#define T_MAX 50
#endif

void syr2k(int n, int alpha, int beta, double *restrict A, double *restrict B, double *restrict CG)
{

	int i = 0;
	int j = 0;
	int k = 0;
	double tmp = 0;

	/* ACCELERATOR */
//#pragma acc data copy(CG[0:n*n]), copyin(A[0:n*n], B[0:n*n])
#pragma acc kernels copy(CG[0:n*n]) copyin(A[0:n*n],B[0:n*n])
	{
//#pragma acc parallel loop num_gangs(512) vector_length(32) num_workers(32)
#pragma acc loop independent //worker(32)
		for (i = 0; i < n; i++){
#pragma acc loop independent vector(32)
			for (j = 0; j < n; j++){
				CG[i*n + j] *= beta;
			}
		}
//#pragma acc parallel loop num_gangs(512) vector_length(32) num_workers(32)
#pragma acc loop independent reduction(+:tmp) //worker(32)
		for (i = 0; i < n; i++){
//#pragma acc loop private(tmp)
#pragma acc loop independent
			for (j = 0; j < n; j++){
				tmp = 0;//CG[i*n + j];
//#pragma acc loop reduction(+:tmp)
#pragma acc loop independent
				for (k = 0; k < n; k++){
					tmp += (alpha * A[i*n + k] * B[j*n + k]) + (alpha * B[i*n + k] * A[j*n + k]);
				}
				CG[i*n + j] += tmp;
			}
		}
	} /* end kernel */
}

int main(int argc, char **argv) {

	int n = atoi(*(argv + 1));

	struct timeval tTime1, tTime2;
	double time=0;

	/* Array declaration */
	double *restrict A = (double*)malloc(n*n * sizeof(double));
	double *restrict B = (double*)malloc(n*n * sizeof(double));
	double *restrict CG = (double*)malloc(n*n * sizeof(double));

	if (A == NULL || B == NULL || CG == NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		return(-10000);
	}


	const int restrict alpha = 124.35;
	const int restrict beta = 45.46;

	/* Init */
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			A[i*n + j] = rand() / (1.0 + RAND_MAX);
			B[i*n + j] = rand() / (1.0 + RAND_MAX);
			CG[i*n + j] = rand() / (1.0 + RAND_MAX);
		}
	}

	//syr2k(64, alpha, beta, A, B, CG);

	gettimeofday(&tTime1, NULL);
	syr2k(n, alpha, beta, A, B, CG);
	gettimeofday(&tTime2, NULL);

	time = (tTime2.tv_sec - tTime1.tv_sec) + (tTime2.tv_usec - tTime1.tv_usec)*1e-6;
	double short_n = n/1000;
	double nFlop = 6 * short_n * short_n * short_n + short_n * short_n;
	float intensity = 0.25;
	double flops = nFlop / time;

	printf("---SYR2K---n:%d\n", n);
	printf("---SYR2K---Excution Time:%lf s\n", time);
	printf("---SYR2K---Arithemtic Intensity:%lf\n", intensity);
	printf("---SYR2K---FLOPS:%lf GFLOPS\n", flops);

	/* Free malloc'd memory to prevent leaks */
	free(A);
	free(B);
	free(CG);
	return EXIT_SUCCESS;
}

