#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

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


void atax(int n, double *restrict A, double *restrict x, double *restrict y, double *restrict tmp){

	int i = 0;
	int j = 0;
	double tmpscalar0, tmpscalar1;

	//*****************NOTICE IN THIS PLACE VECTOR WORKER HAS DIFFERENT EFFECT
#pragma acc data create(tmp[0:n]) copyin(A[0:n*n],x[0:n]) copyout(y[0:n])
	{
#pragma acc parallel num_workers(32)
{
#pragma acc loop independent gang,worker private(tmpscalar0)
		for (i = 0; i < n; i++){
			tmpscalar0 = 0;
#pragma acc loop independent vector
			for (j = 0; j < n; j++){
				tmpscalar0 += A[i*n + j] * x[j];
			}
			tmp[i] = tmpscalar0;
		}
#pragma acc loop independent gang,worker private(tmpscalar1) 
		for (j = 0; j < n; j++){
			tmpscalar1 = 0;
#pragma acc loop independent vector 
			for (i = 0; i < n; i++){
				tmpscalar1 += A[i*n + j] * tmp[i];
			}
			y[j] = tmpscalar1;

		}
	} 
}
}

int main(int argc, char **argv) {

  int n = atoi(*(argv+1));

  struct timeval tTime1, tTime2;
  double copy_time=0;

	double *restrict A = NULL;
	double *restrict x = NULL;
	double *restrict y = NULL;
	double *restrict tmp = NULL;

	A = (double *)malloc(sizeof(double)*n*n);
	x = (double *)malloc(sizeof(double)*n);
	y = (double *)malloc(sizeof(double)*n);
	tmp = (double *)malloc(sizeof(double)*n);

	if (A == NULL || x == NULL || y == NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		return(-10000);
	}

	for (int i = 0; i < n; i++){
		x[i] = rand() / (1.0 + RAND_MAX);
		y[i] = 0;
		for (int j = 0; j < n; j++){
			A[i*n + j] = rand() / (1.0 + RAND_MAX);
		}
	}

  	atax(16, A, x, y, tmp);
	gettimeofday(&tTime1, NULL);
  	atax(n, A, x, y, tmp);
	gettimeofday(&tTime2, NULL);

	copy_time = (tTime2.tv_sec - tTime1.tv_sec) + (tTime2.tv_usec - tTime1.tv_usec)*1e-6;
	long long nFlop = 4 * n * n;
	double dIntensity = 0.5;
	double flops = (nFlop / copy_time) / 1000000000;

	printf("---ATAX---n:%d\n", n);
	printf("---ATAX---TIME:%lf s\n", copy_time);
	printf("---ATAX---INTENSITY:%lf\n", dIntensity);
	printf("---ATAX---FLOPS:%lf GFlops\n", flops);

	/* Free malloc'd memory to prevent leaks */
	free(A);
	free(x);
	free(y);

  return EXIT_SUCCESS;
}
