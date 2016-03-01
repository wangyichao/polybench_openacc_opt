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

void atax(int n,int blocksize, double *A,double *x,double *ya){
	int blocknum = n / blocksize;
	int block = 0;

	int  i, j;
	double *tmp = NULL;
	tmp = (double *)malloc(sizeof(double)*n);

#pragma acc data create(A[0:n*n],tmp[0:n],ya[0:n]) copyin(x[0:n])
{
	for (block = 0; block < blocknum; block++)
	{
		int ystart = block*blocksize;
		int yend = ystart + blocksize;
#pragma acc update device (A[ystart*n:b*blocksize])  async(block%4)
#pragma acc parallel loop async(block%4)
		for (i = ystart; i < yend; i++)
		{
			for (j = 0; j < n; j++)
			{
				tmp[i] += A[i*n + j] * x[j];
			}
		}

#pragma acc parallel loop async(block%4)
		for (i = ystart; i < yend; i++)
		{
			for (j = 0; j < n; j++)
			{
				ya[i] += A[i*n + j] * x[j];
			}
		}
#pragma acc update self(ya[ystart:blocksize]) async(block%4)
	}
#pragma acc wait
}
	free(tmp);
}

int main(int argc, char **argv) {

	int n = atoi(*(argv + 1));
	int blocksize = (argc>1) ? atoi(argv[2]) : 8;

	struct timeval tTime1, tTime2;
	double copy_time = 0;


	double t_end = 0;
	double t_start = 0;
	int flag = 0;

	double *A = NULL;
	double *x = NULL;
	double *ya = NULL;
	double *tmp = NULL;

	A = (double *)malloc(sizeof(double)*n*n);
	x = (double *)malloc(sizeof(double)*n);
	ya = (double *)malloc(sizeof(double)*n);

	if (A == NULL || x == NULL   || ya == NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		return(-10000);
	}
	
	for (int i = 0; i < n; i++){
		x[i] = rand() / (1.0 + RAND_MAX);
		ya[i] = 0;
		for (int j = 0; j < n; j++){
			A[i*n + j] = rand() / (1.0 + RAND_MAX);
		}
	}




	atax(n, blocksize, A, x, ya);


	gettimeofday(&tTime1, NULL);
	atax(n, blocksize, A, x, ya);
	gettimeofday(&tTime2, NULL);




	copy_time = (tTime2.tv_sec - tTime1.tv_sec) + (tTime2.tv_usec - tTime1.tv_usec)*1e-6;
	long long nFlop = 4 * n*n;
	long long nMa = 8 * (n*n + 5 * n);
	double dIntensity = (nFlop + 0.0) / nMa;
	double flops = (nFlop / copy_time) / 1000000000;

	printf("---ATAX---n:%d\n", n);
	printf("---ATAX---FLOP:[4*n*n]:%ld\n", nFlop);
	printf("---ATAX---MA:[8*(n*n+5*n)]:%ld\n", nMa);
	printf("---ATAX---INTENSITY:%lf\n", dIntensity);
	printf("---ATAX---FLOPS:%lf GFlops\n", flops);
	printf("---ATAX---TIME:%lf s\n", copy_time);

	/* Free malloc'd memory to prevent leaks */
	free(A);
	free(x);
	free(ya);


	return EXIT_SUCCESS;
}
