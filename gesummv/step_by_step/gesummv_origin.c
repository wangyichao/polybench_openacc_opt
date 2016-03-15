#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

double gesummv(int n, double alpha, double beta, double *A, double *B, double *x, double *Ay)
{
	int i,j;
	double t1 = 0;
	double t2 = 0;

	struct timeval t_time3, t_time4;
	/* Accelerator */
#pragma acc data copyin(A[0:n*n],B[0,n*n],x[n]) copyout(Ay[0:n])
	{
	gettimeofday(&t_time3, NULL);
#pragma acc kernels
		for (i = 0; i < n; i++)
		{
			t1 = 0;
			t2 = 0;
			for (j = 0; j < n; j++)
			{
				t1 += A[i*n + j] * x[j];
				t2 += B[i*n + j] * x[j];
			}
			Ay[i] = alpha * t1 + beta * t2;
		}
	gettimeofday(&t_time4, NULL);
	} /* end data */
	double excution_time = (t_time4.tv_sec - t_time3.tv_sec) + (t_time4.tv_usec - t_time3.tv_usec)*1e-6;
	return excution_time;
}

int main(int argc, char **argv) 
{

	int n = atoi(*(argv + 1));

	struct timeval t_time1, t_time2;
	double excution_time, total_time, data_time;

	double *A = (double*)malloc(n*n * sizeof(double));
	double *B = (double*)malloc(n*n * sizeof(double));
	double *x = (double*)malloc(n * sizeof(double));
	double *Ay = (double*)malloc(n * sizeof(double));

	if (A == NULL || B == NULL || x == NULL || Ay == NULL )
	{
		/* Something went wrong in the memory allocation here, fail gracefully */
		return(-10000);
	}
	
	/* Initialization */
	double alpha = 435.32;
	double beta = 123.13;
	for (int i = 0; i < n; i++)
	{
		x[i] = rand() / (1.0 + RAND_MAX);
		Ay[i]=0;
		for (int j = 0; j < n; j++)
		{
			A[i*n + j] = rand() / (1.0 + RAND_MAX);
			B[i*n + j] = rand() / (1.0 + RAND_MAX);
		}
	}

	/* Warm up */
	gesummv(16, alpha, beta, B, A, x , Ay);

	gettimeofday(&t_time1, NULL);
	excution_time = gesummv(n, alpha, beta, A, B, x , Ay);
	gettimeofday(&t_time2, NULL);

	/* Free malloc'd memory to prevent leaks */
	free(A);
	free(B);
	free(x);
	free(Ay);

	total_time = (t_time2.tv_sec - t_time1.tv_sec) + (t_time2.tv_usec - t_time1.tv_usec)*1e-6;
	data_time = total_time - excution_time;
	long long nflop = 4*n*n + 3*n;
	float intensity = 0.25;
	double flops = nflop / excution_time / 1000000000;

	printf("---gesummv---Size:%d\n", n);
	printf("---gesummv---Total Time:%f s\n", total_time);
	printf("---gesummv---Data Transfer Time:%f s\n", data_time);
	printf("---gesummv---Arithmetic Intensity:%f\n", intensity);
	printf("---gesummv---Performance:%lf Gflops\n", flops);
	
	return EXIT_SUCCESS;
}
