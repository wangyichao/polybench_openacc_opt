#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

#define FLOAT double 

double mm_gpu(int n, FLOAT * restrict a, FLOAT * restrict b, FLOAT * restrict c){
	
	struct timeval t_start_1, t_end_2;
	FLOAT temp=0;
	int i, j, k;

#pragma acc data copyin(a[0:n*n],b[0:n*n]) copyout(c[0:n*n])
	{
	gettimeofday(&t_start_1, NULL);
#pragma acc kernels
		for( i=0; i<n; i++ ){
			for( j=0; j<n; j++ ){
				temp=0;
				for( k=0; k<n; k++ ){
					temp += a[i*n+k] * b[k*n+j];
				}
				c[i*n+j] = temp;
			}
		}
	gettimeofday(&t_end_2, NULL);
	}
	double excution_time = (t_end_2.tv_sec-t_start_1.tv_sec)+(t_end_2.tv_usec-t_start_1.tv_usec)*1e-6;

	return excution_time;
}

int main( int argc, char **argv ){

	int n = atoi(*(argv + 1));

	FLOAT *restrict a, *restrict b, *restrict c;
	int i, j;
	struct timeval t_start_3, t_end_4;
	double excution_time, total_time, data_time;

	a=(FLOAT*)malloc(n*n*sizeof(FLOAT));
	b=(FLOAT*)malloc(n*n*sizeof(FLOAT));
	c=(FLOAT*)malloc(n*n*sizeof(FLOAT));

	srand(13);

	for( i=0; i<n*n; i++ )
		a[i]=rand()/(FLOAT)RAND_MAX;
	for( j=0; j<n*n; j++ )
		b[j]=rand()/(FLOAT)RAND_MAX;

	excution_time = mm_gpu(16, a, b, c);
	gettimeofday(&t_start_3, NULL);
	excution_time = mm_gpu(n, a, b, c);
	gettimeofday(&t_end_4, NULL);

	total_time = (t_end_4.tv_sec-t_start_3.tv_sec)+(t_end_4.tv_usec-t_start_3.tv_usec)*1e-6;
	data_time = total_time - excution_time;
	double size=n/100;
	double flops=2*size*size*size/excution_time/1000;
        printf("Total Time is: %lf s\n", total_time);
        printf("Data Transfer Time is: %lf s\n", data_time);
        printf("Performance is: %lf GFLOPS\n", flops);

	free(a);
	free(b);
	free(c);

	return 0;
}

