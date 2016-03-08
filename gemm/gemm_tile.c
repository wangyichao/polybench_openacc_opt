#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define WA 8192
#define HA 8192
#define WB 8192
#define FLOAT double 

void mm_gpu( FLOAT * restrict a, FLOAT * restrict b, FLOAT * restrict c, int wa, int ha, int wb){
	FLOAT temp=0;
	int i, j, k;
#pragma acc kernels copyin(a[0:wa*ha],b[0:wa*wb]) copyout(c[0:ha*wb]) create(temp)
	{
#pragma acc loop independent tile(32,32)
		for( i=0; i<ha; i++ ){
#pragma acc loop independent 
			for( j=0; j<wb; j++ ){
//#pragma acc cache(a[i*WA:WA])//in this place should always be a constant, it cant be the value pass by function
				temp=0;
				for( k=0; k<wa; k++ ){
					temp += a[i*wa+k] * b[k*wb+j];
				}
				c[i*wb+j] = temp;
			}
		}
	}
}

void mm_cpu( const FLOAT *a, const FLOAT *b, FLOAT *c, int wa, int ha, int wb){
	FLOAT temp=0;
	int i, j, k;
	for( i=0; i<ha; i++ ){
		for( j=0; j<wb; j++ ){
			temp=0;
			for( k=0; k<wa; k++ ){
				temp += a[i*wa+k] * b[k*wb+j];
			}
			c[i*wb+j] = temp;
		}
	}
}

int main( int argc, char **argv ){

	FLOAT *restrict a, *restrict b, *restrict c, *restrict d;
	int i, j, n_a, n_b, n_c;
	struct timeval t_start, t_end;

	n_a=WA*HA;
	n_b=WB*WA;
	n_c=WB*HA;
	a=(FLOAT*)malloc(n_a*sizeof(FLOAT));
	b=(FLOAT*)malloc(n_b*sizeof(FLOAT));
	c=(FLOAT*)malloc(n_c*sizeof(FLOAT));
	d=(FLOAT*)malloc(n_c*sizeof(FLOAT));

	srand(13);

	for( i=0; i<n_a; i++ )
		a[i]=rand()/(FLOAT)RAND_MAX;
	for( j=0; j<n_b; j++ )
		b[j]=rand()/(FLOAT)RAND_MAX;

	mm_gpu( c, a, b, WA, HA, WB );

	gettimeofday(&t_start, NULL);
	mm_gpu( a, b, c, WA, HA, WB );
	gettimeofday(&t_end, NULL);

	float cost=(t_end.tv_sec-t_start.tv_sec)*1000000+(t_end.tv_usec-t_start.tv_usec);
	cost=cost/1000000;
	int wa=WA/100;
	int ha=HA/100;
	int wb=WB/100;
	float flops=2*wa*ha*wb/cost/1000;
        printf("Time is: %f s\n", cost);
        printf("Performance is: %f GFLOPS\n", flops);
	//printf("Time is: %f s\n", cost);
	//for(int k=0;k<n_c;k++)
	//	if(c[k]!=d[k]) printf("%f \n",c[k]-d[k]);

	free(a);
	free(b);
	free(c);

	return 0;
}

