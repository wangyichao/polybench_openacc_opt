#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#ifndef TOL
#define TOL 1e-14
#define FDTOL 1e-12 /* See fdtd2d for justification. */
#endif
void twomm(int n, double*restrict A, double*restrict B, double*restrict C, double*restrict D, double*restrict E){

  int i = 0;
  int j = 0;
  int k = 0;
  double tmp0 = 0;
  double tmp1 = 0;
  //int flag=0;

  //double *restrict H = (double*)malloc(n*n * sizeof(double));
  /*
  * Host code for reference.
  * Store result in H.
  */
/*
  for (i = 0; i < n; i++){
  for (j = 0; j < n; j++){
  C[i*n+j] = 0;
  for (k = 0; k < n; k++){
  C[i*n+j] += A[i*n+k] * B[k*n+j];
  }
  }
  }

  for (i = 0; i < n; i++){
  for (j = 0; j < n; j++){
  H[i*n+j] = 0;
  for (k = 0; k < n; k++)
  H[i*n+j] += C[i*n+k] * D[k*n+j];
  }
  }
*/
  
  /* Accelerator code */


  /* E := A*B*C */

#pragma acc data create(C[0:n*n]) copyin(A[0:n*n],B[0:n*n],D[0:n*n]) copyout(E[0:n*n])
  {
#pragma acc kernels
  {
#pragma acc loop independent tile(32,32)
    for (i = 0; i < n; i++){
#pragma acc loop independent
      for (j = 0; j < n; j++){
        tmp0 = 0;
        for (k = 0; k < n; k++){
          tmp0 += A[i*n + k] * B[k*n + j];
        }
        C[i*n + j] = tmp0;
      }
    }
  }

#pragma acc kernels
  {
#pragma acc loop independent tile(32,32)
    for (i = 0; i < n; i++){
#pragma acc loop independent
      for (j = 0; j < n; j++){
        tmp1 = 0;
        for (k = 0; k < n; k++){
          tmp1 += C[i*n + k] * D[k*n + j];
        }
        E[i*n + j] = tmp1;
      }
    }
  }
  }
  /* end data */

  /* Compare Host + Device code. */
/* 
  for (i = 0; i < n; i++){
  for (j = 0; j < n; j++){
  if (  fabs(H[i*n+j] - E[i*n+j])/H[i*n+j] > TOL ){
  flag = 1;
  }
  }
  }
  

  if (flag == 1) printf("result is error");;
  //else return(-11000);

  free(H);
*/
}

int main(int argc, char **argv) {

  int n = atof(*(argv + 1));

  struct timeval tTime1, tTime2;
  double copy_time = 0;
  int i, j;

  /*
  * We need to hold 5 arrays on the GPU.
  * Make them all square to fit within datasize.
  */

  double *restrict C = (double*)malloc(n*n * sizeof(double));
  double *restrict A = (double*)malloc(n*n * sizeof(double));
  double *restrict B = (double*)malloc(n*n * sizeof(double));
  double *restrict D = (double*)malloc(n*n * sizeof(double));
  double *restrict E = (double*)malloc(n*n * sizeof(double));

  if (A == NULL || B == NULL || E == NULL || D == NULL){
    /* Something went wrong in the memory allocation here, fail gracefully */
    return(-10000);
  }

  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++){
      A[i*n + j] = rand() / (1.0 + RAND_MAX);
    }
  }
  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++){
      B[i*n + j] = rand() / (1.0 + RAND_MAX);
    }
  }
  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++){
      D[i*n + j] = rand() / (1.0 + RAND_MAX);
    }
  }

  twomm(16, A, B, C, D, E);
  gettimeofday(&tTime1, NULL);
  twomm(n, A, B, C, D, E);
  gettimeofday(&tTime2, NULL);

  copy_time = (tTime2.tv_sec - tTime1.tv_sec) + (tTime2.tv_usec - tTime1.tv_usec)*1e-6;

  long long nTmp = n;
  long long nFlop = 4 * nTmp*nTmp*nTmp;
  double dIntensity = 0.5;
  double flops = nFlop / copy_time / 1000000000;

  printf("---2MM---n:%d\n", n);
  printf("---2MM---Time:%lf s\n", copy_time);
  printf("---2MM---INTENSITY:%lf\n", dIntensity);
  printf("---2MM---FLOPS:%lf GFlops\n", flops);

  /* Free malloc'd memory to prevent leaks */
  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  return EXIT_SUCCESS;
}
