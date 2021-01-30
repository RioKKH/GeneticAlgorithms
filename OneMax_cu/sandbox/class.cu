#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <helper_functions.h>

__global__ void gpuKernel1(int N, float* C, float* A)
{
	for (int i = 0; i < N; i++) {
		C[i] = A[i];
		printf("%d\n", i);
	}
}

__global__ void gpuKernel2(int N, float* C)
{
	for (int i = 0; i < N; i++) {
		printf("gpuKernel2 %d, %d, %d, %f\n",
				threadIdx.x, blockIdx.x, blockDim.x, C[i]);
	}
}


class Population
{
public:
	Population();
	~Population();
	Individual **ind;
};

class Individual
{
public:
	Individual();
	~Individual();
	int* d_chromosome;
	int* h_chromosome;
};


