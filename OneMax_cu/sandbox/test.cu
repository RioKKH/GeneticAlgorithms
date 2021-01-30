#include <stdio.h>
#include <numeric>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void gpukernel1(int N, int* C, int* A)
{
	for (int i = 0; i < N; i++) {
		C[i] = A[i];
		// printf("%d\n", i);
	}
}

__global__ void gpukernel2(int N, int* C)
{
	/*
	for (int i = 0; i < N; i++) {
		printf("gpuKernel2 %d, %d, %d, %d\n",
				threadIdx.x, blockIdx.x, blockDim.x, C[i]);
	}
	*/
}

/*
__device__ int selection(int *A, unsigned int random1, unsigned int random2)
{
}
*/

int divRoundUp(int value, int radix) {
	return (value + radix - 1) / radix;
}

int main(int argc, char** argv) {
	int chromosome_size = 10;
	int population_size = 10;
	int *h_A, *h_C;
	int *d_A, *d_C;

	// preparation of data on host
	h_A = (int*)malloc(sizeof(int) * chromosome_size * population_size);
	h_C = (int*)malloc(sizeof(int) * chromosome_size * population_size);

	for (int i = 0; i < (chromosome_size * population_size); i++) {
		h_A[i] = rand() % 2;
		h_C[i] = 0.0f;
	}

	// preparation of data on device
	cudaMalloc((void**)&d_A, sizeof(int) * chromosome_size * population_size);
	cudaMalloc((void**)&d_C, sizeof(int) * chromosome_size * population_size);
	cudaMemcpy(d_A, h_A, sizeof(int) * (chromosome_size * population_size), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(int) * (chromosome_size * population_size), cudaMemcpyHostToDevice);

	dim3 blockDim(10, 2);
	dim3 gridDim(divRoundUp(chromosome_size, blockDim.x), divRoundUp(population_size, blockDim.y));

	gpukernel1<<<gridDim, blockDim>>>(chromosome_size * population_size, d_C, d_A);
	cudaDeviceSynchronize();

	gpukernel2<<<gridDim, blockDim>>>(chromosome_size * population_size, d_C);
	cudaMemcpy(h_C, d_C, sizeof(int)*chromosome_size * population_size, cudaMemcpyDeviceToHost);

	for (int y = 0; y < population_size; y++) {
		for (int x = 0; x < chromosome_size; x++) {
			int index = y*chromosome_size + x;
			printf("%d", h_C[index]);
		}
		printf("\n");
	}
	
	cudaFree(d_A);
	cudaFree(d_C);
	free(h_A);
	free(h_C);
	
	return 0;
}

