#include <stdlib.h>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>


int main() {
	int r;
	int *h_ind;
    int *d_ind;
	int pop_size = 10;

	h_ind = (int *)malloc(pop_size*sizeof(int));
	for (int i = 0; i < pop_size; i++) {
		r = rand() % 2;
		h_ind[i] = r;
		// std::cout << i << ":" << r << std::endl;
	}
	std::cout << sizeof(int) * pop_size << std::endl;
	free(h_ind);
	cudaMalloc((int**)&d_ind, sizeof(int) * pop_size);
    cudaMemcpy(d_ind, h_ind, sizeof(int) * pop_size, cudaMemcpyHostToDevice);
	free(h_ind);
    cudaFree(d_ind);
	cudaDeviceReset();
}


