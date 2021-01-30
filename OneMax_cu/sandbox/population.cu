#include <cuda_runtime.h>
#include <numeric>
// #include <helper_function.h>
#include <iostream>
#include <stdio.h>
// #include <plog/Log.h> // manybe later

#include "population.hpp"
// #include "parameters.hpp"

__device__ void calc_fitness(int, int, int*d, int*);

int divRoundUp(int value, int radix) {
	return (value + radix - 1) / radix;
}

__global__ void gpu_evolve(int pop_num, int chromosome_num,
		                 int* d_ind, int* d_next_ind, int* d_fitness)
{
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = chromosome_num * y + x;
	// printf("%d,%d,%d,%d\n", y,x,idx,d_ind[idx]);
	// d_next_ind[idx] = d_ind[idx];
	if (x < chromosome_num && y < pop_num) {
		d_next_ind[idx] = d_ind[idx];
	}
	calc_fitness(pop_num, chromosome_num, d_ind, d_fitness);
}

__device__ void calc_fitness(int pop_num, int chromosome_num, int* d_ind, int* d_fitness)
{
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	// unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < chromosome_num; i++) {
		d_fitness[y] += d_ind[y + i];
	}
}


// constructor
population::population()
{
    // load parameters
    pop_num = 20;
	chromosome_num = 10;
	int ind_size = sizeof(int) * chromosome_num;
	int pop_size = pop_num * ind_size;

	int *h_ind, *h_next_ind, *h_fitness;
	int *d_ind, *d_next_ind, *d_fitness;

    // initialize of individuals = population
	std::cout << "current generation" << std::endl;
	h_fitness = (int *)malloc(ind_size);
	h_ind = (int *)malloc(pop_size);
	h_next_ind = (int *)malloc(pop_size);

	std::cout << "initialization" << std::endl;
	int k = 0;
    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			k = i * chromosome_num + j;
			h_ind[k] = rand() % 2;
			h_next_ind[i * chromosome_num + j] = 0;
		}
		h_fitness[i] = 0;
    }

    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			printf("%d", h_ind[i * chromosome_num + j]);
		}
		printf("\n");
    }

	cudaMalloc((void **)&d_ind, pop_size);
	cudaMalloc((void **)&d_next_ind, pop_size);
	cudaMalloc((void **)&d_fitness, ind_size);
	cudaMemcpy(d_ind, h_ind, pop_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_next_ind, h_next_ind, pop_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fitness, h_fitness, ind_size, cudaMemcpyHostToDevice);

    // evaluate();
	dim3 blockDim(1, 1);
	// dim3 gridDim(10, 20);
	dim3 gridDim(divRoundUp(chromosome_num, blockDim.x), divRoundUp(pop_num, blockDim.y));

	std::cout << "block_num:" << blockDim.x << ":" << blockDim.y << std::endl;
	std::cout << "grid_num:" << gridDim.x << ":" << gridDim.y << std::endl;

	gpu_evolve<<<gridDim, blockDim>>>(
			pop_num, 
			chromosome_num,
			d_ind,
			d_next_ind, 
			d_fitness);
	// gpuKernel1<<<gridDim, blockDim>>>(pop_size, chromosome_size, d_ind);
	cudaDeviceSynchronize();
	// cudaMemcpy(d_next_ind, h_next_ind, nBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_next_ind, d_next_ind, pop_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fitness, d_fitness, ind_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			printf("%d", h_next_ind[i * chromosome_num + j]);
		}
		printf(":%d\n", h_fitness[i]);
    }
}

// destructor
population::~population()
{
    // int i;
    // for (i = 0; i < pop_size; i++) {
	// 	// std::cout << i << "destructor" << std::endl;
    //     delete h_ind[i];
    //     delete h_next_ind[i];
    // }
	// std::cout << "last delete" << std::endl;
    delete[] h_ind;
    delete[] h_next_ind;
	cudaFree(d_ind);
	cudaFree(d_next_ind);
	cudaDeviceReset();
    // delete[] tr_fit;
}

/**
 * @brief   evaluation of the fitness of each individuals then sort
 * individuals by fitness value.
 * @param   None
 * @return  void
 */
// void population::evaluate()
// {
//     for (int i = 0; i < pop_size; i++) {
//         ind[i]->evaluate();
//     }
//     sort(0, pop_size - 1);
// }
// 
// /**
//  * @brief Quick sort
//  * @param lb: integer. Lower limit of the index of the target element of the sort.
//  * @param ub: integer. Upper limit of the index of the target element of the sort.
//  */
// void population::sort(int lb, int ub)
// {
//     int i, j, k;
//     double pivot;
//     individual *tmp;
// 
//     if (lb < ub) {
//         k = (lb + ub) / 2;
//         pivot = ind[k]->fitness;
//         i = lb;
//         j = ub;
//         do {
//             while (ind[i]->fitness < pivot) {
//                 i++;
//             }
//             while (ind[j]->fitness > pivot) {
//                 j--;
//             }
//             if (i <= j) {
//                 tmp = ind[i];
//                 ind[i] = ind[j];
//                 ind[j] = tmp;
//                 i++;
//                 j--;
//             }
//         } while (i <= j);
//         sort(lb, j);
//         sort(i, ub);
//     }
// }
// 
// 
// /**
//  * @brief   Move generation forward
//  * @param   None
//  * @return  void
//  */
// void population::alternate()
// {
//     static int generation = 0;
//     int i, j, p1, p2;
//     individual **tmp;
// 
//     // printf("initialize tr_fit\n");
//     //* this is only for roulette selection
//     /*
//     denom = 0.0;
//     for (i = 0; i < POP_SIZE; i++) {
//         tr_fit[i] = (ind[POP_SIZE - 1]->fitness - ind[i]->fitness)
//             / (ind[POP_SIZE - 1]->fitness - ind[0]->fitness);
//         denom += tr_fit[i];
//     }
//     */
//     // evaluate
//     // printf("evaluate\n");
//     // evaluate();
// 
//     /*
//     printf("print fitness value\n");
//     for (i = 0; i < pop_size; i++) {
//         printf("index %d: fitness: %d: ", i, ind[i]->fitness);
//         for (j = 0; j < N; j++) {
//             printf("%d", ind[i]->chromosome[j]);
//         }
//         printf("\n");
//     }
//     */
// 
//     // Apply elitism and pick up elites for next generation
//     // printf("Elitism\n");
//     for (i = 0; i < elite; i++) {
//         for (j = 0; j < N; j++) {
//         // for (j = 0; j < N; j++) {
//             next_ind[i]->chromosome[N - j] = ind[i]->chromosome[N - j];
//         }
//     }
// 
//     //- select parents and do the crossover
//     for (; i < pop_size; i++) {
//         p1 = select_by_tournament();
//         p2 = select_by_tournament();
//         next_ind[i]->apply_crossover_tp(ind[p1], ind[p2]);
//         // next_ind[i]->apply_crossover_sp(ind[p1], ind[p2]);
// 
//         // Debug Info
//         /*
//         printf("p1: ");
//         for (int j = 0; j < N; j++) {
//             printf("%d", ind[p1]->chromosome[j]);
//         }
//         printf("\n");
//         printf("p2: ");
//         for (int j = 0; j < N; j++) {
//             printf("%d", ind[p2]->chromosome[j]);
//         }
//         printf("\n");
//         printf("nx: %d ", i);
//         for (int j = 0; j < N; j++) {
//             printf("%d", next_ind[i]->chromosome[j]);
//         }
//         printf("\n");
//         */
//     }
// 
//     //- Mutate candidate of next generation
//     for (i = 1; i < pop_size; i++) {
//         next_ind[i]->mutate();
//     }
// 
//     //- change next generation to current generation
//     tmp = ind;
//     ind = next_ind;
//     next_ind = tmp;
// 
//     //- evaluate
//     evaluate();
//     generation++;
// 
//     /*
//     //- Show the result of this generation
//     int sum = 0;
//     float mean = 0;
//     float var = 0;
//     float stdev = 0;
// 
//     for (int i = 0; i < pop_size; i++) {
//         sum += ind[i]->fitness;
//     }
//     mean = (float)sum / pop_size;
//     for (int i = 0; i < pop_size; i++) {
//         var += ((float)ind[i]->fitness - mean) * ((float)ind[i]->fitness - mean);
//     }
//     stdev = sqrt(var / (pop_size - 1));
// 
//     // generation, max, min, mean, stdev
//     printf("%d,%d,%d,%f,%f\n", generation, ind[N-1]->fitness, ind[0]->fitness, mean, stdev); 
//     */
// }
// 
// 
// /**
//  * @brief   Select one individual as parent based on rank order of fitness value.
//  * @param   None
//  * @return  population size as integer
//  */
// /*
// int population::select_by_ranking()
// {
//     int num, denom, r;
// 
//     // denom = POP_SIZE * (POP_SIZE + 1) / 2;
//     // r = ((rand() << 16) + 
//     do {
//         r = rand();
// */
// 
// /**
//  * @brief   Roulette selection
//  * @param   None
//  * @return  Integer as index of parent
//  */
// int population::select_by_roulette()
// {
//     int rank;
//     double prob, r;
// 
//     r = RAND_01;
//     for (rank = 1; rank < pop_size; rank++) {
//         prob = tr_fit[rank - 1] / denom;
//         if (r <= prob) {
//             break;
//         }
//         r -= prob;
//     }
//     return rank - 1;
// }
// 
// /**
//  * @brief   Tournament selection
//  * @param   None
//  * @return  Integer as index of parent
//  */
// int population::select_by_tournament()
// {
//     int i, ret, num, r;
//     int best_fit;
//     int *tmp;
//     tmp = new int[pop_size];
// 
//     // printf("initialize tmp\n");
//     for (i = 0; i < pop_size; i++) {
//         tmp[i] = 0;
//     }
// 
//     ret = -1;
//     best_fit = 0; // in case of one-max prob., bigger fitness is better.
//     num = 0;
//     // printf("enter while loop\n");
//     while(1) {
//         r = rand() % pop_size; // ここはPOP_SIZEの剰余でないとおかしいと思う
//         // printf("r: %d, tmp[%d]: %d\n", r, r, tmp[r]);
//         // r = rand() % N;
//         if (tmp[r] == 0) { // 既に確認済みの個体については除外出来るようにしている
//             tmp[r] = 1; 
//             // debug print
//             // printf("check if fitness is better than current best fitness\n");
//             // printf("num: %d/%d\n", num + 1, tournament_size);
//             // printf("current best fitness value %i , candidate fitness value %d\n",
//             //         best_fit, ind[r]->fitness);
//             if (ind[r]->fitness > best_fit) {
//                 ret = r;
//                 best_fit = ind[r]->fitness;
//             }
//             if (++num == tournament_size) {
//                 break;
//             }
//         }
//     }
//     delete[] tmp;
//     return ret;
// }
// 
// 
// /**
//  * @brief   show results on stdout
//  * @param   None
//  * @return  void
//  */
// void population::print_result()
// {
//     // int i;
// }
