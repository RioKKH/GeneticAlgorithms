#include <iostream>
#include <stdio.h>
#include <numeric>
// #include <helper_function.h>
#include <cuda_runtime.h>
// #include <plog/Log.h> // manybe later

#include "population.hpp"
// #include "parameters.hpp"


#define CHECK(call)                                                           \
{                                                                             \
	const cudaError_t error = call;                                           \
	if (error != cudaSuccess)                                                 \
	{                                                                         \
		printf("Error: %s:%d, ", __FILE__, __LINE__);                         \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));    \
		exit(1);                                                              \
	}                                                                         \
}                                                                             \

__global__ void gpu_evolve(int, int, int*, int*, int*, int*, int*);
__device__ void calc_fitness(int, int, int*d, int*, int*);
__device__ void evaluate(int, int, int*d, int*, int*);
__device__ void get_fitness(int, int* ,int*);


int divRoundUp(int value, int radix) {
	return (value + radix - 1) / radix;
}


__global__ void gpu_evolve(int pop_num, int chromosome_num,
		                 int* d_ind,
						 int* d_next_ind, int* d_temp_ind, int* d_fitness,
						 int* d_diag)
{
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = chromosome_num * y + x;
	if (x < chromosome_num && y < pop_num) {
		d_next_ind[idx] = d_ind[idx];
	}
	calc_fitness(pop_num, chromosome_num, d_ind, d_temp_ind, d_fitness);
	get_fitness(chromosome_num, d_next_ind, d_fitness);
	// calc_fitness(pop_num, chromosome_num, d_ind, d_fitness);
	// calc_fitness2(pop_num, chromosome_num, d_ind, d_temp_ind, d_fitness);
}


/*
__device__ void calc_fitness(int pop_num, int chromosome_num, int* d_ind, int* d_fitness)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int idx = chromosome_num * y + x;
	int tmp = 0;
	for (int i = 0; i < chromosome_num; i++) {
		tmp += d_ind[chromosome_num * x + i];
	}
	d_fitness[k] = tmp;
}
*/


__device__ void calc_fitness(int pop_num, int chromosome_num,
		                 int* d_ind, int* d_temp_ind, int* d_fitness)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	int tmp = 0;
	for (int i = 0; i < chromosome_num; i++) {
		tmp += d_ind[chromosome_num * y + i] * d_temp_ind[chromosome_num * i + x];
	}
	d_fitness[chromosome_num * y + x] = tmp;
}

__device__ void get_fitness(int chromosome_num, int* d_next_ind, int* d_fitness)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = chromosome_num * y + x;
	if (x == y) {
		d_next_ind[idx] = d_fitness[idx];
	} else {
		d_next_ind[idx] = 0;
	}
}



// constructor
population::population()
{
    // load parameters
    pop_num = 20;
	chromosome_num = 20;
	int ind_size = sizeof(int) * chromosome_num;
	int pop_size = pop_num * ind_size;

    // initialize of individuals = population
	// std::cout << "Generation 0" << std::endl;
	h_ind      = (int *)malloc(sizeof(int)*pop_num*chromosome_num);
	h_next_ind = (int *)malloc(sizeof(int)*pop_num*chromosome_num);
	h_temp_ind = (int *)malloc(sizeof(int)*pop_num*chromosome_num);
	h_fitness  = (int *)malloc(sizeof(int)*pop_num*chromosome_num);
	h_diag     = (int *)malloc(sizeof(int)*pop_num*chromosome_num);
	// h_fitness  = (int *)malloc(sizeof(int)*pop_num);

	std::cout << "Initialize generation 0" << std::endl;
	int k = 0;
    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			k = i * chromosome_num + j;
			h_ind[k] = rand() % 2;
			h_next_ind[k] = 0;
			h_temp_ind[k] = 1;
			h_fitness[i] = 0;
			if (i == j) {
				h_diag[k] = 1;
			} else {
				h_diag[k] = 0;
			}
			printf("%d,%d,%d,%d,%d\n", i, j, k, h_ind[k], h_next_ind[k]);
		}
    }
	// memset(h_ind, rand() % 2, sizeof(int)*pop_num*chromosome_num);
	// memset(h_next_ind, 0, sizeof(int)*pop_num*chromosome_num);
	// memset(h_temp_ind, 1, sizeof(int)*pop_num*chromosome_num);
	// memset(h_fitness, 0, sizeof(int)*pop_num);

	/*
	k = 0;
    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			k = i * chromosome_num + j;
			// h_fitness[i] += h_ind[k];
			// printf("Pop:%d, Chrom:%d, TempFitness:%d, h_ind:%d\n",
			// 		i, j, h_fitness[i], h_ind[k]);
		}
	}
	*/

	k = 0;
    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			k = i * chromosome_num + j;
			printf("%d", h_ind[k]);
		}
		// printf(":%d\n", h_fitness[i]);
    }

	std::cout << "cudaMalloc" << std::endl;
	CHECK(cudaMalloc((int**)&d_ind,      pop_size));
	CHECK(cudaMalloc((int**)&d_next_ind, pop_size));
	CHECK(cudaMalloc((int**)&d_temp_ind, pop_size));
	CHECK(cudaMalloc((int**)&d_fitness,  pop_size));
	CHECK(cudaMalloc((int**)&d_diag,     pop_size));

	std::cout << "cudaMemcpy" << std::endl;
	CHECK(cudaMemcpy(d_ind,      h_ind,      pop_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_next_ind, h_next_ind, pop_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_temp_ind, h_temp_ind, pop_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_fitness,  h_fitness,  pop_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_diag,     h_diag,     pop_size, cudaMemcpyHostToDevice));

	CHECK(cudaGetLastError());

    // evaluate();
	dim3 blockDim(20, 5);
	dim3 gridDim(divRoundUp(chromosome_num, blockDim.x), divRoundUp(pop_num, blockDim.y));

	std::cout << "block_num:" << blockDim.x << ":" << blockDim.y << std::endl;
	std::cout << "grid_num:" << gridDim.x << ":" << gridDim.y << std::endl;

	gpu_evolve<<<gridDim, blockDim>>>(pop_num, chromosome_num,
			                          d_ind, d_next_ind, d_temp_ind,
									  d_fitness, d_diag);
	cudaDeviceSynchronize();
	cudaMemcpy(h_next_ind, d_next_ind, pop_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fitness,  d_fitness,  pop_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			printf("%d", h_ind[i * chromosome_num + j]);
			// printf("%d", h_next_ind[i * chromosome_num + j]);
		}
		printf("\n");
		// printf(":%d\n", h_fitness[i]);
    }
	std::cout << "fitness matrix" << std::endl;
    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			printf("%d,", h_fitness[i * chromosome_num + j]);
			// printf("%d", h_next_ind[i * chromosome_num + j]);
		}
		printf("\n");
		// printf(":%d\n", h_fitness[i]);
    }
	std::cout << "next generation matrix" << std::endl;
    for (int i = 0; i < pop_num; i++) {
		for (int j = 0; j < chromosome_num; j++) {
			printf("%d,", h_next_ind[i * chromosome_num + j]);
			// printf("%d", h_next_ind[i * chromosome_num + j]);
		}
		printf("\n");
		// printf(":%d\n", h_fitness[i]);
    }
}

// destructor
population::~population()
{
    free(h_ind);
    free(h_next_ind);
	free(h_temp_ind);
	free(h_fitness);
	free(h_diag);
	cudaFree(d_ind);
	cudaFree(d_next_ind);
	cudaFree(d_temp_ind);
	cudaFree(d_fitness);
	cudaFree(d_diag);
	cudaDeviceReset();
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
