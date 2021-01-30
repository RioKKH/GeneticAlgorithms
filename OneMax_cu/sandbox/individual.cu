#include <list>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>

#include "individual.hpp"

#include <cuda_runtime.h>
#include <helper_cuda.h>


individual::individual()
{
	N = 10;
	mutate_prob = 0.1;
	chromosome  =new int[N];
	for (int i = 0; i < N; i++) {
		chromosome[i] = rand() % 2;
	}
	fitness = 0;
}

individual::~individual()
{
	delete[] chromosome;
}
