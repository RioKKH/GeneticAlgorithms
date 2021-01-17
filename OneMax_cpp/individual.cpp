#include <list>
#include <numeric>

#include "individual.hpp"

#define SIZE_OF_ARRAY(array) (sizeof(array)/sizeof(array[0]))


// constructor
individual::individual()
{
    for (int i = 0; i < N; i++) {
        chromosome[i] = rand() % 2;
    }
    fitness = 0;
}

// deconstructor
individual::~individual() {}


// 適応度を算出する
void individual::evaluate()
{
    // fitness = 0;
    fitness = std::accumulate(chromosome, chromosome + SIZE_OF_ARRAY(chromosome), 0);
}


/**
 * @brief    Sigle point crossover
 * @param    p1: Individual (parent1)
 * @param    p2: Individual (parent2)
 * @return   void
 */
void individual::apply_crossover_sp(individual *p1, individual *p2)
{
    int point, i;
    point = rand() % (N - 1);
    for (i = 0; i <= point; i++) {
        chromosome[i] = p1->chromosome[i];
    }
    for (; i < N; i++) {
        chromosome[i] = p2->chromosome[i];
    }
}


/**
 * @brief    Two-point crossover
 * @param    p1: Individual (parent1)
 * @param    p2: Individual (parent2)
 * @return   void
 */
void individual::apply_crossover_tp(individual *p1, individual *p2)
{
    int point1, point2, tmp, i;

    point1 = rand() & (N - 1);
    point2 = (point1 + (rand() % (N - 2) + 1));

    if (point1 > point2) {
        tmp = point1;
        point1 = point2;
        point2 = tmp;
    }
    for (i = 0; i <= point1; i++) {
        chromosome[i] = p1->chromosome[i];
    }
    for (; i <= point2; i++) {
        chromosome[i] = p2->chromosome[i];
    }
    for (; i < N; i++) {
        chromosome[i] = p1->chromosome[i];
    }
}


/**
 * @brief    Uniform crossover
 * @param    p1: Individual (parent1)
 * @param    p2: Individual (parent2)
 * @return   void
 */
void individual::apply_crossover_uniform(individual *p1, individual *p2)
{
    int i;

    for (i = 0; i < N; i++) {
        if (rand() % 2 == 1) {
            chromosome[i] = p1->chromosome[i];
        } else {
            chromosome[i] = p2->chromosome[i];
        }
    }
}


/**
 * @brief    Mutation
 * @param    Nothing
 * @return   void
 */
void individual::mutate()
{
    for (int i = 0; i < N; i++) {
        if (RAND_01 < MUTATE_PROB) {
            chromosome[i] = 1 - chromosome[i];
        }
    }
}
