#include <stdio.h>
#include "population.hpp"

// constructor
population::population()
{
    int i;
    ind = new individual* [POP_SIZE];
    next_ind = new individual* [POP_SIZE];
    for (i = 0; i < POP_SIZE; i++) {
        ind[i] = new individual();
        next_ind[i] = new individual();
    }
}

// destructor
population::~population()
{
    int i;
    for (i = 0; i < POP_SIZE; i++) {
        delete ind[i];
        delete next_ind[i];
    }
    delete[] ind;
    delete[] next_ind;
}


/**
 * @brief   evaluation of the fitness of each individuals then sort
 * individuals by fitness value.
 * @param   None
 * @return  void
 */
void population::evaluate()
{
    int i;

    for (i = 0; i < POP_SIZE; i++) {
        ind[i]->evaluate();
    }
    sort(0, POP_SIZE - 1);
}

/**
 * @brief Quick sort
 * @param lb: integer. Lower limit of the index of the target element of the sort.
 * @param ub: integer. Upper limit of the index of the target element of the sort.
 */
void population::sort(int lb, int ub)
{
    int i, j, k;
    double pivot;
    individual *tmp;

    if (lb < ub) {
        k = (lb + ub) / 2;
        pivot = ind[k]->fitness;
        i = lb;
        j = ub;
        do {
            while (ind[i]->fitness < pivot) {
                i++;
            }
            while (ind[j]->fitness > pivot) {
                j--;
            }
            if (i <= j) {
                tmp = ind[i];
                ind[i] = ind[j];
                ind[j] = tmp;
                i++;
                j--;
            }
        } while (i <= j);
        sort(lb, j);
        sort(i, ub);
    }
}


/**
 * @brief   Move generation forward
 * @param   None
 * @return  void
 */
void population::alternate()
{
    int i, j, p1, p2;
    individual **tmp;

    printf("initialize tr_fit\n");
    //* this is only for roulette selection
    /*
    denom = 0.0;
    for (i = 0; i < POP_SIZE; i++) {
        tr_fit[i] = (ind[POP_SIZE - 1]->fitness - ind[i]->fitness)
            / (ind[POP_SIZE - 1]->fitness - ind[0]->fitness);
        denom += tr_fit[i];
    }
    */
    // evaluate
    printf("evaluate\n");
    evaluate();

    printf("print fitness value\n");
    for (i = 0; i < POP_SIZE; i++) {
        printf("index %d: fitness: %d: ", i, ind[i]->fitness);
        for (j = 0; j < N; j++) {
            printf("%d", ind[i]->chromosome[j]);
        }
        printf("\n");
    }

    // Apply elitism and pick up elites for next generation
    printf("Elitism\n");
    for (i = 0; i < ELITE; i++) {
        for (j = 0; j < N; j++) {
            next_ind[i]->chromosome[j] = ind[i]->chromosome[j];
        }
    }

    // select parents and do the crossover
    printf("select parents and do the crossover\n");
    for (; i < POP_SIZE; i++) {
        p1 = select_by_tournament();
        p2 = select_by_tournament();
        printf("p1: ");
        for (int j = 0; j < N; j++) {
            printf("%d", ind[p1]->chromosome[j]);
        }
        printf("\n");
        printf("p2: ");
        for (int j = 0; j < N; j++) {
            printf("%d", ind[p2]->chromosome[j]);
        }
        printf("\n");
        next_ind[i]->apply_crossover_sp(ind[p1], ind[p2]);
        printf("nx: ");
        for (int j = 0; j < N; j++) {
            printf("%d", next_ind[i]->chromosome[j]);
        }
        printf("\n");
    }

    // Mutate candidate of next generation
    printf("Mutate candidate of next generation\n");
    for (i = 1; i < POP_SIZE; i++) {
        next_ind[i]->mutate();
    }

    // change next generation to current generation
    printf("change next generation to current generation\n");
    tmp = ind;
    ind = next_ind;
    next_ind = tmp;

    // evaluate
    printf("evaluate\n");
    evaluate();
}


/**
 * @brief   Select one individual as parent based on rank order of fitness value.
 * @param   None
 * @return  population size as integer
 */
/*
int population::select_by_ranking()
{
    int num, denom, r;

    // denom = POP_SIZE * (POP_SIZE + 1) / 2;
    // r = ((rand() << 16) + 
    do {
        r = rand();
*/

/**
 * @brief   Roulette selection
 * @param   None
 * @return  Integer as index of parent
 */
int population::select_by_roulette()
{
    int rank;
    double prob, r;

    r = RAND_01;
    for (rank = 1; rank < POP_SIZE; rank++) {
        prob = tr_fit[rank - 1] / denom;
        if (r <= prob) {
            break;
        }
        r -= prob;
    }
    return rank - 1;
}

/**
 * @brief   Tournament selection
 * @param   None
 * @return  Integer as index of parent
 */
int population::select_by_tournament()
{
    int i, ret, num, r;
    int best_fit;
    int tmp[POP_SIZE];
    // int tmp[N];

    printf("initialize tmp\n");
    for (i = 0; i < POP_SIZE; i++) {
    // for (i = 0; i < N; i++) {
        tmp[i] = 0;
    }
    ret = -1;
    // bset_fit = DBL_MAX;
    best_fit = 0; // in case of one-max prob., bigger fitness is better.
    num = 0;
    printf("enter while loop\n");
    while(1) {
        r = rand() % POP_SIZE; // ここはPOP_SIZEの剰余でないとおかしいと思う
        printf("r: %d, tmp[%d]: %d\n", r, r, tmp[r]);
        // r = rand() % N;
        if (tmp[r] == 0) { // 既に確認済みの個体については除外出来るようにしている
            tmp[r] = 1; 
            // printf("check if fitness is better than current best fitness\n");
            printf("num: %d/%d\n", num + 1, TOURNAMENT_SIZE);
            printf("current best fitness value %i , candidate fitness value %d\n",
                    best_fit, ind[r]->fitness);
            if (ind[r]->fitness > best_fit) {
                ret = r;
                best_fit = ind[r]->fitness;
            }
            if (++num == TOURNAMENT_SIZE) {
                break;
            }
        }
    }
    return ret;
}


/**
 * @brief   show results on stdout
 * @param   None
 * @return  void
 */
void population::print_result()
{
    // int i;
}

