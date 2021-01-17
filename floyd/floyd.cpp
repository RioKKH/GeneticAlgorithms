#include <stdio.h>
#include "population.hpp"

int main()
{
    int i;
    Population *population;

    srand((unsigned int)time(NULL));

    population = new Population;
    for(i=1; i<=GEN_MAX; i++) {
        population->alternate();
        printf("%d th generation : maximum fitness: %f\n",
                i, population->ind[0]->fitness);
    }
    population->printResult();
    delete population;

    return 0;
}
