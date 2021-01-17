#include <stdio.h>
#include "population.hpp"
#include "parameters.hpp"

int main(int argc, char **argv)
{
    int i;
    population *pop;
    Parameters *prms;
    prms = new Parameters(*prms);
    // prms = new Parameters();
    prms->loadParams();

    srand((unsigned int)time(NULL));
    printf("!!!Start!!!\n");

    pop = new population;
    printf("!!!After Population!!!\n");

    for (i = 1; i <= GEN_MAX; i++) {
        printf("Generation: %d\n", i);
        pop->alternate();
        printf("%d th generation : maximum fitness: %d\n",
                i, pop->ind[0]->fitness);
    }
    pop->print_result();

    // delete pointers
    delete pop;
    delete prms;
    printf("!!!End!!!\n");

    return 0;
}
