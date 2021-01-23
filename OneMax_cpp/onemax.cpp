#include <iostream>
#include <stdio.h>
#include "population.hpp"

int main(int argc, char **argv)
{
    int gen_max = 0;

    Parameters *prms;
    prms = new Parameters();
    gen_max = prms->getGenMax();
    std::cout << gen_max << std::endl;

    srand((unsigned int)time(NULL));
    std::cout << "!!!Start!!!" << std::endl;;

    population *pop;
    pop = new population(prms);

    std::cout << "!!!After Population!!!" << std::endl;

    for (int i = 1; i <= gen_max; i++) {
        printf("Generation: %d\n", i);
        pop->alternate();
        printf("%d th generation : maximum fitness: %d\n",
                i, pop->ind[0]->fitness);
    }
    pop->print_result();

    // delete pointers
    delete pop;
    delete prms;
    std::cout << "!!!End!!!" << std::endl;

    return 0;
}
