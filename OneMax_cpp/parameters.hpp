// #ifndef PARAMETERS_H
// #define PARAMETERS_H

#include <string>

typedef struct {
    int GEN_MAX;            // Total number of generations to evolve
    int POP_SIZE;           // Population size
    int ELITE;              // 
    int N;
    int TOURNAMENT_SIZE;
    float MUTATE_PROB;

} TEvolutionParameters;

class Parameters {
private:
    TEvolutionParameters* params;

public:
    Parameters();
    ~Parameters();
    void loadParams();
    int getGenMax();
    int getPopSize();
    int getElite();
    int getLengthOfChromosome();
    int getTournamentSize();
    float getMutateProbability();
};

// #endif /* PARAMETERS_H */
