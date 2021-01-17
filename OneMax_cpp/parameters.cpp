#include <iostream>
#include <fstream>
#include <string>

#include "parameters.hpp"

#define PARAMNAME "onemax_prms.dat"

Parameters::Parameters()
{
    std::cout << "constructor" << std::endl;
    params = new TEvolutionParameters;
    params->GEN_MAX = 100;
    params->POP_SIZE = 100;
    params->ELITE = 1;
    params->N = 10;
    params->TOURNAMENT_SIZE = 5;
    params->MUTATE_PROB = 0.01;
    std::cout << "End of constructor" << std::endl;
}

Parameters::~Parameters() {
    std::cout << "destructor" << std::endl;
    delete params;
    std::cout << "End of destructor" << std::endl;
}

void Parameters::loadParams()
{
    std::ifstream infile(PARAMNAME);
    std::string line;
    while (getline(infile, line)) {
        std::cout << line << std::endl;
    }
    infile.close();
}

int Parameters::getGenMax()
{
    // std::cout << "GEN_MAX: " << params->GEN_MAX << std::endl;
    return params->GEN_MAX;
}

int Parameters::getPopSize()
{
    return params->POP_SIZE;
}

int Parameters::getElite()
{
    return params->ELITE;
}

int Parameters::getLengthOfChromosome()
{
    return params->N;
}

int Parameters::getTournamentSize()
{
    return params->TOURNAMENT_SIZE;
}

float Parameters::getMutateProbability()
{
    return params->MUTATE_PROB;
}
