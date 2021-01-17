#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deap import base
from deap import creator
from deap import tools

import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# problem constants:
ONE_MAX_LENGTH = 100 # Length of bit string to be optimized.

# Genetic Algorithm constants:
POPULATION_SIZE = 200 # number of individuals in population
P_CROSSOVER = 0.9     # probability for crossover
P_MUTATION = 0.1      # probability for mutating an individual
MAX_GENERATIONS = 1  # max number of generations for stopping condition
#MAX_GENERATIONS = 50  # max number of generations for stopping condition

# set the random seed:
# We set the random function seed to a constant number of some value
# to make our experiment repeatable.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
# random.randint accepts two arguments (a and b) and returns one
# random interger N such that a <= N <= b.
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the individual class based on list
creator.create("Individual", list, fitness=creator.FitnessMax)
#creator.create("Individual", array.array, typecode='b', 
#               fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
# How to use initRepeat(): ex. tools.initRepeat(list, random.random, 30)
# 1. The container type in which we would like to put the resulting objects
# 2. The function used to generate objects that will be put into the container
# 3. The number of objects we want to generate
toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                 toolbox.zeroOrOne, ONE_MAX_LENGTH)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list,
                 toolbox.individualCreator)


# fitness calculation:
# compute the number of '1's in the individual
def oneMaxFitness(individual):
    return sum(individual), # return a tuple

toolbox.register("evaluate", oneMaxFitness)

# genetic operators:

# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=3)

# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # calculate fitness tuple for each individual in the population:
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    # extract fitness values from all individuals in population:
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # initialize statistics accumulators:
    maxFitnessValues = []
    meanFitnessValues = []

    # main evolutionary loop:
    # stop if max fitness value reached the known max value
    # OR if number of generations exceeded the preset value:
    while max(fitnessValues) < ONE_MAX_LENGTH + 20 and generationCounter < MAX_GENERATIONS:
    #while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        # update counter:
        generationCounter = generationCounter + 1
        # apply the selection operator, to select the next generation's
        # individuals:
        print('population:', len(population), np.shape(population))
        offspring = toolbox.select(population, len(population))
        # clone the selected individuals:
        # shape of offspring is (200, 100) because population size is 200 and
        # ONEONE_MAX_LENGTH is 100.
        print(type(offspring), np.shape(offspring))
        offspring = list(map(toolbox.clone, offspring))
        print(type(offspring), np.shape(offspring))
        #print(offspring)


        # apply the crossover operator to pairs of offspring:

        # slice of list
        # [start:stop:step]
        # [::2] means to pick up even-numbered element.
        # [1::2] means to pick up odd-numbered element.
        print('offspring:', np.shape(offspring))
        print('offspring[::2]:', np.shape(offspring[::2]))
        print('offfspring[1::2]', np.shape(offspring[1::2]))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            #print(child1, child2)
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for i, mutant in enumerate(offspring):
            #print(i, mutant)
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # calculate fitness for the individuals with no previous calculated
        # fitness value
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals,
                                            freshFitnessValues):
            individual.fitness.values = fitnessValue

        # replace the current population with the offspring:
        population[:] = offspring

        # collect fitnessValues into a list, update statistics and print:
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print(f"- Genration {generationCounter}: Max Fitness = {maxFitness}, Avg Fitness = {meanFitness}")

        # find and print best individual:
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")

    # Genetic Algorithm is done - plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()

