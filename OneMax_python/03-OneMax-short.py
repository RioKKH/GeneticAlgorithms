#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# problem constants:
ONE_MAX_LENGTH = 100 # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.1  # probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 10

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                 toolbox.zeroOrOne, ONE_MAX_LENGTH)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list,
                 toolbox.individualCreator)


# fitness calculation:
# comupte the number of '1's in the individual
def oneMaxFitness(individual):
    return sum(individual), # return a tuple


parma_dict = {
    "selection":oneMaxFitness,
    "crossover":tools.selTournament,
    "mutation":tools.cxOnePoint,
    "population_size":400,
    "p_crossover":
    "p_mutation":


toolbox.register("evaluate", oneMaxFitness)

# genetic operators:mutFlipBit

# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=3)

# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

def save_result(logbook, fname):
    df = pd.DataFrame(logbook)
    df.set_index('gen', drop=True, inplace=True)
    df.to_csv(fname + ".csv")
    

# Gentic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow:
    population, logbook = algorithms.eaSimple(population,
                                              toolbox,
                                              cxpb=P_CROSSOVER,
                                              mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=True)

    # print Hall of Fame info:
    print("Hall of Fame Individuals = ", *hof.items, sep="\n")
    print("Best Ever Individual = ", hof.items[0])

    save_result(logbook, "test")

    # Genetic Algorithm is done - extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()


