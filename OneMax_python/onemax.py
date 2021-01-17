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
#import japanize_matplotlib
#import seaborn as sns


class OneMax(object):

    def __init__(self):
        # problem constants:
        self.ONE_MAX_LENGTH = 100 # length of bit string to be optimized

        # Genetic Algorithm constants:
        self.POPULATION_SIZE = 200
        self.P_CROSSOVER = 0.9 # probability for crossover
        self.P_MUTATION = 0.1  # probability for mutating an individual
        self.MAX_GENERATIONS = 50
        self.HALL_OF_FAME_SIZE = 10
        self.TOURNSIZE = 3
        self.INDPB_NUMERATOR = 1.0

        # set the random seed:
        self.RANDOM_SEED = 42
        random.seed(self.RANDOM_SEED)

        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.oneMaxFitness)
        self.toolbox.register("select", tools.selTournament, self.TOURNSIZE)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutFlipBit,
                              indpb=self.INDPB_NUMERATOR/self.ONE_MAX_LENGTH)

        # create an operator that randomly returns 0 or 1:
        self.toolbox.register("zeroOrOne", random.randint, 0, 1)

        # define a single objective, maximizing fitness strategy:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # create the Individual class based on list:
        creator.create("Individual",
                       list, 
                       fitness=creator.FitnessMax)

        # create the individual operator to fill up an Individual instance:
        self.toolbox.register("individualCreator", 
                              tools.initRepeat, 
                              creator.Individual,
                              self.toolbox.zeroOrOne,
                              self.ONE_MAX_LENGTH)

        # create the population operator to generate a list of individuals:
        self.toolbox.register("populationCreator", 
                              tools.initRepeat,
                              list,
                              self.toolbox.individualCreator)

        self.name = 'Test'
        self.annotation = 'Test'


    def oneMaxFitness(self, individual):
        # fitness calculation:
        # comupte the number of '1's in the individual
        return sum(individual), # return a tuple


    def set_param(self,
                  population_size=200,
                  p_crossover=0.9,
                  p_mutation=0.1,
                  tournament_size=3,
                  max_generations=50,
                  hof=10,
                  indpb_numerator=1.0,
                  selection=tools.selTournament,
                  crossover=tools.cxOnePoint,
                  mutation=tools.mutFlipBit):

        self.POPULATION_SIZE = population_size
        self.P_CROSSOVER = p_crossover
        self.P_MUTATION = p_mutation
        self.TOURNSIZE = tournament_size
        self.MAX_GENERATIONS = max_generations
        self.HALL_OF_FAME_SIZE = hof
        self.INDPB_NUMERATOR = indpb_numerator

        self.toolbox.register("evaluate", self.oneMaxFitness)

        # genetic operators:
        # Tournament selection with tournament size of 3:
        self.toolbox.register("select", selection[1], tournsize=self.TOURNSIZE)

        # Single-point crossover:
        self.toolbox.register("mate", crossover[1])

        # Flip-bit mutation:
        # indpb: Independent probability for each attribute to be flipped
        self.toolbox.register("mutate", mutation[1],
                              indpb=self.INDPB_NUMERATOR/self.ONE_MAX_LENGTH)

        self.name = "_".join([str(self.POPULATION_SIZE),
                               str(self.P_CROSSOVER),
                               str(self.P_MUTATION),
                               str(self.TOURNSIZE),
                               str(self.MAX_GENERATIONS),
                               str(self.HALL_OF_FAME_SIZE),
                               str(self.INDPB_NUMERATOR),
                               str(selection[0]),
                               str(crossover[0]),
                               str(mutation[0])])

        s = "Population size: %s\n" % str(self.POPULATION_SIZE) 
        s += "Selection Operator: %s\n" % str(selection[0])
        s += "Crossover Operator: %s\n" % str(crossover[0])
        s += "Probability of Crossover: %s\n" % str(self.P_CROSSOVER)
        s += "Mutation Operator: %s\n" % str(mutation[0])
        s += "Probability of Mutation: %s\n" % str(self.P_MUTATION)
        s += "Probability of Flip bit mutation: %0.2f\n" %\
                (self.INDPB_NUMERATOR / self.ONE_MAX_LENGTH)
        s += "Tournament size: %d\n" % self.TOURNSIZE
        self.annotation = s


    def save_result(self, logbook, fname):
        df = pd.DataFrame(logbook)
        df.set_index('gen', drop=True, inplace=True)
        df.to_csv(fname + ".csv")
    

    # Gentic Algorithm flow:
    def run(self):

        # create initial population (generation 0):
        population = self.toolbox.populationCreator(n=self.POPULATION_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)

        # perform the Genetic Algorithm flow:
        population, logbook =\
            algorithms.eaSimple(population,
                                self.toolbox,
                                cxpb=self.P_CROSSOVER,
                                mutpb=self.P_MUTATION,
                                ngen=self.MAX_GENERATIONS,
                                stats=stats,
                                halloffame=hof,
                                verbose=True)

        # print Hall of Fame info:
        print("Hall of Fame Individuals = ", *hof.items, sep="\n")
        print("Best Ever Individual = ", hof.items[0])

        self.save_result(logbook, self.name)

        # Genetic Algorithm is done - extract statistics:
        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

        # plot statistics:
        plt.style.use('ggplot')
        plt.plot(maxFitnessValues, label='Maximum Fitness')
        plt.plot(meanFitnessValues, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Max / Average Fitness')
        plt.ylim(40, 110)
        plt.title('Max and Average Fitness over Generations')
        plt.text(15, 40, self.annotation, fontsize=12)
        plt.legend(loc='best', framealpha=0.5)
        plt.savefig(self.name + ".png")
        #plt.show()
        plt.clf()


def experiment():
    sel = {
        0: ["Tounament", tools.selTournament],
        1: ["Roulette", tools.selRoulette],
    }

    xover = {
        0: ["SinglePointCrossover", tools.cxOnePoint],
        1: ["TwoPointCrossover", tools.cxTwoPoint],
    }

    mut= {
        0: ["FlipBitMutation", tools.mutFlipBit]
    }

    om = OneMax()
    # experiments for checking the impact of population size
    #om.set_param( 50, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(100, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(400, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(800, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()

    # experiments for checking the impact of crossover operator
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0])
    #om.run()
    #om.set_param(200, 0.9, 0.3, 3, 50, 10, 1.0, sel[0], xover[0], mut[0])
    #om.run()
    #om.set_param(200, 0.9, 0.6, 3, 50, 10, 1.0, sel[0], xover[0], mut[0])
    #om.run()
    #om.set_param(200, 0.9, 0.9, 3, 50, 10, 1.0, sel[0], xover[0], mut[0])
    #om.run()


    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 5.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 10.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 20.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 50.0, sel[0], xover[0], mut[0]) 
    #om.run()

    #om.set_param(200, 0.9, 0.1, 2, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 3, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 5, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 10, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 100, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()
    #om.set_param(200, 0.9, 0.1, 200, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    #om.run()

    om.set_param(200, 0.9, 0.3, 200, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    om.run()
    om.set_param(200, 0.9, 0.6, 200, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    om.run()
    om.set_param(200, 0.9, 0.9, 200, 50, 10, 1.0, sel[0], xover[0], mut[0]) 
    om.run()

if __name__ == '__main__':
    onemax = OneMax()
    onemax.run()


