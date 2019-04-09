import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats
from time import time
import datetime
from datetime import timedelta



class GeneticAlgorithm(object):

    def __init__(self, chromosome, parent_selector, generations=70, num_population=20, crossover_prob=0.5,
                 mutation_prob=0.7, maximize_fitness=True, statistical_validation=True, save_pop=False, training_hours=1e3):
        '''
        Class to generate a basic Genetic Algorithms.

        :param chromosome: object of class Chromosome
        :param parent_selector: object of class ParentSelector
        :param generations: Number of generations to evolve the population
        :param num_population: Number of individuals of the population
        :param crossover_prob: Probability to do a crossover between parents
        :param mutation_prob: Probability to make a mutation of a chromosome
        :param maximize_fitness: If the fitness has to be maximized (True) or minimized (false)
        '''
        self.num_generations = generations
        self.pop_size = num_population
        self.prob_muta = mutation_prob
        self.cross_prob = crossover_prob
        self.chromosome = chromosome
        self.parent_selector = parent_selector
        self.statistical_validation = statistical_validation
        self.maximize = maximize_fitness
        self.history = np.empty((self.pop_size, self.num_generations + 1))
        self.history_fitness = {}
        self.population_history = []
        self.best_fit_history = {}
        self.save_pop = save_pop
        self.parent_selector.set_genetic_algorithm(self)
        self.training_hours = training_hours
        print("Genetic algorithm params:")
        print("Number of generations: %d" % self.num_generations)
        print("Population size: %d" % self.pop_size)

    def create_random_indiv(self):
        return self.chromosome.random_indiv()

    def create_simple_indiv(self):
        return self.chromosome.simple_indiv()

    def initial_population(self):
        population = []
        for i in range(self.pop_size):
            population.append(self.create_random_indiv())
        return population

    @staticmethod
    def mutation(offspring):
        for crom in offspring:
            crom.mutate()
        return offspring

    def actualize_history(self, generation, rank):
        for i in range(len(rank)):
            self.history[i, generation] = rank[i][1]

    def rank(self, population):
        fitness_result = {}
        for i in range(self.pop_size):
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                self.history_fitness[gen] = population[i].fitness()
            elif population[i].fit is None:
                population[i].fit = self.history_fitness[gen]
            fitness_result[i] = self.history_fitness[gen]
        return sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=self.maximize)

    def show_history(self):
        x = np.linspace(0, self.num_generations, self.num_generations + 1)
        mean = np.mean(self.history, axis=0)
        max_ = np.max(self.history, axis=0)
        min_ = np.min(self.history, axis=0)
        plt.plot(x, mean, label="mean", color='r', lw=1)
        plt.plot(x, max_, label='max', color='b', lw=1)
        plt.plot(x, min_, label='min', color='g', lw=1)
        plt.legend()
        plt.xlabel("num generation")
        plt.ylabel('Fitness')
        plt.show()


class GenerationalGA(GeneticAlgorithm):
    def __init__(self, num_parents, **kwargs):

        if type(num_parents) == int:
            self.num_parents = num_parents
        else:
            self.num_parents = int(kwargs['num_population'] * num_parents)
        super().__init__(**kwargs)
        self.offspring_size = self.pop_size - self.num_parents
        print("num parents: %d" % self.num_parents)
        print("offspring size: %d" % self.offspring_size)

    def replace(self, population, next_generation, all_parents):
        fitness_result = {}
        for i in range(len(population)):
      
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                self.history_fitness[gen] = population[i].fitness()
            elif population[i].fit is None:
                population[i].fit = self.history_fitness[gen]

            fitness_result[i] = self.history_fitness[gen]
        fitness_result = dict(sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=self.maximize))
        idxs = list(fitness_result.keys())[:-len(next_generation)]
        return [population[i] for i in idxs] + next_generation

    def validate_best(self, ranking, population, iters=5):
        '''
        If the evaluation process of each gen is not deterministic, it's necessary
        to do a statical validation of the performance. To do this, this function
        makes a validation of the best gen of the generation in the population, evaluating it
        a 'iters' number of times, and add a list with the results to a dictionary.

        :param ranking:     The ranking of the population in the actual generation as a dict with
                            the name of each gen as a key, and a tuple (id, fitnees) as a value
        :param population:  Al the genes in the actual generation
        :param iters:       The number of times to compute the metric of the best gen of the generation
        :return:
        '''
        best = population[ranking[0][0]]
        all_fits = [ranking[0][1]]
        if best.__repr__() not in self.best_fit_history.keys():
            for i in range(1, iters):
                all_fits.append(best.fitness())
            self.best_fit_history[best.__repr__()] = all_fits
            self.history_fitness[best.__repr__()] = np.mean(all_fits)

    def evolve(self, show=True):
        population = self.initial_population()
        ti = time()
        start_time = datetime.datetime.now()
        limit_time = start_time + timedelta(hours=self.training_hours)
        for generation in range(self.num_generations + 1):                
            ranking = self.rank(population)
            if self.save_pop:
                self.population_history.append(population)
            self.validate_best(ranking, population)
            self.actualize_history(generation, ranking)
            if self.num_generations <= 10 and show:
                print("%d) best fit: %0.3f in batch time: %0.2f" % (generation + 1, ranking[0][1], time() - ti))
                ti = time()
            elif show and (generation % int(self.num_generations / 10) == 0):
                print("%d) best fit: %0.3f in batch time: %0.2f" % (generation + 1, ranking[0][1], time() - ti))
                ti = time()

            next_generation, all_parents = self.parent_selector.next_gen(population, self.offspring_size)
            population = self.replace(population, next_generation, all_parents)
            if ( start_time > limit_time ):
                break

        ranking = self.rank(population)
        self.validate_best(ranking, population)
        win_idx = ranking[0][0]
        best_fit = ranking[0][1]
        winner = population[win_idx]
        if self.statistical_validation:
            print("Making statistical validation")
            winner_data = self.best_fit_history[winner.__repr__()]
            benchmark_data = np.array([self.chromosome.fitness() for i in range(len(winner_data)) ])
            print("Logging data:")
            print(winner_data)
            print(benchmark_data)
            print("Benchmark score: %0.4f. Winner score: %0.4f" % (np.mean(benchmark_data), np.mean(winner_data)))
            t_value, p_value = stats.ttest_ind(winner_data, benchmark_data)
            print("t = " + str(t_value))
            print("p = " + str(p_value))
        if show:
            print("Best Gen -> ", winner)
            print("With Fitness: %0.3f" % best_fit)
            self.show_history()
        return winner, best_fit, ranking

