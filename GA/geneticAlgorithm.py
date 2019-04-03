import numpy as np
import operator
import matplotlib.pyplot as plt


class GeneticAlgorithm(object):

    def __init__(self, chromosome, parent_selector, generations=70, num_population=20, crossover_prob=0.5,
                 mutation_prob=0.7, maximize_fitness=True):
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

        self.maximize = maximize_fitness
        self.history = np.empty((self.pop_size, self.num_generations + 1))
        self.history_fitness = {}
        self.parent_selector.set_genetic_algorithm(self)
        print("Generic algorith params:")
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

    def evolve(self, show=True):
        population = self.initial_population()
        for generation in range(self.num_generations + 1):
            ranking = self.rank(population)
            self.actualize_history(generation, ranking)
            if self.num_generations <= 10 and show:
                print("%d) best fit: %0.3f" % (generation + 1, ranking[0][1]))
            elif show and (generation % int(self.num_generations / 10) == 0):
                print("%d) best fit: %0.3f" % (generation + 1, ranking[0][1]))
            if generation == self.num_generations:
                break
            next_generation, all_parents = self.parent_selector.next_gen(population, self.offspring_size)
            population = self.replace(population, next_generation, all_parents)

        ranking = self.rank(population)
        win_idx = ranking[0][0]
        best_fit = ranking[0][1]
        winner = population[win_idx]
        if show:
            print("Best Gen -> ", winner)
            print("With Fitness: %0.3f" % best_fit)
            self.show_history()
        return winner, best_fit, ranking

