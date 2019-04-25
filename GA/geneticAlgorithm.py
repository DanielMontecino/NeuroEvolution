import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats
from time import time
import datetime
from datetime import timedelta
import os
import pickle


class GeneticAlgorithm(object):

    def __init__(self, chromosome, parent_selector, generations=70, num_population=20,
                 maximize_fitness=True, statistical_validation=True, training_hours=1e3,
                 folder=None):
        '''
        Class to generate a basic Genetic Algorithms.

        :param chromosome: object of class Chromosome
        :param parent_selector: object of class ParentSelector
        :param generations: Number of generations to evolve the population
        :param num_population: Number of individuals of the population
        :param maximize_fitness: If the fitness has to be maximized (True) or minimized (false)
        '''
        self.num_generations = generations
        self.pop_size = num_population
        self.chromosome = chromosome
        self.parent_selector = parent_selector
        self.statistical_validation = statistical_validation
        self.maximize = maximize_fitness
        self.history = np.empty((self.pop_size, self.num_generations + 1))
        self.history_fitness = {}
        self.population_history = []
        self.best_fit_history = {}
        self.parent_selector.set_genetic_algorithm(self)
        self.training_hours = training_hours
        self.filename = self.get_file_to_save(folder)
        self.generation = 0
        print("Genetic algorithm params")
        print("Number of generations: %d" % self.num_generations)
        print("Population size: %d" % self.pop_size)
        print("Folder to save: %s" % self.filename)

    @staticmethod
    def load_genetic_algorithm(filename=None, folder=None):
        assert (filename, folder)!=(None, None)
        if filename is None:
            files = os.listdir(folder)
            # Files in format: id_Y-M-D-h-m
            id_files = np.array([f.split("_")[0] for f in files], dtype=np.int32)
            filename = files[np.argmax(id_files)]
            filename = os.path.join(folder, filename, 'GA_experiment')
            print("Loading file %s" % filename)
        """ Static access method. """
        infile = open(filename, 'rb')
        generational = pickle.load(infile)
        #generational.chromosome.reset_params()
        infile.close()
        return generational

    @staticmethod
    def get_file_to_save(folder):
        if folder is None:
            return None
        os.makedirs(folder, exist_ok=True)
        exps = os.listdir(folder)
        exp_indices = np.array([exp.split("_")[0] for exp in exps], dtype=np.int32)
        if exp_indices.size == 0:
            new_ind = 0
        else:
            new_ind = np.max(exp_indices) + 1
        str_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        new_folder = os.path.join(folder, str(new_ind) + "_" + str_time)
        os.makedirs(new_folder, exist_ok=True)
        filename = os.path.join(new_folder, 'GA_experiment')
        return filename

    def save_genetic_algorithm(self, verbose=0):
        if self.filename is None:
            print("Error!, folder is not defined")
            return
        if verbose:
            print("Saving...", end=" ")
        ti = time()
        outfile = open(self.filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()
        if verbose:
            print("Elapsed saved time: %0.3f" % (time() - ti))

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
        print("offspring size: %d\n" % self.offspring_size)

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
        val_rank = dict(ranking)
        if best.__repr__() not in self.best_fit_history.keys():
            #all_fits += best.cross_val()
            for i in range(1, iters):
                all_fits.append(best.fitness())
            self.best_fit_history[best.__repr__()] = all_fits
            self.history_fitness[best.__repr__()] = np.mean(all_fits)
            val_rank[ranking[0][0]] = np.mean(all_fits)
        return sorted(val_rank.items(), key=operator.itemgetter(1), reverse=self.maximize)

    def evolve(self, show=True):
        if self.generation == 0:
            self.population = self.initial_population()
            print("Initial population")
            for i, p in enumerate(self.population):
                print("Individual %d" % (i+1))
                print(p)
        self.start_time = datetime.datetime.now()
        self.limit_time = self.start_time + timedelta(hours=self.training_hours)
        print("\nStart evolution process...\n")
        ti = time()
        for self.generation in range(self.generation, self.num_generations + 1):
            ranking = self.rank(self.population)
            self.population_history.append(self.population)
            if self.statistical_validation:
                ranking = self.validate_best(ranking, self.population)
            self.actualize_history(self.generation, ranking)
            if (self.num_generations <= 10 or (self.generation % int(self.num_generations / 10) == 0)) and show:
                print("%d) best fit: %0.3f in batch time: %0.2f mins" %
                      (self.generation + 1, ranking[0][1], (time() - ti)/60.))
                self.save_genetic_algorithm(verbose=True)

            next_generation, all_parents = self.parent_selector.next_gen(self.population, self.offspring_size)
            self.population = self.replace(self.population, next_generation, all_parents)

            if self.start_time > self.limit_time:
                self.save_genetic_algorithm(verbose=True)
                win_idx = ranking[0][0]
                best_fit = ranking[0][1]
                winner = self.population[win_idx]
                print("Best fit 'til now : %0.4f" % best_fit)
                print(winner)
                return winner, best_fit, ranking
                
        ranking = self.rank(self.population)
        if self.statistical_validation:
            ranking = self.validate_best(ranking, self.population)
            self.actualize_history(self.generation, ranking)
        win_idx = ranking[0][0]
        best_fit = ranking[0][1]
        winner = self.population[win_idx]

        if self.statistical_validation:
            self.make_statistical_validation(winner)
        if show:
            print("Best Gen -> \n%s" % winner)
            print("With Fitness (val): %0.3f" % best_fit)
            self.show_history()
        return winner, best_fit, ranking

    def make_statistical_validation(self, winner):
        print("Making statistical validation")
        winner_data_val = self.best_fit_history[winner.__repr__()]
        benchmark_data_val = [self.chromosome.fitness() for _ in winner_data_val]
        # winner_data    = np.array(winner.cross_val(exclude_first=False, test=True))
        # benchmark_data = np.array(self.chromosome.cross_val(exclude_first=False, test=True))
        print("Benchmark Val score: %0.4f. Winner Val score: %0.4f" % (
        np.mean(benchmark_data_val), np.mean(winner_data_val)))
        t_value, p_value = stats.ttest_ind(winner_data_val, benchmark_data_val)
        print("t = %0.4f, p = %0.4f" % (t_value, p_value))
        winner_data_test = [winner.fitness(test=True) for _ in winner_data_val]
        benchmark_data_test = [self.chromosome.fitness(test=True) for _ in winner_data_val]
        print("Benchmark Test score: %0.4f. Winner Test score: %0.4f" % (
        np.mean(benchmark_data_test), np.mean(winner_data_test)))
        t_value, p_value = stats.ttest_ind(winner_data_test, benchmark_data_test)
        print("t = %0.4f, p = %0.4f" % (t_value, p_value))

