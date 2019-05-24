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

    def __init__(self, chromosome, parent_selector, fitness, generations=70, num_population=20,
                 maximize_fitness=True, statistical_validation=True, training_hours=1e3,
                 folder=None, save_progress=True):
        '''
        Class to generate a basic Genetic Algorithms.

        :param chromosome: object of class Chromosome
        :param parent_selector: object of class ParentSelector
        :param fitness: The Fitness Object that eval chromosomes's fitness
        :param generations: Number of generations to evolve the population
        :param num_population: Number of individuals of the population
        :param maximize_fitness: If the fitness has to be maximized (True) or minimized (false)
        :param statistical_validation: if make a statical validation of the computed fitness, computing it a given number
         of times and/or making cross validation.
        :param training_hours: the total hours than the genetic algorithm will look for a solution
        :param folder: the folder name where the progress are saving and loaded from
        :param save_progress: if save the progress on each generation
        '''
        self.num_generations = generations
        self.pop_size = num_population
        self.chromosome = chromosome
        self.parent_selector = parent_selector
        self.fitness_evaluator = fitness
        self.statistical_validation = statistical_validation
        self.maximize = maximize_fitness
        self.history = np.empty((self.pop_size, self.num_generations + 1))
        self.history_fitness = {}
        self.best_fit_history = {}
        self.parent_selector.set_genetic_algorithm(self)
        self.training_hours = training_hours
        self.filename = self.get_file_to_save(folder)
        self.generation = 0
        self.population = []
        self.save_progress = save_progress
        print("Genetic algorithm params")
        print("Number of generations: %d" % self.num_generations)
        print("Population size: %d" % self.pop_size)
        if self.save_progress:
            print("Folder to save: %s" % self.filename)

    @staticmethod
    def load_genetic_algorithm(filename=None, folder=None):
        assert (filename, folder) != (None, None)
        if filename is None:
            files = os.listdir(folder)
            # Files in format: id_Y-M-D-h-m
            id_files = np.array([f.split("_")[0] for f in files], dtype=np.int32)
            filename = files[int(np.argmax(id_files))]
            filename = os.path.join(folder, filename, 'GA_experiment')
            print("Loading file %s" % filename)
        """ Static access method. """
        infile = open(filename, 'rb')
        generational = pickle.load(infile)
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

    def maybe_save_genetic_algorithm(self, verbose=0, force=False):
        if not self.save_progress or not force:
            return
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

    def create_random_individual(self):
        return self.chromosome.random_individual()

    def create_simple_individual(self):
        return self.chromosome.simple_individual()

    def initial_population(self):
        population = []
        for i in range(self.pop_size):
            population.append(self.create_random_individual())
        return population

    @staticmethod
    def mutation(offspring):
        for chromosome in offspring:
            chromosome.mutate()
        return offspring

    def actualize_history(self, generation, rank):
        for i in range(len(rank)):
            self.history[i, generation] = rank[i][1]

    def rank(self, population):
        fitness_result = {}
        population_ids_to_eval = []
        for i in range(self.pop_size):
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                population_ids_to_eval.append(i)
                #self.history_fitness[gen] = population[i].fitness()
            else:
                fitness_result[i] = self.history_fitness[gen]
        evaluated_fitness = self.fitness_evaluator.eval_list([population[id_to_eval] for
                                                              id_to_eval in population_ids_to_eval])
        for fit, i in zip(evaluated_fitness, population_ids_to_eval):
            fitness_result[i] = fit
            self.history_fitness[population[i].__repr__()] = fit
        return sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=self.maximize)

    def show_history(self):
        self.show_history_()
        self.show_history_(zoom=True)
        
    def show_history_(self, zoom=False):
        colors = np.array([[31, 119, 180], [255, 127, 14]]) / 255.
        h = self.history
        epochs = np.linspace(1, h.shape[1], h.shape[1])
        bests = [np.min(h, axis=0), np.max(h, axis=0)][self.maximize]
        plt.figure(figsize=(10, 5))
        s = 6
        for a in h:
            plt.scatter(epochs, a, s=s, color='k', alpha=0.5, marker='.')
        plt.scatter(epochs, bests, color=colors[0], s=s * 5, marker='*')
        plt.plot(epochs, np.mean(h, axis=0), color=colors[1], lw=1, label='mean', linestyle='--')
        plt.plot(epochs, bests, color=colors[0], lw=1, label='best', linestyle='--')
        if zoom:
            last_gen = h[:, -1]
            lim_inf_y = [np.min(last_gen), np.mean(last_gen)][self.maximize]
            lim_sup_y = [np.mean(last_gen), np.max(last_gen)][self.maximize]
            lim_inf_y = lim_inf_y - (lim_sup_y - lim_inf_y) * 0.1
            lim_sup_y = lim_sup_y + (lim_sup_y - lim_inf_y) * 0.1
            plt.ylim(lim_inf_y, lim_sup_y)
        plt.grid()
        plt.legend()
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
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

    def replace(self, population, rank, next_generation, all_parents):
        fitness_result = dict(rank)
        ids = list(fitness_result.keys())[:-len(next_generation)]
        return [population[i] for i in ids] + next_generation

    def maybe_validate_best(self, ranking, population, iters=5):
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
        if not self.statistical_validation:
            return ranking
        best = population[ranking[0][0]]
        all_fits = [ranking[0][1]]
        val_rank = dict(ranking)
        if best.__repr__() not in self.best_fit_history.keys():
            all_fits += self.fitness_evaluator.eval_list([best for _ in range(1, iters)])
            self.best_fit_history[best.__repr__()] = all_fits
            self.history_fitness[best.__repr__()] = np.mean(all_fits)
            val_rank[ranking[0][0]] = np.mean(all_fits)
        return sorted(val_rank.items(), key=operator.itemgetter(1), reverse=self.maximize)

    def evolve(self, show=True):
        if self.generation == 0 or self.population == []:
            self.population = self.initial_population()
            print("Creating Initial population")
        self.start_time = datetime.datetime.now()
        self.limit_time = self.start_time + timedelta(hours=self.training_hours)
        print("\nStart evolution process...\n")
        ti = time()

        for self.generation in range(self.generation, self.num_generations):
            ranking = self.rank(self.population)

            # Make statical validation if is necessary
            ranking = self.maybe_validate_best(ranking, self.population)
            self.actualize_history(self.generation, ranking)

            # To show the progress of the evolution
            if (self.num_generations <= 10 or (self.generation % int(self.num_generations / 10) == 0)) and show:
                print("%d) best fit: %0.3f in batch time: %0.2f mins" %
                      (self.generation + 1, ranking[0][1], (time() - ti)/60.))
                print("Current winner:")
                print(self.population[ranking[0][0]])
                      
            # Save the generation
            self.maybe_save_genetic_algorithm(verbose=True)

            next_generation, all_parents = self.parent_selector.next_gen(self.population, ranking, self.offspring_size)
            self.population = self.replace(self.population, ranking, next_generation, all_parents)

            if datetime.datetime.now() > self.limit_time:
                self.maybe_save_genetic_algorithm(verbose=True)
                win_idx, best_fit = ranking[0]
                winner = self.population[win_idx]
                print("Best fit 'til now : %0.4f" % best_fit)
                print(winner)
                return winner, best_fit, ranking
                
        ranking = self.rank(self.population)
        ranking = self.maybe_validate_best(ranking, self.population)
        self.actualize_history(self.generation + 1, ranking)

        win_idx, best_fit = ranking[0]
        winner = self.population[win_idx]
        fit_test = self.maybe_make_statistical_validation(winner)

        if show:
            print("Best Gen -> \n%s" % winner)
            print("With Fitness (val): %0.4f and (test): %0.4f" % (best_fit, fit_test))
            self.show_history()
        return winner, best_fit, ranking

    def maybe_make_statistical_validation(self, winner):
        if not self.statistical_validation:
            return self.fitness_evaluator.calc(winner, test=True)
        print("Making statistical validation")
        winner_data_val = self.best_fit_history[winner.__repr__()]
        benchmark_data_val = self.fitness_evaluator.eval_list([self.chromosome for _ in winner_data_val])
        # winner_data    = np.array(winner.cross_val(exclude_first=False, test=True))
        # benchmark_data = np.array(self.chromosome.cross_val(exclude_first=False, test=True))
        print("Benchmark Val score: %0.4f. Winner Val score: %0.4f" % (
        np.mean(benchmark_data_val), np.mean(winner_data_val)))
        t_value, p_value = stats.ttest_ind(winner_data_val, benchmark_data_val)
        print("t = %0.4f, p = %0.4f" % (t_value, p_value))
        winner_data_test = self.fitness_evaluator.eval_list([winner for _ in winner_data_val], test=True)
        benchmark_data_test = self.fitness_evaluator.eval_list([self.chromosome for _ in winner_data_val], test=True)
        print("Benchmark Test score: %0.4f. Winner Test score: %0.4f" % (
        np.mean(benchmark_data_test), np.mean(winner_data_test)))
        t_value, p_value = stats.ttest_ind(winner_data_test, benchmark_data_test)
        print("t = %0.4f, p = %0.4f" % (t_value, p_value))
        return np.mean(winner_data_test)

