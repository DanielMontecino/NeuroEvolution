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
                 folder=None, save_progress=True, age_survivors_rate=0.0, precision_val=False,
                 precision_individuals=5, unlimit_evolve=False):
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
        self.save_progress = save_progress
        self.filename = self.get_file_to_save(folder) if self.save_progress else None
        self.generation = 0
        self.population = []
        self.age_survivors_rate = self.set_age_survivors_rate(age_survivors_rate)
        self.best_individual = {'winner': None, 'best_fit': None}
        self.make_precision_validation = precision_val
        self.N_precision_individuals = precision_individuals
        self.generations_without_improve = 0
        self.time = None
        self.unlimit_evolve = unlimit_evolve
        if self.make_precision_validation:
            self.history_precision_fitness = {}
            self.history_precision = np.empty((self.N_precision_individuals, self.num_generations + 1))
        print("Number of individuals eliminated by age: %d" % self.age_survivors_rate)
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

    def set_age_survivors_rate(self, older_rate):
        if older_rate > 1:
            older_rate = older_rate * 1. / self.pop_size
        elif older_rate < 0:
            older_rate = 0
        return older_rate

    def maybe_save_genetic_algorithm(self, verbose=0, force=False):
        if not self.save_progress and not force:
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
        while generation >= self.history.shape[1]:
            self.history = np.append(self.history, np.zeros((self.history.shape[0], 1)), axis=1)
            self.history_precision = np.append(self.history_precision,
                                               np.zeros((self.history_precision.shape[0], 1)), axis=1)

        for i in range(len(rank)):
            self.history[i, generation] = rank[i][1]
        if self.make_precision_validation:
            best_generation = [self.population[rank[i][0]] for i in range(self.N_precision_individuals)]
            best_generation_fitness = [self.history_precision_fitness[individual.__repr__()] for
                                       individual in best_generation]
            for i, fit in enumerate(best_generation_fitness):
                self.history_precision[i, generation] = fit

    def rank(self, population):
        fitness_result = {}
        population_ids_to_eval = []
        for i in range(self.pop_size):
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                population_ids_to_eval.append(i)
                # self.history_fitness[gen] = population[i].fitness()
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

        if self.make_precision_validation:
            h2 = self.history_precision
            bests = [np.min(h2, axis=0), np.max(h2, axis=0)][self.maximize]
            for a in h2:
                plt.scatter(epochs, a, s=s, color='r', alpha=0.5, marker='.')
            plt.scatter(epochs, bests, color='r', s=s * 5, marker='*')
            plt.plot(epochs, bests, color='r', lw=1, label='precision best', linestyle='--')

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

    def increase_population_age(self, population):
        for individual in population:
            individual.increase_age()
        self.generations_without_improve += 1

    def replace(self, population, rank, next_generation, all_parents):
        ages = [individual.age for individual in population]
        ages_dict = dict(zip(population, ages))
        ages_dict = sorted(ages_dict.items(), key=operator.itemgetter(1), reverse=False)

        n_survivors_by_age = int((len(population) - len(next_generation)) * self.age_survivors_rate)
        survivors_by_age = ages_dict[0:n_survivors_by_age]
        survivors_by_age = list(dict(survivors_by_age).keys())

        population_ids_ordered_by_fitness = list(dict(rank).keys())
        final_survivors = survivors_by_age
        for id_ in population_ids_ordered_by_fitness:
            if len(final_survivors) == len(population) - len(next_generation):
                break
            if population[id_] not in final_survivors:
                final_survivors.append(population[id_])

        return final_survivors + next_generation

    def maybe_validate_best(self, ranking, population, iters=3):
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
        if self.statistical_validation:
            while population[ranking[0][0]].__repr__() not in self.best_fit_history.keys():
                best = population[ranking[0][0]]
                all_fits = [ranking[0][1]]
                val_rank = dict(ranking)
                all_fits += self.fitness_evaluator.eval_list([best for _ in range(1, iters)])
                self.best_fit_history[best.__repr__()] = all_fits
                self.history_fitness[best.__repr__()] = np.mean(all_fits)
                val_rank[ranking[0][0]] = np.mean(all_fits)
                ranking = sorted(val_rank.items(), key=operator.itemgetter(1), reverse=self.maximize)
        generational_best = population[ranking[0][0]]
        fit_best_of_gen = self.history_fitness[generational_best.__repr__()]
        self.actualize_best(generational_best, fit_best_of_gen)
        return ranking

    def actualize_best(self, generational_best, generational_best_fitness):
        if self.best_individual['winner'] is None:
            self.best_individual['winner'] = generational_best
            self.best_individual['best_fit'] = generational_best_fitness
            self.generations_without_improve = 0
        elif self.maximize and generational_best_fitness > self.best_individual['best_fit']:
            self.best_individual['winner'] = generational_best
            self.best_individual['best_fit'] = generational_best_fitness
            self.generations_without_improve = 0
        elif (not self.maximize) and generational_best_fitness < self.best_individual['best_fit']:
            self.best_individual['winner'] = generational_best
            self.best_individual['best_fit'] = generational_best_fitness
            self.generations_without_improve = 0

    def maybe_precision_validation(self, ranking, population):

        if not self.make_precision_validation:
            return

        # Get the "n_individuals" best individuals of the generation
        best_generation_individuals = [population[ranking[i][0]] for i in range(self.N_precision_individuals)]

        # Eval the individuals who haven't been evaluated yet (in precision mode)
        individuals_to_evaluate = []
        for individual in best_generation_individuals:
            if individual.__repr__() not in self.history_precision_fitness.keys():
                individuals_to_evaluate.append(individual)

        evaluated_fitness = self.fitness_evaluator.eval_list(individuals_to_evaluate, precise_mode=True)
        # Add newly evaluated individuals to the history
        for fit, individual in zip(evaluated_fitness, individuals_to_evaluate):
            self.history_precision_fitness[individual.__repr__()] = fit

        # verify if there is a new best individual
        # Get the best individual of this generation
        best_generation_fitness = [self.history_precision_fitness[individual.__repr__()]
                                   for individual in best_generation_individuals]
        if self.maximize:
            arg = np.argmax(best_generation_fitness).astype(np.int32)
        else:
            arg = np.min(best_generation_fitness).astype(np.int32)
        best_precision_individual = best_generation_individuals[arg]
        best_precision_individual_fit = best_generation_fitness[arg]

        self.actualize_best(best_precision_individual, best_precision_individual_fit)
        return

    def get_best(self):
        return self.best_individual['winner'], self.best_individual['best_fit']

    def evolve(self, show=True):
        if self.generation == 0 or self.population == []:
            self.population = self.initial_population()
            self.time = 0
            print("Creating Initial population")
        self.start_time = datetime.datetime.now()
        self.limit_time = timedelta(hours=self.training_hours)
        print("\nStart evolution process...\n")
        ti = time()

        lim_generations = [self.num_generations, 10000][self.unlimit_evolve]
        for self.generation in range(self.generation, lim_generations):
            ranking = self.rank(self.population)
            # Make statical validation if is necessary
            ranking = self.maybe_validate_best(ranking, self.population)
            self.maybe_precision_validation(ranking, self.population)
            self.actualize_history(self.generation, ranking)
            self.increase_population_age(self.population)
            # To show the evolution's progress
            if (self.num_generations <= 10 or (self.generation % int(self.num_generations / 10) == 0)) and show:
                best, fit = self.get_best()
                print("%d) best fit: %0.3f in batch time: %0.2f mins" %
                      (self.generation + 1, fit, (time() - ti) / 60.))
                print("Current winner:")
                print(best)

            # Save the generation
            self.maybe_save_genetic_algorithm(verbose=True)

            # Conditions to break the evolution process
            break_for_no_improvement = (self.generations_without_improve > self.generation / 2) and \
                                       self.generation >= self.num_generations

            self.time = datetime.datetime.now() - self.start_time
            break_for_time = self.time > self.limit_time

            if break_for_no_improvement or break_for_time:
                self.maybe_save_genetic_algorithm(verbose=True)
                winner, best_fit = self.get_best()
                if break_for_no_improvement:
                    print("Breaking because there isn't improvement.")
                    print("Current generation: %d" % int(self.generation))
                    print("Generations without improve: %d" % self.generations_without_improve)
                if break_for_time:
                    print("Breaking because time limit was reached")
                    print("Limit time: %0.4f hours" % self.training_hours)
                    print("Elapsed time: %0.4f minutes" % (self.time.seconds/60))
                print("Best fit until generation %d : %0.4f" % (self.generation, best_fit))
                print(winner)
                break
                return winner, best_fit, ranking

            next_generation, all_parents = self.parent_selector.next_gen(self.population, ranking, self.offspring_size)
            self.population = self.replace(self.population, ranking, next_generation, all_parents)

        ranking = self.rank(self.population)
        ranking = self.maybe_validate_best(ranking, self.population)
        self.maybe_precision_validation(ranking, self.population)
        self.actualize_history(self.generation + 1, ranking)

        winner, best_fit = self.get_best()
        val_score, val_std, val_max, test_score, test_std, test_max = self.maybe_make_statistical_validation(winner)
        self.best_individual['test'] = test_score
        self.maybe_save_genetic_algorithm()
        self.time = datetime.datetime.now() - self.start_time
        if show:
            print("Best Gen -> \n%s" % winner)
            print("With Fitness (evo val): %0.4f" % best_fit)
            print("Val results: mean %0.4f, std %0.4f, best %0.4f" % (val_score, val_std, val_max))
            print("Test results: mean %0.4f, std %0.4f, best %0.4f" % (test_score, test_std, test_max))
            print("Total elapsed time: %0.4f" % (self.time.seconds / 60))
            self.show_history()
        return winner, best_fit, ranking

    def maybe_make_statistical_validation(self, winner):
        if not self.statistical_validation:
            val_score, test_score = self.fitness_evaluator.calc(winner, test=True, precise_mode=True)
            return val_score, 0, val_score, test_score, 0, test_score
        print("Making statistical validation")
        winner_data_evo_val = self.best_fit_history[winner.__repr__()]
        winner_data = self.fitness_evaluator.eval_list([winner for _ in winner_data_evo_val], test=True,
                                                       precise_mode=True)

        benchmark_data = self.fitness_evaluator.eval_list([self.chromosome for _ in winner_data_evo_val],
                                                          precise_mode=True, test=True)
        winner_data_val = [data[0] for data in winner_data]
        winner_data_test = [data[1] for data in winner_data]
        benchmark_data_val = [data[0] for data in benchmark_data]
        benchmark_data_test = [data[1] for data in benchmark_data]

        # winner_data    = np.array(winner.cross_val(exclude_first=False, test=True))
        # benchmark_data = np.array(self.chromosome.cross_val(exclude_first=False, test=True))
        print("Benchmark Val score: %0.4f. Winner Val score: %0.4f" % (
            np.mean(benchmark_data_val), np.mean(winner_data_val)))
        t_value, p_value = stats.ttest_ind(winner_data_val, benchmark_data_val)
        print("t = %0.4f, p = %0.4f" % (t_value, p_value))

        print("Benchmark Test score: %0.4f. Winner Test score: %0.4f" % (
            np.mean(benchmark_data_test), np.mean(winner_data_test)))
        t_value, p_value = stats.ttest_ind(winner_data_test, benchmark_data_test)
        print("t = %0.4f, p = %0.4f" % (t_value, p_value))
        val_best = [np.min(winner_data_val), np,max(winner_data_val)][self.maximize]
        test_best = [np.min(winner_data_test), np,max(winner_data_test)][self.maximize]
        return np.mean(winner_data_val), np.std(winner_data_val), val_best, \
               np.mean(winner_data_test), np.std(winner_data_test), test_best
