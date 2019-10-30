import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats
from time import time
import datetime
from datetime import timedelta
import os
import pickle

from GA.parentSelector.parentSelector import TournamentSelection


class GeneticAlgorithm(object):

    def __init__(self, chromosome, parent_selector, fitness, generations=70, num_population=20,
                 maximize_fitness=True, statistical_validation=True, training_hours=1e3,
                 folder=None, save_progress=True, age_survivors_rate=0.0, precision_val=False,
                 precision_individuals=5, printn=False, perform_evo=False):
        '''
        Class to generate a basic Genetic Algorithms.

        :type save_progress: bool
        :param chromosome: object of class Chromosome
        :param parent_selector: object of class ParentSelector
        :param fitness: The Fitness Object that eval chromosomes's fitness
        :param generations: Number of generations to evolve the population
        :param num_population: Number of individuals of the population
        :param maximize_fitness: If the fitness has to be maximized (True) or minimized (false)
        :param statistical_validation: if make a statical validation of the computed fitness, computing it a given
         number of times and/or making cross validation.
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
        self.parent_selector.set_params(self.maximize, self.history_fitness)
        self.training_hours = training_hours
        self.save_progress = save_progress
        self.filename = self.get_file_to_save(folder) if self.save_progress else None
        self.generation = 0
        self.population = []
        self.age_survivors_rate = self.set_age_survivors_rate(age_survivors_rate)
        self.best_individual = {'winner': chromosome, 'best_fit': None}
        self.make_precision_validation = precision_val
        self.N_precision_individuals = precision_individuals
        self.paint = printn
        self.perfm_evo = [] if perform_evo else None
        if self.make_precision_validation:
            self.history_precision_fitness = {}
            self.history_precision = np.empty((self.N_precision_individuals, self.num_generations + 1))
        self.print("Number of individuals eliminated by age: %d" % self.age_survivors_rate)
        self.print("Genetic algorithm params")
        self.print("Number of generations: %d" % self.num_generations)
        self.print("Population size: %d" % self.pop_size)
        if self.save_progress:
            self.print("Folder to save: %s" % self.filename)

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

    def print(self, string, end="\n"):
        if self.paint:
            print(string, end=end)

    @staticmethod
    def get_folder_to_save(folder):
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
        return new_folder

    @staticmethod
    def get_file_to_save(folder):
        new_folder = GeneticAlgorithm.get_folder_to_save(folder)
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
            self.print("Error!, folder is not defined")
            return
        if verbose:
            self.print("Saving...", end=" ")
        ti = time()
        outfile = open(self.filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()
        if verbose:
            self.print("Elapsed saved time: %0.3f" % (time() - ti))

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
        if self.make_precision_validation:
            best_generation = [self.population[rank[i][0]] for i in range(self.N_precision_individuals)]
            best_generation_fitness = [self.history_precision_fitness[individual.__repr__()] for
                                       individual in best_generation]
            for i, fit in enumerate(best_generation_fitness):
                self.history_precision[i, generation] = fit

    def rank(self, population):
        fitness_result = {}
        population_ids_to_eval = []
        for i in range(len(population)):
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                population_ids_to_eval.append(i)
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
        self.print("num parents: %d" % self.num_parents)
        self.print("offspring size: %d\n" % self.offspring_size)
        self.start_time = None
        self.limit_time = None
        self.samples = 3

    @staticmethod
    def increase_population_age(population):
        for individual in population:
            individual.increase_age()

    def replace(self, population, rank, next_generation):
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

    def maybe_validate_best(self, ranking, population):
        '''
        If the evaluation process of each gen is not deterministic, it's necessary
        to do a statical validation of the performance. To do this, this function
        makes a validation of the best gen of the generation in the population, evaluating it
        a 'self.samples' number of times, and add a list with the results to a dictionary.

        :param ranking:     The ranking of the population in the actual generation as a dict with
                            the name of each gen as a key, and a tuple (id, fitnees) as a value
        :param population:  Al the genes in the actual generation
        :return: ranking with the population and its fitness
        '''
        if self.statistical_validation:
            c = 0
            while population[ranking[0][0]].__repr__() not in self.best_fit_history.keys():
                best = population[ranking[0][0]]
                all_fits = [ranking[0][1]]
                val_rank = dict(ranking)
                all_fits += self.fitness_evaluator.eval_list([best for _ in range(1, self.samples)])
                self.best_fit_history[best.__repr__()] = all_fits
                self.history_fitness[best.__repr__()] = np.mean(all_fits)
                val_rank[ranking[0][0]] = np.mean(all_fits)
                ranking = sorted(val_rank.items(), key=operator.itemgetter(1), reverse=self.maximize)
                c += 1
                if c > 100:
                    print("Error in while!")
                    raise AssertionError
        generational_best = population[ranking[0][0]]
        fit_best_of_gen = self.history_fitness[generational_best.__repr__()]
        self.validate_best(generational_best, fit_best_of_gen)
        return ranking

    def validate_best(self, generational_best, generational_best_fitness):
        if self.best_individual['best_fit'] is None:
            self.best_individual['winner'] = generational_best
            self.best_individual['best_fit'] = generational_best_fitness
        elif self.maximize and generational_best_fitness >= self.best_individual['best_fit']:
            self.best_individual['winner'] = generational_best
            self.best_individual['best_fit'] = generational_best_fitness
        elif (not self.maximize) and generational_best_fitness <= self.best_individual['best_fit']:
            self.best_individual['winner'] = generational_best
            self.best_individual['best_fit'] = generational_best_fitness

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
        best_precision_individual = best_generation_individuals[int(arg)]
        best_precision_individual_fit = best_generation_fitness[int(arg)]

        self.validate_best(best_precision_individual, best_precision_individual_fit)
        return best_generation_individuals

    def get_best(self):
        return self.best_individual['winner'], self.best_individual['best_fit']

    def initialize_evolution(self):
        if self.generation == 0 or self.population == []:
            self.population = self.initial_population()
            self.population[0] = self.chromosome
            self.print("Creating Initial population")
        self.start_time = datetime.datetime.now()
        self.limit_time = self.start_time + timedelta(hours=self.training_hours)
        self.print("\nStart evolution process...\n")

    def reset(self):
        self.generation = 0
        self.population = []
        self.history = np.empty((self.pop_size, self.num_generations + 1))
        self.history_fitness = {}
        self.best_fit_history = {}
        self.parent_selector.set_params(self.maximize, self.history_fitness)
        self.best_individual = {'winner': self.chromosome, 'best_fit': None}
        self.perfm_evo = [] if self.perfm_evo is not None else None
        if self.make_precision_validation:
            self.history_precision_fitness = {}
            self.history_precision = np.empty((self.N_precision_individuals, self.num_generations + 1))

    def evolve(self, show=True):
        self.initialize_evolution()
        ti = time()

        for self.generation in range(self.generation, self.num_generations):
            ranking = self.rank(self.population)
            # Make statical validation if is necessary
            ranking = self.maybe_validate_best(ranking, self.population)
            self.maybe_precision_validation(ranking, self.population)
            self.actualize_history(self.generation, ranking)
            self.increase_population_age(self.population)
            # To show the evolution's progress
            if (self.num_generations <= 10 or (self.generation % int(self.num_generations / 10) == 0)) and show:
                best, fit = self.get_best()
                self.print("%d) best fit: %0.3f in batch time: %0.2f mins" %
                           (self.generation + 1, fit, (time() - ti) / 60.))
                self.print("Current winner:")
                self.print(best)

            # Save the generation
            self.maybe_save_genetic_algorithm(verbose=True)

            if datetime.datetime.now() > self.limit_time:
                self.maybe_save_genetic_algorithm(verbose=True)
                winner, best_fit = self.get_best()
                self.print("Best fit 'til generation %d : %0.4f" % (self.generation, best_fit))
                self.print(winner)
                break

            next_generation, all_parents = self.parent_selector.next_gen(self.population, ranking, self.offspring_size)
            self.population = self.replace(self.population, ranking, next_generation)

            if self.perfm_evo is not None:
                _, fit_test = self.finishing_evolution(show=False)
                self.perfm_evo.append(fit_test)

        ranking = self.rank(self.population)
        ranking = self.maybe_validate_best(ranking, self.population)
        self.maybe_precision_validation(ranking, self.population)
        self.actualize_history(self.generation + 1, ranking)

        winner, fit_test = self.finishing_evolution(show=show)
        if self.perfm_evo is not None:
            return winner, fit_test, self.perfm_evo
        else:
            return winner, fit_test, ranking

    def finishing_evolution(self, show):
        winner, best_fit = self.get_best()
        fit_test = self.maybe_make_statistical_validation(winner)
        self.best_individual['test'] = fit_test
        self.maybe_save_genetic_algorithm()

        if show:
            self.print("Best Gen -> \n%s" % winner)
            self.print("With Fitness (val): %0.4f and (test): %0.4f" % (best_fit, fit_test))
            self.show_history()
        return winner, fit_test

    def maybe_make_statistical_validation(self, winner):
        if not self.statistical_validation:
            return self.fitness_evaluator.calc(winner, test=True, precise_mode=True)
        self.print("\nMaking statistical validation")
        winner_data_test = self.fitness_evaluator.eval_list([winner for _ in range(self.samples)], test=True,
                                                            precise_mode=True)
        return np.mean(winner_data_test)
        try:
            winner_data_val = self.best_fit_history[winner.__repr__()]
        except KeyError:
            self.print("Winner is not in best_fit_history")
            winner_data_val = [self.history_fitness[winner.__repr__()]]
            winner_data_val += self.fitness_evaluator.eval_list([winner for _ in range(1, self.samples)])

        if self.chromosome.__repr__() in self.best_fit_history.keys():
            benchmark_data_val = self.best_fit_history[self.chromosome.__repr__()]
        else:
            benchmark_data_val = [self.history_fitness[self.chromosome.__repr__()]]
            benchmark_data_val += self.fitness_evaluator.eval_list([self.chromosome for _ in range(1, self.samples)])

        # winner_data    = np.array(winner.cross_val(exclude_first=False, test=True))
        # benchmark_data = np.array(self.chromosome.cross_val(exclude_first=False, test=True))
        self.print("Benchmark Val score: %0.4f. Winner Val score: %0.4f" % (
            np.mean(benchmark_data_val), np.mean(winner_data_val)))
        t_value, p_value = stats.ttest_ind(winner_data_val, benchmark_data_val)
        self.print("t = %0.4f, p = %0.4f" % (t_value, p_value))

        winner_data_test = self.fitness_evaluator.eval_list([winner for _ in range(self.samples)], test=True,
                                                            precise_mode=True)
        benchmark_data_test = self.fitness_evaluator.eval_list([self.chromosome for _ in range(self.samples)],
                                                               test=True, precise_mode=True)
        self.print("Benchmark Test score: %0.4f. Winner Test score: %0.4f" % (
            np.mean(benchmark_data_test), np.mean(winner_data_test)))
        t_value, p_value = stats.ttest_ind(winner_data_test, benchmark_data_test)
        self.print("t = %0.4f, p = %0.4f" % (t_value, p_value))
        return np.mean(winner_data_test)


class TwoLevelGA(GenerationalGA):

    def __init__(self, chromosome, fitness, generations, population_first_level, population_second_level,
                 training_hours, save_progress, maximize_fitness, statistical_validation, folder, start_level2,
                 frequency_second_level=1, perform_evo=False):

        self.population_1 = []
        self.population_2 = []
        self.scnd_level_freq = frequency_second_level
        self.num_pop_level1 = population_first_level
        self.num_pop_level2 = population_second_level
        self.ti = None
        self.start_level2 = start_level2

        self.num_parents_level1 = self.num_pop_level1 // 4
        self.num_parents_level2 = self.num_pop_level2 // 2

        self.offspring_size_level1 = self.num_pop_level1 - self.num_parents_level1
        self.offspring_size_level2 = self.num_pop_level2 - self.num_parents_level2

        self.parent_selector_level1 = TournamentSelection(self.num_parents_level1)
        self.parent_selector_level2 = TournamentSelection(self.num_parents_level2)

        super().__init__(num_parents=self.num_parents_level1, chromosome=chromosome,
                         parent_selector=self.parent_selector_level1, fitness=fitness,
                         generations=generations, num_population=self.num_pop_level1,
                         maximize_fitness=maximize_fitness, statistical_validation=statistical_validation,
                         training_hours=training_hours, folder=folder, save_progress=save_progress,
                         age_survivors_rate=0.0, precision_val=True, precision_individuals=self.num_pop_level2,
                         perform_evo=perform_evo)

        self.parent_selector_level1.set_params(self.maximize, self.history_fitness)
        self.parent_selector_level2.set_params(self.maximize, self.history_precision_fitness)
        self.print("Population size level one: %d" % self.num_pop_level1)
        self.print("Population size level two: %d" % self.num_pop_level2)
        self.print("Number of parents level one:", self.num_parents_level1)
        self.print("Number of parents level two:", self.num_parents_level2)
        self.print("Offspring size level one:", self.offspring_size_level1)
        self.print("Offspring size level two:", self.offspring_size_level2)

    def rank_precision(self, population):
        fitness_result = {}
        population_ids_to_eval = []
        for i in range(len(population)):
            gen = population[i].__repr__()
            if gen not in self.history_precision_fitness.keys():
                population_ids_to_eval.append(i)
            else:
                fitness_result[i] = self.history_precision_fitness[gen]
        evaluated_fitness = self.fitness_evaluator.eval_list([population[id_to_eval] for
                                                              id_to_eval in population_ids_to_eval], precise_mode=True)
        for fit, i in zip(evaluated_fitness, population_ids_to_eval):
            fitness_result[i] = fit
            self.history_precision_fitness[population[i].__repr__()] = fit
        return sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=self.maximize)

    @staticmethod
    def actualize_given_history(generation, rank, history):
        for i in range(len(rank)):
            history[i, generation] = rank[i][1]

    def evolve(self, show=True):
        self.initialize_evolution()
        self.population_1 = self.population
        self.ti = time()

        for self.generation in range(self.generation, self.num_generations):
            # Evaluate the population, ranking it, actualize parameters and SAVE
            ranking1 = self.evaluate_population(level=1)

            if self.generation < self.start_level2:
                next_generation_level1, all_parents_level1 = \
                    self.parent_selector_level1.next_gen(self.population_1, ranking1, self.offspring_size_level1)
                self.population_1 = self.replace(self.population_1, ranking1, next_generation_level1)

            elif self.generation == self.start_level2:
                # Get the best of the first level
                self.population_2 = [self.population_1[ranking1[i][0]] for i in range(self.num_pop_level2)]

                # Evaluate the population, ranking it, actualize parameters and SAVE
                self.evaluate_population(level=2)

                next_generation_level1, all_parents_level1 = \
                    self.parent_selector_level1.next_gen(self.population_1, ranking1, self.offspring_size_level1)
                self.population_1 = self.replace(self.population_1, ranking1, next_generation_level1)

            elif self.generation % self.scnd_level_freq == 0:
                # Evaluate the population, ranking it, actualize parameters and SAVE
                ranking2 = self.evaluate_population(level=2)

                offspring_level2, all_parents_level2 = self.parent_selector_level2.next_gen(self.population_2, ranking2,
                                                                                            self.offspring_size_level2)

                # Get the offspring of the first level
                next_generation_level1, all_parents_level1 = \
                    self.parent_selector_level1.next_gen(self.population_1, ranking1, self.offspring_size_level1)

                # Get the N best of the first level
                best_individuals_level1 = []
                i = 0
                c = 0
                while len(best_individuals_level1) < self.offspring_size_level2:
                    if self.population_1[ranking1[i][0]] not in self.population_2:
                        best_individuals_level1.append(self.population_1[ranking1[i][0]])
                    i += 1
                    c += 1
                    if c > 100:
                        print("Error in while!")
                        raise AssertionError

                # Replace the N heirs of the second level in N random heirs of the first
                next_generation_level1[0:self.offspring_size_level2] = offspring_level2

                # Put the N best individuals from the first level in the offspring of the second
                assert self.offspring_size_level2 == len(best_individuals_level1)
                next_generation_level2 = best_individuals_level1

                # To show the evolution's progress
                self.show_progress(ranking1, ranking2)

                # replace in each population
                self.population_1 = self.replace(self.population_1, ranking1, next_generation_level1)
                self.population_2 = self.replace(self.population_2, ranking2, next_generation_level2)

            else:
                next_generation_level1, all_parents_level1 = \
                    self.parent_selector_level1.next_gen(self.population_1, ranking1, self.offspring_size_level1)
                self.population_1 = self.replace(self.population_1, ranking1, next_generation_level1)

                # Evaluate the population, ranking it, actualize parameters and SAVE
                self.evaluate_population(level=2)

            if datetime.datetime.now() > self.limit_time:
                self.maybe_save_genetic_algorithm(verbose=True)
                winner, best_fit = self.get_best()
                self.print("Best fit 'til generation %d : %0.4f" % (self.generation, best_fit))
                self.print(winner)
                break

            if self.perfm_evo is not None:
                _, fit_test = self.finishing_evolution(show=False)
                self.perfm_evo.append(fit_test)

        self.generation += 1
        ranking1 = self.evaluate_population(level=1)
        ranking2 = self.evaluate_population(level=2)

        # To show the evolution's progress
        self.show_progress(ranking1, ranking2)

        winner, fit_test = self.finishing_evolution(show=show)
        if self.perfm_evo is not None:
            return winner, fit_test, self.perfm_evo
        else:
            return winner, fit_test, ranking2

    def show_progress(self, rank1, rank2):
        best_id_level1, best_fit_level1 = rank1[0]
        best_id_level2, best_fit_level2 = rank2[0]
        self.print("\nGeneration (%d) in %0.2f minutes." % (self.generation, (time() - self.ti) / 60.))
        self.print("Best first level fitness: %0.5f" % best_fit_level1)
        self.print("Best second level fitness: %0.5f" % best_fit_level2)

    def evaluate_population(self, level):
        assert level in [1, 2]
        if level == 1:
            population = self.population_1
            history_fitness = self.history_fitness
            history = self.history
        else:
            population = self.population_2
            history_fitness = self.history_precision_fitness
            history = self.history_precision

        local_ti = time()
        self.print("\n%d) Ranking level %d... " % (self.generation, level), end="")
        self.print("Models to train: %d ..." % self.count_untrained(population, history_fitness), end="")

        if level == 1:
            # Evaluate the models and ranking them
            ranking = self.rank(population)
            self.print("OK (in %0.2f minutes)" % ((time() - local_ti) / 60.0))
            local_ti = time()
            # Make statical validation if is necessary
            ranking = self.maybe_validate_best(ranking, population)
            self.print("Statistical validation in %0.2f minutes" % ((time() - local_ti) / 60.0))
            self.increase_population_age(self.population_1)

        else:
            # Evaluate the models and ranking them
            ranking = self.rank_precision(population)
            self.print("OK (in %0.2f minutes)" % ((time() - local_ti) / 60.0))

        # actualize the global best by
        best_id, best_fit = ranking[0]
        self.validate_best(population[best_id], best_fit)

        # Actualize the historical matrix
        self.actualize_given_history(self.generation, ranking, history)

        # Save the generation
        self.maybe_save_genetic_algorithm(verbose=True)
        return ranking

    @staticmethod
    def count_untrained(population, history_fitness):
        c = 0
        for p in population:
            if p.__repr__() not in history_fitness.keys():
                c += 1
        return c
