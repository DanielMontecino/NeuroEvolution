import os
from time import time
import numpy as np
import sys
import operator
import pickle
import datetime
sys.path.append('../')


class RandomSearcher:

    def __init__(self, individual_example, fitness_evaluator, samples=None, time_limit=None, population_batch=10, maximize=False, save=True,
                 folder=None):
        self.time_limit = time_limit
        self.population_size = population_batch
        self.individual = individual_example
        self.fitness_evaluator = fitness_evaluator
        self.maximize = maximize
        self.save_progress = save
        self.filename = self.get_file_to_save(folder) if self.save_progress else None
        self.history_fitness = {}
        self.elapsed_time = 0
        self.samples = samples

    def get_best_individual(self):
        sorted_individuals = sorted(self.history_fitness.items(), key=operator.itemgetter(1), reverse=self.maximize)
        return sorted_individuals[0]  # best individual, fitness

    def next_population(self, samples):
        population = [self.individual.random_individual() for _ in range(samples)]
        return population

    def evaluate_population(self, population):
        population_to_eval = [individual for individual in population if individual.__repr__() not in
                              self.history_fitness.keys()]
        evaluated_fitness = self.fitness_evaluator.eval_list(population_to_eval)
        for i in range(len(population_to_eval)):
            fitness_result = evaluated_fitness[i]
            individual = population_to_eval[i]
            self.history_fitness[individual.__repr__()] = fitness_result

    def evolve(self):
        while self.stop_condition():
            init_time = time()
            population = self.next_population(self.population_size)
            self.evaluate_population(population)
            elapsed_time = time() - init_time
            self.elapsed_time += elapsed_time
            self.maybe_save()
            self.println("Elapsed time: %0.1f minutes" % (self.elapsed_time/60))
            self.println("Best until now:")
            best, fit = self.get_best_individual()
            self.println("Best individual with fitness: %0.3f" % fit)
            self.println("%s" % best)

        winner, val_fitness = self.get_best_individual()
        self.fitness_evaluator.calc(winner, test=True, precise_mode=True)

    def stop_condition(self):
        if self.samples is not None:
            return len(self.history_fitness.keys()) < self.samples
        else:
            return (self.elapsed_time/3600) < self.time_limit

    @staticmethod
    def load_genetic_algorithm(filename=None, folder=None):
        assert (filename, folder) != (None, None)
        if filename is None:
            files = os.listdir(folder)
            # Files in format: id_Y-M-D-h-m
            id_files = np.array([f.split("_")[0] for f in files], dtype=np.int32)
            filename = files[int(np.argmax(id_files))]
            filename = os.path.join(folder, filename, 'RS_experiment')
            print("Loading file %s" % filename)
        """ Static access method. """
        infile = open(filename, 'rb')
        generational = pickle.load(infile)
        infile.close()
        return generational

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
        new_folder = RandomSearcher.get_folder_to_save(folder)
        filename = os.path.join(new_folder, 'GA_experiment')
        return filename

    def maybe_save(self, verbose=0):
        if not self.save_progress:
            return
        if self.filename is None:
            print("Error!, folder is not defined")
            return
        self.println("Saving...", end=" ")
        ti = time()
        outfile = open(self.filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def println(self, string, end="\n"):
        if self.filename is not None:
            folder = os.path.dirname(self.filename)
            file_to_save = os.path.join(folder, "std_experiment")
            with open(file_to_save, 'a') as f:
                f.write(string + end)
        if self.println:
            print(string, end=end)

