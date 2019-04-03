import random
import operator
import numpy as np


class ParentSelector(object):

    def __init__(self, maximize=True, history={}):
        self.maximize = maximize
        self.history_fitness = history
        self.ga = None

    def set_genetic_algorithm(self, genetic_algorithm):
        self.ga = genetic_algorithm
        self.maximize = self.ga.maximize
        self.history_fitness = self.ga.history_fitness

    def eval_individual(self, chromosome):
        gen = chromosome.__repr__()
        if gen not in self.history_fitness.keys():
            self.history_fitness[gen] = chromosome.fitness()
        elif chromosome.fit is None:
            chromosome.fit = self.history_fitness[gen]
        return chromosome.fit

    def next_gen(self, population, num_offspring=1):
        next_generation = []
        all_parents = []
        for n in range(num_offspring):
            offspring, parents = self.get_one_offspring(population)
            next_generation.append(offspring)
            all_parents.append(parents)
        return next_generation, all_parents

    def rank(self, population):
        fitness_result = {}
        for i in range(len(population)):
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                self.history_fitness[gen] = population[i].fitness()
            elif population[i].fit is None:
                population[i].fit = self.history_fitness[gen]
            fitness_result[i] = self.history_fitness[gen]
        return sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=self.maximize)

    def get_one_offspring(self, population):
        raise NotImplementedError("Not implemented yet!")


class RandomParentSelector(ParentSelector):
    def get_one_offspring(self, population):
        parent1, parent2 = random.choices(population, k=2)
        counter = 10
        while parent1.equals(parent2) and counter < 10:
            parent2 = random.choice(population)
            counter += 1
        offspring = parent1.cross(parent2)
        offspring.mutate()
        self.eval_individual(parent1)
        self.eval_individual(parent2)
        self.eval_individual(offspring)
        return offspring, (parent1, parent2)


class LinealOrder(ParentSelector):
    def get_one_offspring(self, population):
        ranking = dict(self.rank(population))
        ids = list(ranking.keys())
        probes = np.linspace(len(population), 1, len(population))
        probes /= np.sum(probes)
        idx_1, idx_2 = random.choices(ids, weights=probes, k=2)
        while idx_1 == idx_2:
            idx_2 = random.choices(ids, weights=probes)[0]
        parent1 = population[idx_1]
        parent2 = population[idx_2]
        offspring = parent1.cross(parent2)
        offspring.mutate()
        self.eval_individual(offspring)
        return offspring, (parent1, parent2)


class LinealOrderII(ParentSelector):
    def __init__(self):
        super().__init__()
        self.num_parents = 1

    def set_genetic_algorithm(self, genetic_algorithm):
        super().set_genetic_algorithm(genetic_algorithm)
        self.num_parents = self.ga.num_parents

    def get_one_offspring(self, population):
        ranking = dict(self.rank(population))
        idxs = list(ranking.keys())
        positions = np.linspace(1, len(population), len(population))

        n_keep = self.num_parents
        sum_n_keep = n_keep * (n_keep + 1) / 2
        probes = (n_keep + 1 - positions) / sum_n_keep
        probes[self.num_parents:] *= 0
        idx_1, idx_2 = random.choices(idxs, weights=probes, k=2)
        while idx_1 == idx_2:
            idx_2 = random.choices(idxs, weights=probes)[0]
        parent1 = population[idx_1]
        parent2 = population[idx_2]
        offspring = parent1.cross(parent2)
        offspring.mutate()
        self.eval_individual(offspring)
        return offspring, (parent1, parent2)


class WheelSelection(ParentSelector):
    def get_one_offspring(self, population):
        ranking = dict(self.rank(population))
        idxs = list(ranking.keys())
        fitness = list(ranking.values())
        if self.maximize:
            probs = np.array(fitness) / np.sum(fitness)
        else:
            probs = 1 / (np.array(fitness) + 1e-6)
            probs /= np.sum(probs)
        idx_1, idx_2 = random.choices(idxs, weights=probs, k=2)
        counter = 0
        while idx_1 == idx_2:
            idx_2 = random.choices(idxs, weights=probs)[0]
            counter += 1
            if counter + 1 % 1000 == 0:
                print("PROBLEMS HERE!")
                print(len(idxs))
                print(probs)
        parent1 = population[idx_1]
        parent2 = population[idx_2]
        offspring = parent1.cross(parent2)
        offspring.mutate()
        self.eval_individual(offspring)
        return offspring, (parent1, parent2)


class TournamentSelection(ParentSelector):
    def __init__(self, N_participants=3, **kwards):
        super().__init__(**kwards)
        self.N = N_participants

    def get_one_offspring(self, population):
        idxs = np.linspace(0, len(population) - 1, len(population)).astype(np.int32)
        idxs_perm = np.random.permutation(idxs)
        participants_1 = [population[idxs_perm[i]] for i in range(self.N)]
        participants_2 = [population[idxs_perm[-i]] for i in range(1, self.N + 1)]
        if self.maximize:
            win_1 = np.argmax([self.eval_individual(chrom) for chrom in participants_1])
            win_2 = np.argmax([self.eval_individual(chrom) for chrom in participants_2])
        else:
            win_1 = np.argmin([self.eval_individual(chrom) for chrom in participants_1])
            win_2 = np.argmin([self.eval_individual(chrom) for chrom in participants_2])
        parent1 = participants_1[win_1]
        parent2 = participants_2[win_2]
        offspring = parent1.cross(parent2)
        offspring.mutate()
        self.eval_individual(offspring)
        return offspring, (parent1, parent2)
