import random
import operator
import numpy as np


class ParentSelector(object):

    def __init__(self, maximize=True, history={}, model_filter=None):
        self.maximize = maximize
        self.history_fitness = history
        self.model_filter = model_filter
        self.n_filtered = 0

    def set_params(self, maximize, history_fitness, **kwargs):
        self.maximize = maximize
        self.history_fitness = history_fitness

    def eval_individual(self, chromosome):
        gen = chromosome.__repr__()
        if gen not in self.history_fitness.keys():
            self.history_fitness[gen] = chromosome.fitness()
        return self.history_fitness[gen]

    def next_gen(self, population, rank, num_offspring=1):
        next_generation = []
        all_parents = []
        for n in range(num_offspring):
            offspring, parents = self.get_one_filtered_offspring(population, rank, show_probs=n == -1)
            next_generation.append(offspring)
            all_parents.append(parents)
        return next_generation, all_parents

    def rank_(self, population):
        fitness_result = {}
        for i in range(len(population)):
            gen = population[i].__repr__()
            if gen not in self.history_fitness.keys():
                self.history_fitness[gen] = population[i].fitness()
            elif population[i].fit is None:
                population[i].fit = self.history_fitness[gen]
            fitness_result[i] = self.history_fitness[gen]
        return sorted(fitness_result.items(), key=operator.itemgetter(1), reverse=self.maximize)

    def get_one_filtered_offspring(self, population, rank, show_probs=False):
        offspring, (parent1, parent2) = self.get_one_offspring(population, rank, show_probs)
        if hasattr(self, 'model_filter') and self.model_filter is not None:
            n_trials = 1
            while not self.model_filter.is_model_ok(offspring):
                offspring, (parent1, parent2) = self.get_one_offspring(population, rank, show_probs)
                n_trials += 1
                if n_trials % 100 == 0:
                    print("\nTrying to find a good model candidate. Trial number %d" % n_trials)
            print("Filtered now: %d" % (n_trials - 1))
            self.n_filtered += n_trials - 1
            print("Total filtered until now: %d" % self.n_filtered)
        return offspring, (parent1, parent2)

    def get_one_offspring(self, population, rank, show_probs=False):
        raise NotImplementedError("Not implemented yet!")


class RandomParentSelector(ParentSelector):
    def get_one_offspring(self, population, rank, show_probs=False):
        parent1, parent2 = random.choices(population, k=2)
        counter = 10
        while parent1.equals(parent2) and counter < 10:
            parent2 = random.choice(population)
            counter += 1
        offspring = parent1.cross(parent2)
        offspring.mutate()
        return offspring, (parent1, parent2)


class LinealOrder(ParentSelector):
    def get_one_offspring(self, population, rank, show_probs=False):
        ranking = dict(rank)
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
        return offspring, (parent1, parent2)


class LinealOrderII(ParentSelector):
    def __init__(self):
        super().__init__()
        self.num_parents = 1

    def set_params(self, num_parents, **kwargs):
        super().set_params(**kwargs)
        self.num_parents = num_parents

    def get_one_offspring(self, population, rank, show_probs=False):
        ranking = dict(rank)
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
        return offspring, (parent1, parent2)


class WheelSelection(ParentSelector):
    def get_one_offspring(self, population, rank, show_probs=False):
        ranking = dict(rank)
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
        if show_probs:
            for i in range(len(population)):
                print("Fit: %0.4f. Prob: %0.4f" % (fitness[i], probs[i]))
                print(population[idxs[i]])
                #print("[" ,end='')
                #[print("%0.3f" % f, end=', ') for f in probs]
                #print("] -> %0.3f, %0.3f" % (probs[idx_1], probs[idx_2]))
        parent1 = population[idx_1]
        parent2 = population[idx_2]
        offspring = parent1.cross(parent2)
        offspring.mutate()
        return offspring, (parent1, parent2)


class TournamentSelection(ParentSelector):
    def __init__(self, N_participants=3, **kwards):
        super().__init__(**kwards)
        self.N = N_participants

    def get_one_offspring(self, population, rank, show_probs=False):
        ranking = dict(rank)
        idxs = list(ranking.keys())
        fitness = list(ranking.values())

        # the indices are shuffled
        id_perm = np.linspace(0, len(population) - 1, len(population)).astype(np.int32)
        id_perm = np.random.permutation(id_perm)

        # the initial N indices are selected for participants of the 1st tournament
        participants_1 = [population[idxs[id_perm[i]]] for i in range(self.N)]

        # the indices are shuffled again
        id_perm = np.random.permutation(id_perm)

        # the final N indices are selected for participants of the 2nd tournament, this make
        # sense when the indices are shuffled only once
        participants_2 = [population[idxs[id_perm[-i]]] for i in range(1, self.N + 1)]
        fitness_1 = [fitness[id_perm[i]] for i in range(self.N)]
        fitness_2 = [fitness[id_perm[-i]] for i in range(1, self.N + 1)]
        if self.maximize:
            win_1 = np.argmax(fitness_1)
            win_2 = np.argmax(fitness_2)
            #win_1 = np.argmax([self.eval_individual(chrom) for chrom in participants_1])
            #win_2 = np.argmax([self.eval_individual(chrom) for chrom in participants_2])
        else:
            win_1 = np.argmin(fitness_1)
            win_2 = np.argmin(fitness_2)
            #win_1 = np.argmin([self.eval_individual(chrom) for chrom in participants_1])
            #win_2 = np.argmin([self.eval_individual(chrom) for chrom in participants_2])
        parent1 = participants_1[int(win_1)]
        parent2 = participants_2[int(win_2)]
        offspring = parent1.cross(parent2)
        offspring.mutate()
        return offspring, (parent1, parent2)
