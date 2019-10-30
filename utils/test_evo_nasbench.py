import os
import random

import numpy as np
import sys
import matplotlib.pyplot as plt

from utils.codifications import Chromosome, Fitness

sys.path.append('../')
from utils.codification_NASBench import ChromosomeNASBench, FitnessNASBench
from GA.geneticAlgorithm import TwoLevelGA


class GeneticEvalOnNASBecnh:
    ALLOWED_EPOCHS = [4, 12, 36, 108]

    def __init__(self, records_path='../../nasbench/nasbench_full.tfrecord'):
        self.experiments_folder = '../exps_testing'
        os.makedirs(self.experiments_folder, exist_ok=True)
        self.fitness_cnn = FitnessNASBench(records_path=records_path)
        exp_folder = os.path.join(self.experiments_folder, 'nas')
        self.folder = os.path.join(exp_folder, 'genetic')
        os.makedirs(self.folder, exist_ok=True)

    def eval_genetic(self, epoch1, epoch2, generations, pop1, pop2, freq, start, reps=100, stat_val=False,
                     op_mutate=0.1, grow_mutate=0.1, dec_prob=0.1, conn_prob=0.15):

        assert epoch1 in self.ALLOWED_EPOCHS
        assert epoch2 in self.ALLOWED_EPOCHS

        ChromosomeNASBench._change_op_prob = op_mutate
        ChromosomeNASBench._grow_prob = grow_mutate
        ChromosomeNASBench._decrease_prob = dec_prob
        ChromosomeNASBench._connection_prob = conn_prob

        print("\nEVOLVING NAS-BENCH 101...\n")

        c = ChromosomeNASBench.random_individual()
        self.fitness_cnn.set_params(epochs=epoch1, precise_epochs=epoch2)
        generational = TwoLevelGA(chromosome=c,
                                  fitness=self.fitness_cnn,
                                  generations=generations,
                                  population_first_level=pop1,
                                  population_second_level=pop2,
                                  training_hours=1,
                                  save_progress=False,
                                  maximize_fitness=False,
                                  statistical_validation=stat_val,
                                  start_level2=start,
                                  folder=self.folder,
                                  frequency_second_level=freq,
                                  perform_evo=False)
        t, r, curves = [], [], []
        for i in range(reps):
            if i == 0:
                print("Evolution progress: [", end='')
            elif i == reps - 1:
                print("]. Done!")
            elif i % (reps // 20) == 0:
                print("=", end='')
            generational.reset()
            self.fitness_cnn.reset()
            winner, best_fit, perform_evo = generational.evolve(show=False)
            t.append(self.fitness_cnn.total_training_time)
            r.append(self.fitness_cnn.get_test_regret(winner))
            curves.append(perform_evo)
        print("Regret %0.5f. Time: %0.2f:" % (float(np.mean(r)), float(np.mean(t))))
        return r, t, curves, generational

    def plot_performance_curve(self, curves):
        curves = self.fitness_cnn.best_mean_acc + np.mean(np.array(curves), axis=0) - 1
        plt.plot(curves)
        plt.xlabel("Generations")
        plt.ylabel("Test accuracy")
        plt.show()

    def eval_list(self, **kwargs):
        values_list_keys = []
        n_values = 0
        for k, val in kwargs.items():
            if isinstance(val, list):
                values_list_keys.append(k)
                n_values = len(val)

        for k in values_list_keys:
            assert len(kwargs[k]) == n_values

        r_means, r_stds, t_means, t_stds, curves = [], [], [], [], []
        new_kwargs = kwargs.copy()
        for i in range(n_values):
            for k in values_list_keys:
                new_kwargs[k] = kwargs[k][i]
            r, t, curv, gen = self.eval_genetic(**new_kwargs)
            r_means.append(np.mean(r))
            r_stds.append(np.std(r))
            t_means.append(np.mean(t))
            t_stds.append(np.std(t))
            curves.append(np.mean(np.array(curves), axis=0))
        return r_means, r_stds, t_means, t_stds, curves

    def pop_size_sensitization(self, epoch1, generations, pop2_frac, freq, reps=500,
                               op_mutate=0.1, grow_mutate=0.1, dec_prob=0.1, conn_prob=0.15):
        pop1 = list(np.arange(20, 110, 10))
        pop2 = [int(p * pop2_frac) for p in pop1]
        kwargs = {'epoch1': epoch1,
                  'epoch2': 108,
                  'generations': generations,
                  'pop1': pop1,
                  'pop2': pop2,
                  'freq': freq,
                  'start': freq - 1,
                  'reps': reps,
                  'stat_val': False,
                  'op_mutate': op_mutate,
                  'grow_mutate': grow_mutate,
                  'dec_prob': dec_prob,
                  'conn_prob': conn_prob}
        return self.eval_list(**kwargs)

    def plot_over_figure(self, r_means, t_means):
        import cv2
        p1 = (145, 77)
        p2 = (314, 77)
        p3 = (314, 609)

        p1 = (322, 174)
        p2 = (684, 174)
        p3 = (684, 1322)
        im = cv2.imread('../../plot.png')
        im = cv2.imread('../../plot2.png')

        def get_y(y):
            h = p2[0] - p1[0]
            y2 = p1[0] - h * (np.log(y) / np.log(10) + 2)
            return int(y2)

        def get_x(x):
            h = p3[1] - p1[1]
            x2 = p1[1] + ((np.log(x) / np.log(10) - 1) / 6) * h
            return int(x2)

        for i in range(len(r_means)):
            x = get_x(t_means[i])
            y = get_y(r_means[i])
            im = cv2.rectangle(im, (y-2, x-2), (y+2, x+2), (255, 0, 0), 4)

        for i in range(1, len(r_means)):
            x0 = get_x(t_means[i-1])
            y0 = get_y(r_means[i-1])
            x1 = get_x(t_means[i])
            y1 = get_y(r_means[i])
            im = cv2.line(im, (x0, y0), (x1, y1), (255, 0, 0), 2)

        plt.imshow(im)
        plt.axis('off')
        plt.savefig("test.png", bbox_inches='tight')
        plt.show()
        return im

    def freq_sensitization(self, epoch1, generations, pop1, pop2_frac, reps=500,
                           op_mutate=0.1, grow_mutate=0.1, dec_prob=0.1, conn_prob=0.15):
        pop2 = int(pop1 * pop2_frac)
        freqs = [1, 2, 4, 5, 10, 20]
        starts = [f - 1 for f in freqs]
        kwargs = {'epoch1': epoch1,
                  'epoch2': 108,
                  'generations': generations,
                  'pop1': pop1,
                  'pop2': pop2,
                  'freq': freqs,
                  'start': starts,
                  'reps': reps,
                  'stat_val': False,
                  'op_mutate': op_mutate,
                  'grow_mutate': grow_mutate,
                  'dec_prob': dec_prob,
                  'conn_prob': conn_prob}
        return self.eval_list(**kwargs)

    def epochs_sensitization(self, generations, pop1, pop2_frac, freq, reps=100,
                             op_mutate=0.1, grow_mutate=0.1, dec_prob=0.1, conn_prob=0.15):
        pop2 = int(pop1 * pop2_frac)
        kwargs = {'epoch1': self.ALLOWED_EPOCHS,
                  'epoch2': 108,
                  'generations': generations,
                  'pop1': pop1,
                  'pop2': pop2,
                  'freq': freq,
                  'start': freq - 1,
                  'reps': reps,
                  'stat_val': False,
                  'op_mutate': op_mutate,
                  'grow_mutate': grow_mutate,
                  'dec_prob': dec_prob,
                  'conn_prob': conn_prob}
        return self.eval_list(**kwargs)

    def gen_sensitization(self, epoch1, pop1, pop2_frac, freq, reps=100,
                           op_mutate=0.1, grow_mutate=0.1, dec_prob=0.1, conn_prob=0.15):
        pop2 = int(pop1 * pop2_frac)
        kwargs = {'epoch1': epoch1,
                  'epoch2': 108,
                  'generations': [20, 30, 40, 50, 60, 70, 80, 90, 100],
                  'pop1': pop1,
                  'pop2': pop2,
                  'freq': freq,
                  'start': freq - 1,
                  'reps': reps,
                  'stat_val': False,
                  'op_mutate': op_mutate,
                  'grow_mutate': grow_mutate,
                  'dec_prob': dec_prob,
                  'conn_prob': conn_prob}
        return self.eval_list(**kwargs)

    def figure0(self, r_means, r_stds, t_means, t_stds, errorbar=True):
        x = np.arange(len(r_means))
        fig, ax1 = plt.subplots()
        color_ = 'tab:red'
        ax1.set_xlabel('population in second level')
        ax1.set_ylabel('test error rate', color=color_)
        if errorbar:
            ax1.errorbar(x + 0.2, r_means, yerr=r_stds, color=color_)
        else:
            # e = 1 - self.fitness_cnn.best_mean_acc + np.array(r_means)
            ax1.plot(x, r_means, color=color_)
        ax1.tick_params(axis='y', labelcolor=color_)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('time', color=color)  # we already handled the x-label with ax1
        if errorbar:
            ax2.errorbar(x - 0.2, t_means, yerr=np.array(t_stds), color=color)
        else:
            ax2.plot(x, t_means, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Time and test error when increasing second level population")
        plt.savefig("figura0")
        plt.show()

    def figure1(self, r_means3, t_means3):
        plt.scatter(np.array(t_means3), r_means3)
        plt.xlabel("log(Time) [seconds]")
        plt.ylabel("Test Regret")
        plt.title("")
        plt.ylim([10 ** -3, 10 ** -1])
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([10 ** 1, 10 ** 7])
        plt.show()

    @staticmethod
    def figure2(r, t, labels):
        for i in range(len(r)):
            plt.plot(t[i], r[i], label=labels[i])
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('time [seconds]')
        plt.ylabel('Test Regret')
        plt.show()


class ChromosomeGA(Chromosome):
    FIRST_LEVEL_EPOCHS = [4, 12, 36]
    GENERATIONS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    POPULATION_SIZE_LEVEL1 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    POPULATION_SIZE_LEVEL2 = [0.2, 0.3, 0.4, 0.5, 0.6]
    FREQ = [1, 2, 4, 5, 10, 20]
    MAX_MUTATION_PROB = 0.5

    MPROB = 0.05

    def __init__(self, epoch1, generations, pop1, pop2, freq, op_mutate, grow_mutate, dec_prob, conn_prob):
        super().__init__()
        self.epoch1 = epoch1
        self.generations = generations
        self.pop1 = pop1
        self.pop2 = pop2
        self.freq = freq
        self.mutation_probs = {'operation': op_mutate,
                               'growth': grow_mutate,
                               'decay': dec_prob,
                               'connection': conn_prob}

    @classmethod
    def random_individual(cls):
        epoch1 = random.choice(cls.FIRST_LEVEL_EPOCHS)
        generations = random.choice(cls.GENERATIONS)
        pop1 = random.choice(cls.POPULATION_SIZE_LEVEL1)
        pop2 = random.choice(cls.POPULATION_SIZE_LEVEL2)
        freq = random.choice(cls.FREQ)
        op_mutate = cls.truncate(cls.MAX_MUTATION_PROB * np.random.rand())
        grow_mutate = cls.truncate(cls.MAX_MUTATION_PROB * np.random.rand())
        dec_prob = cls.truncate(cls.MAX_MUTATION_PROB * np.random.rand())
        conn_prob = cls.truncate(cls.MAX_MUTATION_PROB * np.random.rand())
        return ChromosomeGA(epoch1, generations, pop1, pop2, freq, op_mutate, grow_mutate, dec_prob, conn_prob)

    def simple_individual(self):
        return ChromosomeGA(4, 20, 20, 0.1, 1, 0, 0, 0, 0)

    @staticmethod
    def truncate(value, decimals=2):
        return value - value % 10**(-decimals)

    def cross(self, other_chromosome):
        epoch1 = self.cross_integer(self.epoch1, other_chromosome.epoch1, self.FIRST_LEVEL_EPOCHS)
        generations = self.cross_integer(self.generations, other_chromosome.generations, self.GENERATIONS)
        pop1 = self.cross_integer(self.pop1, other_chromosome.pop1, self.POPULATION_SIZE_LEVEL1)
        pop2 = self.cross_integer(self.pop2, other_chromosome.pop2, self.POPULATION_SIZE_LEVEL2)
        freq = self.cross_integer(self.freq, other_chromosome.freq, self.FREQ)
        op_mutate = self.cross_real(self.mutation_probs['operation'], other_chromosome.mutation_probs['operation'])
        grow_mutate = self.cross_real(self.mutation_probs['growth'], other_chromosome.mutation_probs['growth'])
        dec_prob = self.cross_real(self.mutation_probs['decay'], other_chromosome.mutation_probs['decay'])
        conn_prob = self.cross_real(self.mutation_probs['connection'], other_chromosome.mutation_probs['connection'])
        return ChromosomeGA(epoch1, generations, pop1, pop2, freq, op_mutate, grow_mutate, dec_prob, conn_prob)

    def cross_integer(self, value1, value2, values_list):
        if value1 == value2:
            return value2
        idx1 = values_list.index(value1)
        idx2 = values_list.index(value2)
        new_id = np.random.randint(min(idx1, idx2), max(idx1, idx2) + 1)
        return values_list[new_id]

    def cross_real(self, value1, value2):
        b = np.random.rand()
        return self.truncate(value1 * b + (1 - b) * value2)

    def mutate(self):
        if np.random.rand() < self.MPROB:
            self.epoch1 = random.choice(self.FIRST_LEVEL_EPOCHS)
        if np.random.rand() < self.MPROB:
            self.generations = random.choice(self.GENERATIONS)
        if np.random.rand() < self.MPROB:
            self.pop1 = random.choice(self.POPULATION_SIZE_LEVEL1)
        if np.random.rand() < self.MPROB:
            self.pop2 = random.choice(self.POPULATION_SIZE_LEVEL2)
        if np.random.rand() < self.MPROB:
            self.freq = random.choice(self.FREQ)
        for key in self.mutation_probs.keys():
            if np.random.rand() < self.MPROB:
                self.mutation_probs[key] = self.truncate(self.MAX_MUTATION_PROB * np.random.rand())

    def __repr__(self):
        values = [self.epoch1, self.generations, self.pop1, self.pop2, self.freq] + list(self.mutation_probs.values())
        r = "EP:%d|GEN:%d|POP1:%d|POP2:%0.1f|FREQ:%d|OPm:%0.2f|GROWm:%0.2f|DECm:%0.2f|CONNm:%0.2f" % tuple(values)
        return r

    def self_copy(self):
        op_mutate = self.mutation_probs['operation']
        grow_mutate = self.mutation_probs['growth']
        dec_prob = self.mutation_probs['decay']
        conn_prob = self.mutation_probs['connection']
        return ChromosomeGA(self.epoch1, self.generations, self.pop1, self.pop2, self.freq,
                            op_mutate, grow_mutate, dec_prob, conn_prob)

    def decode(self, **kwargs):
        pass


class FitnessGA(Fitness):
    SECOND_LEVEL_EPOCHS = 108

    def calc(self, chromosome, test=False, precise_mode=False):
        start = chromosome.freq - 1
        reps = 300 if precise_mode else 100
        reps = 500 if test else reps
        pop2 = max(4, int(chromosome.pop1 * chromosome.pop2))
        op_mutate = chromosome.mutation_probs['operation']
        grow_mutate = chromosome.mutation_probs['growth']
        dec_prob = chromosome.mutation_probs['decay']
        conn_prob = chromosome.mutation_probs['connection']
        r, t, curves, generational = self.fitness.eval_genetic(chromosome.epoch1, self.SECOND_LEVEL_EPOCHS,
                                                               chromosome.generations,
                                                               chromosome.pop1, pop2,
                                                               chromosome.freq, start,
                                                               reps, False,
                                                               op_mutate,
                                                               grow_mutate,
                                                               dec_prob,
                                                               conn_prob)
        fit = np.log(np.mean(r))/np.log(10) + np.log(np.mean(t)) / (np.log(10) * 5)

        return fit

    def set_params(self, **kwargs):
        pass

    def __init__(self):
        super().__init__()
        self.fitness = GeneticEvalOnNASBecnh()


def evolve_GA(generations=30, pop1=20, pop2=5, hours=1):
    save_progress = True
    maximize_fitness = False
    statistical_validation = True
    fitness = FitnessGA()
    chromosome = ChromosomeGA.random_individual()

    generational = TwoLevelGA(chromosome=chromosome,
                              fitness=fitness,
                              generations=generations,
                              population_first_level=pop1,
                              population_second_level=pop2,
                              training_hours=hours,
                              save_progress=save_progress,
                              maximize_fitness=maximize_fitness,
                              statistical_validation=statistical_validation,
                              start_level2=3,
                              folder=fitness.fitness.folder,
                              frequency_second_level=2,
                              perform_evo=False)

    try:
        winner, best_fit, perform_evo = generational.evolve(show=False)
        print("BEST FITNESS")
        print(winner)
    except:
        return None, None, None, None, generational
    return winner, best_fit, perform_evo, fitness, generational

# BEST: EP:36|GEN:50|POP1:70|POP2:0.6|FREQ:1|OPm:0.44|GROWm:0.21|DECm:0.26|CONNm:0.11
# generations elapsed: 6
# Time: 10 hours
