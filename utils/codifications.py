from time import time
import numpy as np
import random
import pickle


class Layer(object):

    def cross(self, other_layer):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def compare(self, other_layer):
        raise self.__repr__() == other_layer.__repr__()

    def self_copy(self):
        raise NotImplementedError

    @classmethod
    def random_layer(cls):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    @classmethod
    def create(cls, **kwargs):
        # returns an object of the same class as the one who created it,
        # then if someone inherits this class, the returned object will
        # be from the heir's class
        return cls(**kwargs)
    
    @staticmethod
    def gauss_mutation(val, max_val, min_val, int_=True):
        m = 0
        s = (max_val - min_val) / 10.
        new_val = val + random.gauss(m, s)
        if int_:
            new_val = int(new_val)
        if new_val < min_val:
            new_val = 2 * min_val - new_val
        elif new_val > max_val:
            new_val = max_val - (new_val - max_val)
        if new_val > max_val or new_val < min_val:
            new_val = Layer.gauss_mutation(val, max_val, min_val, int_)
        return new_val


class Chromosome(object):

    def __init__(self, fitness):
        self.evaluator = fitness
        self.age = 0

    def increase_age(self):
        self.age += 1

    def set_fitness(self, fit):
        self.evaluator = fit

    @staticmethod
    def random_individual():
        raise NotImplementedError

    def simple_individual(self):
        raise NotImplementedError

    def cross(self, other_chromosome):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def equals(self, other_chromosome):
        return self.__repr__() == other_chromosome.__repr__()

    def __repr__(self):
        raise NotImplementedError

    def fitness(self, test=False):
        raise NotImplementedError

    def self_copy(self):
        raise NotImplementedError

class Fitness:

    def eval_list(self, chromosome_list, test=False, **kwargs):
        return[self.calc(c, test=test, **kwargs) for c in chromosome_list]

    def calc(self, chromosome, test=False):
        raise NotImplementedError

    def set_params(self, **kwargs):
        raise NotImplementedError

    def calc_mean(self, chromosome, iter=5):
        f = []
        ti = time()
        for i in range(iter):
            f.append(self.calc(chromosome))
        print("Acc: %0.3f" % np.mean(f), np.std(f), np.max(f))
        print("Time elapsed: %0.3f" % (time() - ti))

    def save(self, filename):
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    @staticmethod
    def load(file):
        infile = open(file, 'rb')
        fit = pickle.load(infile)
        infile.close()
        return fit
