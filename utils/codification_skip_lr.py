import keras

import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, PReLU, LeakyReLU, Dropout,SpatialDropout2D
from keras.layers import MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import os
from utils.BN16 import BatchNormalizationF16
from utils.codification_skipc import ChromosomeSkip, FitnessSkip
from utils.codifications import Layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ChromosomeSkipLr(ChromosomeSkip):
    max_lr = 1.0
    min_lr = 0.0001
    lr_prob = 0.1

    def __init__(self, cnn_layers=None, nn_layers=None, connections=None,lr_generator=None,fitness=None):
        super().__init__(cnn_layers=cnn_layers, nn_layers=nn_layers, connections=connections, fitness=fitness)
        self.lr_generator = lr_generator
        self.lr = self.decode_lr(self.lr_generator)
        
    @staticmethod
    def decode_lr(lr_gen):
        max_val = ChromosomeSkipLr.max_lr
        min_val = ChromosomeSkipLr.min_lr
        return (np.exp(2 * (lr_gen)) - 1) * ((max_val - min_val) / (np.exp(2) - 1)) + min_val

    @staticmethod
    def random_individual():
        lr_gen = np.random.rand()
        chromosome_skip = ChromosomeSkip.random_individual()
        return ChromosomeSkipLr(cnn_layers=chromosome_skip.cnn_layers,
                              nn_layers=chromosome_skip.nn_layers,
                              connections=chromosome_skip.connections,
                              lr_generator=lr_gen)

    def simple_individual(self):
        lr_gen = 0.5
        return ChromosomeSkipLr(cnn_layers=chromosome_skip.cnn_layers,
                              nn_layers=chromosome_skip.nn_layers,
                              connections=chromosome_skip.connections,
                              lr_generator=lr_gen)

    def cross(self, other_chromosome):
        new_chromosome = super().cross(other_chromosome)
        new_lr_gen = self.cross_lr_gen(other_chromosome.lr_generator, self.lr_generator)
        return ChromosomeSkipLr(cnn_layers=new_chromosome.cnn_layers,
                              nn_layers=new_chromosome.nn_layers,
                              connections=new_chromosome.connections,
                              lr_generator=new_lr_gen)
    
    @staticmethod
    def cross_lr_gen(lr_gen_1, lr_gen_2):
        b = np.random.rand()
        return lr_gen_1 * b + (1 - b) * lr_gen_2

    def mutate(self):
        super().mutate()
        if np.random.rand() < self.lr_prob:
            self.mutate_lr()
            
    def mutate_lr(self):
        self.lr_generator = Layer.gauss_mutation(self.lr_generator, 1, 0, int_=False)
        self.lr = self.decode_lr(self.lr_generator)

    def __repr__(self):
        return super().__repr__() + "LR:%0.5f\n"%self.lr

    def fitness(self, test=False):
        return self.evaluator.calc(self, test=test)

    def self_copy(self):
        new_chromosome = super().self_copy()
        return ChromosomeSkipLr(cnn_layers=new_chromosome.cnn_layers,
                              nn_layers=new_chromosome.nn_layers,
                              connections=new_chromosome.connections,
                              lr_generator=self.lr_generator)


class FitnessSkipLr(FitnessSkip):
    
    def calc(self, chromosome, test=False, file_model='./model_acc.hdf5', fp=32, precise_mode=False):
        self.learning_rate_base = chromosome.lr
        return super().calc(chromosome, test, file_model, fp, precise_mode)
    
    

