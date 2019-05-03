from __future__ import print_function
import keras
from keras.models import  Model
from keras.layers import Dense, Dropout, Input, Flatten, PReLU, LeakyReLU
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time as T
import traceback
import logging
import random
import numpy as np
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, units=128, activation='relu', dropout=0):
        self.units = units
        #self.posible_Activations = ['relu', 'elu', 'prelu', 'leakyrelu']
        self.posible_activations = ['relu', 'sigmoid', 'tanh', 'elu', 'prelu', 'leakyreLu']
        assert activation in self.posible_activations
        self.activation = activation
        self.dropout = dropout
        self.units_lim = 1024
        self.units_prob = 0.2
        self.act_prob = 0.2
        self.drop_prob = 0.2

    def cross(self, other_layer):
        new_units = self.cross_units(other_layer.units)
        new_activation = self.cross_activation(other_layer.activation)
        new_dropout = self.cross_dropout(other_layer.dropout)
        return Layer(new_units, new_activation, new_dropout)

    def cross_activation(self, other_activation):
        if np.random.rand() > 0.5:
            return self.activation
        return other_activation

    def cross_dropout(self, other_dropout):
        b = np.random.rand()
        return self.dropout * (1 - b) + b * other_dropout

    def cross_units(self, other_units):
        b = np.random.rand()
        return int(self.units * (1 - b) + other_units * b)

    def mutate(self):
        aleatory = np.random.rand(4)
        if aleatory[0] < self.units_prob:
            self.units = np.random.randint(0, self.units_lim)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.posible_activations)
        if aleatory[2] < self.drop_prob:
            self.dropout = np.random.rand()

    def compare(self, other_layer):
        if self.units != other_layer.units:
            return False
        if self.activation != other_layer.activation:
            return False
        if self.dropout != other_layer.dropout:
            return False
        return True

    def self_copy(self):
        return Layer(self.units, self.activation, self.dropout)

    def random_layer(self):
        units = np.random.randint(0, self.units_lim)
        act = random.choice(self.posible_activations)
        drop = np.random.rand()
        return Layer(units, act, drop)

    def __repr__(self):
        return "U:%d|A:%s|D:%0.3f" % (self.units, self.activation, self.dropout)


class Cromosome(object):

    def __init__(self, layers=[], fit=None):
        assert type(layers) == list
        self.n_layers = len(layers)
        self.layers = layers
        self.max_layers = 10
        self.layer_prob = 0.1
        self.fit = None
        self.evaluator = Fitness.get_instance()

    def set_fitness(self, fit):
        self.evaluator = fit

    def random_indiv(self):
        n_layers = np.random.randint(0, self.max_layers)
        layers = [Layer().random_layer() for i in range(n_layers)]
        return Cromosome(layers)

    @staticmethod
    def simple_indiv():
        return Cromosome([Layer()])

    def cross(self, other_cromosome):
        new_layers = []

        if self.n_layers == 0:
            return other_cromosome

        n_intersection = np.random.randint(0, self.n_layers)
        for i in range(self.n_layers):
            if i < n_intersection or i >= other_cromosome.n_layers:
                new_layers.append(self.layers[i].self_copy())
            else:
                try:
                    new_layers.append(self.layers[i].cross(other_cromosome.layers[i - n_intersection]))
                except IndexError:
                    print("Problem with index %d" % i)
                    print("Intersection point at %d" % n_intersection)
                    print(len(self.layers), self.layers)
                    print(len(other_cromosome.layers), other_cromosome.layers)
                    print(len(new_layers), new_layers)
                    raise IndexError
        return Cromosome(new_layers)

    def mutate(self):
        for i in range(self.n_layers):
            self.layers[i].mutate()
        if np.random.rand() < self.layer_prob and len(self.layers) < self.max_layers:
            self.layers.append(Layer().random_layer())
        elif np.random.rand() < self.layer_prob and len(self.layers) > 0:
            self.layers.pop()
        self.n_layers = len(self.layers)
            

    def equals(self, other_cromosome):
        if self.n_layers != other_cromosome.n_layers:
            return False
        for i in range(self.n_layers):
            if not self.layers[i].compare(other_cromosome.layers[i]):
                return False
        return True
    
    def self_copy(self):
        new_layers = []
        for layer in self.layers:
            new_layers.append(layer.self_copy())
        return Cromosome(new_layers)

    def __repr__(self):
        rep = ""
        for i in range(self.n_layers):
            rep += "%d - %s \n" % (i, self.layers[i])
        return rep

    def cross_val(self, exclude_first=True, test=False):
        return self.evaluator.cross_val(self, exclude_first, test=test)

    def fitness(self, test=False):
        return self.evaluator.calc(self, test=test)

class Fitness:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Fitness.__instance == None:
            Fitness()
        return Fitness.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Fitness.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Fitness.__instance = self

    def set_params(self, data, batch_size=128, epochs=100, early_stop=True, reduce_plateau=True, 
                   verbose=1, input_shape=(28,28,1)):
        self.time = 0
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.reduce_plateu = reduce_plateau
        self.verb = verbose
        (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = data
        self.num_clases = self.y_train.shape[1]
        self.callbacks = []
        self.input_shape = input_shape
        #self.callbacks = [EarlyStopping(monitor='val_acc', patience=3,baseline=(1. /self.num_clases)*1.2)]                                         
        if self.early_stop and keras.__version__=='2.2.4':
            self.callbacks.append(EarlyStopping(monitor='val_acc', patience=50, restore_best_weights=True))
        elif self.early_stop:
            self.callbacks.append(EarlyStopping(monitor='val_acc', patience=50))
        if self.reduce_plateu:
            self.callbacks.append(ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                                    patience=5, verbose=self.verb))
        return self 

    def cross_val(self, chromosome, exclude_first=True, test=False):
        folds = int((self.x_val.shape[0] + self.x_train.shape[0]) / self.x_val.shape[0])
        X = np.concatenate([self.x_val, self.x_train], axis=0)
        Y = np.concatenate([self.y_val, self.y_train], axis=0)
        N = X.shape[0]
        n = int(N/folds) # number of elements of each fold
        score = []
        for i in range(exclude_first, folds):
            self.x_val = X[n * i : n * (i + 1),...]
            self.y_val = Y[n * i : n * (i + 1),...]
            self.x_train = np.concatenate([X[:n * i,...], X[n * (i + 1):,...]], axis=0)
            self.y_train = np.concatenate([Y[:n * i,...], Y[n * (i + 1):,...]], axis=0)
            score.append(self.calc(chromosome, test=test))
        self.x_val, self.y_val = X[:n,...], Y[:n,...]
        self.x_train, self.y_train = X[n:,...], Y[n:,...]
        return score

    def calc(self, chromosome, test=False):
        try:
            ti = T.time()
            keras.backend.clear_session()
            model = self.decode(chromosome)
            h = model.fit(self.x_train, self.y_train,
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              verbose=self.verb,
                              validation_data=(self.x_val, self.y_val),
                              callbacks=self.callbacks)
            if test:
                score = model.evaluate(self.x_test, self.y_test, verbose=0)
            else:
                score = model.evaluate(self.x_val, self.y_val, verbose=0)
        except Exception as e:
            score = [0,0]
            print("Some Error with gen:")
            print(chromosome)
            logging.error(traceback.format_exc())
            keras.backend.clear_session()
            time.sleep(5)
        if self.verb and score[0] > 0:
            dataset = ['Val', 'Test'][test]
            print('%s loss: %0.4f,%s acc: %0.4f' % (dataset, score[0], dataset, score[1]))
            self.show_result(h, 'acc')
            self.show_result(h, 'loss')
        self.time += T.time() - ti
        return score[1]

    def decode(self, chromosome):

        inp = Input(shape=self.input_shape)
        x = Flatten()(inp)
        for i in range(chromosome.n_layers):
            act = chromosome.layers[i].activation
            if act in ['relu', 'sigmoid', 'tanh', 'elu']:
                x = Dense(chromosome.layers[i].units, activation=act)(x)
            elif act == 'prelu':
                x = Dense(chromosome.layers[i].units)(x)
                x = PReLU()(x)
            else:
                x = Dense(chromosome.layers[i].units)(x)
                x = LeakyReLU()(x)
            x = Dropout(chromosome.layers[i].dropout)(x)
        x = Dense(self.num_clases, activation='softmax')(x)

        model = Model(inputs=inp, outputs=x)
        if self.verb:
            model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
                      #options = self.run_opts)
        return model

    def show_result(self, history, metric='acc'):
        epochs = np.linspace(0, len(history.history['acc']) - 1, len(history.history['acc']))
        plt.plot(epochs, history.history['val_%s' % metric], label='validation')
        plt.plot(epochs, history.history[metric], label='train')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.show()

    def calc_mean(self, chromosome, iters=5):
        f = []
        ti = T.time()
        for i in range(iters):
            f.append(self.calc(chromosome))
        print("Acc: %0.3f" % np.mean(f), np.std(f), np.max(f))
        print("Time elapsed: %0.3f" % (T.time() - ti))

    def fitness_N_models(self, c1, c2):
        def decode_C(chromosome, inp, name=''):
            x = Flatten()(inp)
            for i in range(chromosome.n_layers):
                act = chromosome.layers[i].activation
                if act in ['relu', 'sigmoid', 'tanh', 'elu']:
                    x = Dense(chromosome.layers[i].units, activation=act)(x)
                elif act == 'prelu':
                    x = Dense(chromosome.layers[i].units)(x)
                    x = PReLU()(x)
                else:
                    x = Dense(chromosome.layers[i].units)(x)
                    x = LeakyReLU()(x)
                x = Dropout(chromosome.layers[i].dropout)(x)
            x = Dense(self.num_clases, activation='softmax', name=name)(x)
            return x

        inputs = Input(shape=self.input_shape)
        model = Model(inputs=inputs, outputs=[decode_C(c1, inputs,'x1'), decode_C(c2, inputs,'x2')],
                        name="all_net")
        losses = {'x1':"categorical_crossentropy", 'x2':"categorical_crossentropy"}
        metrics = {'x1': 'accuracy',
                   'x2': 'accuracy'}
        model.compile(optimizer=Adam(), loss=losses, metrics=metrics)
        model.summary()
        model.fit(self.x_train, [self.y_train, self.y_train],
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=self.verb,
                  validation_data=(self.x_val, [self.y_val, self.y_val]))
                  #callbacks=self.callbacks)
        score = model.evaluate(self.x_val, [self.y_val, self.y_val], verbose=1)
        return score