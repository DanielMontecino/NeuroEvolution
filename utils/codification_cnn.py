from __future__ import print_function
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Flatten, PReLU, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import time as T
import random
import numpy as np
import matplotlib.pyplot as plt

from utils.codifications import Layer, Chromosome, Fitness
from utils.utils import WarmUpCosineDecayScheduler, smooth_labels


class NNLayer(Layer):
    type = 'NN'
    possible_activations = ['relu', 'sigmoid', 'tanh', 'elu', 'prelu', 'leakyreLu']
    units_lim = 1024
    units_prob = 0.2
    act_prob = 0.2
    drop_prob = 0.2

    def __init__(self, units, activation, dropout):
        assert activation in self.possible_activations
        self.activation = activation
        self.dropout = dropout
        self.units = units

    def cross(self, other_layer):
        assert self.type == other_layer.type
        new_units = self.cross_units(other_layer.units)
        new_activation = self.cross_activation(other_layer.activation)
        new_dropout = self.cross_dropout(other_layer.dropout)
        return self.create(units=new_units, activation=new_activation, dropout=new_dropout)

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
            self.units = self.gauss_mutation(self.units, self.units_lim, 1, int_=True)
            #self.units = np.random.randint(1, self.units_lim + 1)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.drop_prob:
            self.dropout = self.gauss_mutation(self.dropout, 1, 0, int_=False)
            #self.dropout = np.random.rand()

    def self_copy(self):
        return self.create(units=self.units, activation=self.activation, dropout=self.dropout)

    @classmethod
    def random_layer(cls):
        units = np.random.randint(1, cls.units_lim + 1)
        act = random.choice(cls.possible_activations)
        drop = float(np.random.rand())
        return cls.create(units=units, activation=act, dropout=drop)

    def __repr__(self):
        return "%s|U:%d|A:%s|D:%0.3f" % (self.type, self.units, self.activation, self.dropout)


class CNNLayer(Layer):
    possible_activations = ['relu', 'sigmoid', 'tanh', 'elu', 'prelu', 'leakyreLu']
    filters_lim = 256
    possible_k = [1, 3, 5, 7]
    k_prob = 0.2
    filter_prob = 0.2
    act_prob = 0.2
    drop_prob = 0.2
    maxpool_prob = 0.3
    type = 'CNN'

    def __init__(self, filters, kernel_size, activation, dropout, maxpool):
        assert activation in self.possible_activations
        self.activation = activation
        self.dropout = dropout
        self.filters = filters
        self.k_size = kernel_size
        self.maxpool = maxpool

    def cross(self, other_layer):
        new_filters = self.cross_filters(other_layer.filters)
        new_k_size = self.cross_kernel(other_layer.k_size)
        new_activation = self.cross_activation(other_layer.activation)
        new_dropout = self.cross_dropout(other_layer.dropout)
        new_maxpool = self.cross_maxpool(other_layer.maxpool)
        return self.create(filters=new_filters, kernel_size=new_k_size,
                           activation=new_activation, dropout=new_dropout, maxpool=new_maxpool)

    def cross_kernel(self, other_kernel):
        kh = random.choice([self.k_size[0], other_kernel[0]])
        kw = random.choice([self.k_size[1], other_kernel[1]])
        # b = np.random.rand(2)
        # kh = int(self.k_size[0] * b[0] + (1 - b[0]) * other_kernel[0])
        # kw = int(self.k_size[1] * b[1] + (1 - b[1]) * other_kernel[1])
        return kh, kw

    def cross_maxpool(self, other_maxpool):
        return random.choice([self.maxpool, other_maxpool])

    def cross_activation(self, other_activation):
        return random.choice([self.activation, other_activation])

    def cross_dropout(self, other_dropout):
        b = np.random.rand()
        return self.dropout * (1 - b) + b * other_dropout

    def cross_filters(self, other_filters):
        b = np.random.rand()
        return int(self.filters * (1 - b) + other_filters * b)

    def mutate(self):
        aleatory = np.random.rand(6)
        if aleatory[0] < self.filter_prob:
            self.filters = self.gauss_mutation(self.filters, self.filters_lim, 1, int_=True)
            #self.filters = np.random.randint(1, self.filters_lim + 1)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.drop_prob:
            #self.dropout = np.random.rand()
            self.dropout = self.gauss_mutation(self.dropout, 1, 0, int_=False)
        if aleatory[4] < self.k_prob:
            self.k_size = tuple(random.choices(self.possible_k, k=2))
        if aleatory[5] < self.maxpool_prob:
            self.maxpool = not self.maxpool
            #self.maxpool = random.choice([True, False])

    def self_copy(self):
        return self.create(filters=self.filters, kernel_size=self.k_size,
                           activation=self.activation, dropout=self.dropout, maxpool=self.maxpool)

    @classmethod
    def random_layer(cls):
        filters = np.random.randint(1, cls.filters_lim + 1)
        k_size = tuple(random.choices(cls.possible_k, k=2))
        act = random.choice(cls.possible_activations)
        drop = float(np.random.rand())
        maxpool = random.choice([True, False])
        return cls.create(filters=filters, kernel_size=k_size, activation=act,
                          dropout=drop, maxpool=maxpool)

    def __repr__(self):
        return "%s|F:%d|K:(%d,%d)|A:%s|D:%0.3f|M:%d" % (self.type, self.filters, self.k_size[0], self.k_size[1],
                                                        self.activation, self.dropout, self.maxpool)


class ChromosomeCNN(Chromosome):
    max_layers = {'CNN': 5, 'NN': 3}
    layers_types = {'CNN': CNNLayer, 'NN': NNLayer}
    grow_prob = 0.1
    decrease_prob = 0.1

    def __init__(self, cnn_layers=None, nn_layers=None, fitness=None):
        super().__init__(fitness)
        self.cnn_layers = cnn_layers
        self.nn_layers = nn_layers
        if cnn_layers is None:
            self.cnn_layers = []
        if nn_layers is None:
            self.nn_layers = []
        assert type(self.cnn_layers) == list
        assert type(self.nn_layers) == list

        self.n_cnn = len(cnn_layers)
        self.n_nn = len(nn_layers)

    def random_individual(self):
        n_cnn = np.random.randint(0, self.max_layers['CNN'] + 1)
        n_nn = np.random.randint(0, self.max_layers['NN'] + 1)
        cnn_layers = [self.layers_types['CNN'].random_layer() for _ in range(n_cnn)]
        nn_layers = [self.layers_types['NN'].random_layer() for _ in range(n_nn)]
        return ChromosomeCNN(cnn_layers, nn_layers, self.evaluator)

    def simple_individual(self):
        return ChromosomeCNN([], [], self.evaluator)

    def cross(self, other_chromosome):
        new_cnn_layers = self.cross_layers(self.cnn_layers, other_chromosome.cnn_layers)
        new_nn_layers = self.cross_layers(self.nn_layers, other_chromosome.nn_layers)
        new_chromosome = ChromosomeCNN(new_cnn_layers, new_nn_layers, self.evaluator)
        return new_chromosome

    @staticmethod
    def cross_layers(this_layers, other_layers):
        new_layers = []

        if -len(other_layers) + 1 >= len(this_layers):
            return [l.self_copy() for l in this_layers]

        t = np.random.randint(-len(other_layers) + 1, len(this_layers))
        for i in range(len(this_layers)):
            j = i - t
            if j < 0 or j >= len(other_layers):
                new_layers.append(this_layers[i].self_copy())
            else:
                new_layers.append(this_layers[i].cross(other_layers[j]))

        return new_layers

    def mutate(self):
        self.mutate_layers(self.cnn_layers, 'CNN')
        self.mutate_layers(self.nn_layers, 'NN')
        self.n_cnn = len(self.cnn_layers)
        self.n_nn = len(self.nn_layers)

    def mutate_layers(self, this_layers, type_):
        for i in range(len(this_layers)):
            this_layers[i].mutate()
        if np.random.rand() < self.grow_prob and len(this_layers) < self.max_layers[type_]:
            this_layers.append(self.layers_types[type_].random_layer())
        elif np.random.rand() < self.decrease_prob and len(this_layers) > 0:
            this_layers.pop()

    def __repr__(self):
        rep = ""
        for l in self.cnn_layers + self.nn_layers:
            rep += "%s\n" % l
        return rep

    def fitness(self, test=False):
        return self.evaluator.calc(self, test=test)


class FitnessCNN(Fitness):

    def set_params(self, data, batch_size=128, epochs=100, early_stop=True, reduce_plateau=True, verbose=1,
                   reset=True, test=False, warm_epochs=0, base_lr=0.001, smooth_label=False, cosine_decay=True):
        self.smooth = smooth_label
        self.warmup_epochs = warm_epochs
        self.learning_rate_base = base_lr
        self.reset = reset
        self.test = test
        self.time = 0
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.reduce_plateu = reduce_plateau
        self.verb = verbose
        (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = data
        self.input_shape = self.x_train[0].shape
        self.num_clases = self.y_train.shape[1]
        self.cosine_decay = cosine_decay
        # self.set_callbacks = self.set_callbacks()
        if self.smooth > 0:
            self.y_train = smooth_labels(self.y_train, self.smooth)
        return self

    def set_callbacks(self):
        callbacks = []
        # Create the Learning rate scheduler.
        total_steps = int(self.epochs * self.y_train.shape[0] / self.batch_size)
        warm_up_steps = int(self.warmup_epochs * self.y_train.shape[0] / self.batch_size)
        base_steps = self.epochs * (not self.cosine_decay)
        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=self.learning_rate_base,
                                                total_steps=total_steps,
                                                warmup_learning_rate=0.0,
                                                warmup_steps=warm_up_steps,
                                                hold_base_rate_steps=base_steps)
        callbacks.append(warm_up_lr)

        checkpoint_last = ModelCheckpoint('./tmp/model_last.hdf5')
        checkpoint_acc = ModelCheckpoint('./tmp/model_acc.hdf5', monitor='val_acc', save_best_only=True)
        checkpoint_loss = ModelCheckpoint('./tmp/model_loss.hdf5', monitor='val_loss', save_best_only=True)
        callbacks.append(checkpoint_acc)
        callbacks.append(checkpoint_loss)
        callbacks.append(checkpoint_last)

        if self.early_stop and keras.__version__ == '2.2.4':
            callbacks.append(EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True))
        elif self.early_stop:
            callbacks.append(EarlyStopping(monitor='val_acc', patience=10))
        if self.reduce_plateu:
            callbacks.append(ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                               patience=5, verbose=self.verb))
        return callbacks

    def calc(self, chromosome, test=False, lr=0.001):
        print(chromosome, end="")
        print("Training...", end=" ")
        try:
            ti = T.time()
            keras.backend.clear_session()
            callbacks = self.set_callbacks()
            model = self.decode(chromosome, lr=lr)
            h = model.fit(self.x_train, self.y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          verbose=self.verb,
                          validation_data=(self.x_val, self.y_val),
                          callbacks=callbacks)

            if test:
                score = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            else:
                score = 1 - np.max(h.history['val_acc'])
        except Exception as e:
            score = [1 / self.num_clases, 1. / self.num_clases]
            if isinstance(e, ResourceExhaustedError):
                print("ResourceExhaustedError\n")
            else:
                print("Some Error!")
                print(e, "\n")
            keras.backend.clear_session()
            T.sleep(5)
            return 1 - score[1]
        if self.verb:
            model = load_model('./tmp/model_last.hdf5')
            score_test = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            score_val = 1 - h.history['val_acc'][-1]
            print('Last -> Val acc: %0.4f,Test acc: %0.4f' % (score_val, score_test))

            model = load_model('./tmp/model_acc.hdf5')
            score_test = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            score_val = 1 - np.max(h.history['val_acc'])
            print('Acc -> Val acc: %0.4f,Test acc: %0.4f' % (score_val, score_test))

            model = load_model('./tmp/model_loss.hdf5')
            score_test = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            score_val = 1 - h.history['val_acc'][int(np.argmin(h.history['val_loss']))]
            print('Loss -> Val acc: %0.4f,Test acc: %0.4f' % (score_val, score_test))

            #dataset = ['Val', 'Test'][test]
            #print('%s loss: %0.4f,%s acc: %0.4f' % (dataset, score[0], dataset, score[1]))
            self.show_result(h, 'acc')
            self.show_result(h, 'loss')
        self.time += T.time() - ti
        print("%0.4f in %0.1f min\n" % (score, (T.time() - ti) / 60))
        return score

    def decode(self, chromosome, lr=0.001):

        inp = Input(shape=self.input_shape)
        x = inp

        for i in range(chromosome.n_cnn):
            act = chromosome.cnn_layers[i].activation
            filters = chromosome.cnn_layers[i].filters
            ksize = chromosome.cnn_layers[i].k_size
            if act in ['relu', 'sigmoid', 'tanh', 'elu']:
                x = Conv2D(filters, ksize, activation=act, padding='same')(x)
            elif act == 'prelu':
                x = Conv2D(filters, ksize, padding='same')(x)
                x = PReLU()(x)
            else:
                x = Conv2D(filters, ksize, padding='same')(x)
                x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(chromosome.cnn_layers[i].dropout)(x)
            if chromosome.cnn_layers[i].maxpool:
                x = MaxPooling2D()(x)

        x = Flatten()(x)

        for i in range(chromosome.n_nn):
            act = chromosome.nn_layers[i].activation
            if act in ['relu', 'sigmoid', 'tanh', 'elu']:
                x = Dense(chromosome.nn_layers[i].units, activation=act)(x)
            elif act == 'prelu':
                x = Dense(chromosome.nn_layers[i].units)(x)
                x = PReLU()(x)
            else:
                x = Dense(chromosome.nn_layers[i].units)(x)
                x = LeakyReLU()(x)
            x = Dropout(chromosome.nn_layers[i].dropout)(x)
        x = Dense(self.num_clases, activation='softmax')(x)

        model = Model(inputs=inp, outputs=x)
        if self.verb:
            model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def show_result(history, metric='acc'):
        epochs = np.linspace(0, len(history.history['acc']) - 1, len(history.history['acc']))
        plt.plot(epochs, history.history['val_%s' % metric], label='validation')
        plt.plot(epochs, history.history[metric], label='train')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.show()

    def fitness_n_models(self, c1, c2):
        def decode_c(chromosome, inp, name=''):
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
        model = Model(inputs=inputs, outputs=[decode_c(c1, inputs, 'x1'), decode_c(c2, inputs, 'x2')],
                      name="all_net")
        losses = {'x1': "categorical_crossentropy", 'x2': "categorical_crossentropy"}
        metrics = {'x1': 'accuracy',
                   'x2': 'accuracy'}
        model.compile(optimizer=Adam(), loss=losses, metrics=metrics)
        model.summary()
        model.fit(self.x_train, [self.y_train, self.y_train],
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=self.verb,
                  validation_data=(self.x_val, [self.y_val, self.y_val]))
        # callbacks=self.callbacks)
        score = model.evaluate(self.x_val, [self.y_val, self.y_val], verbose=1)
        return score