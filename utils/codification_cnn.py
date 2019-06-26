from __future__ import print_function

import random
from time import time
from time import sleep

import keras

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Conv2D, PReLU, LeakyReLU, Dropout, SpatialDropout2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.codifications import Layer, Chromosome, Fitness
from utils.utils import smooth_labels, WarmUpCosineDecayScheduler
from utils.BN16 import BatchNormalizationF16
from utils.lr_finder import LRFinder
import shlex, subprocess
import threading


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
            # self.units = np.random.randint(1, self.units_lim + 1)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.drop_prob:
            self.dropout = self.gauss_mutation(self.dropout, 1, 0, int_=False)
            # self.dropout = np.random.rand()

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
            # self.filters = np.random.randint(1, self.filters_lim + 1)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.drop_prob:
            # self.dropout = np.random.rand()
            self.dropout = self.gauss_mutation(self.dropout, 1, 0, int_=False)
        if aleatory[4] < self.k_prob:
            self.k_size = tuple(random.choices(self.possible_k, k=2))
        if aleatory[5] < self.maxpool_prob:
            self.maxpool = not self.maxpool
            # self.maxpool = random.choice([True, False])

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
    max_layers = {'CNN': 10, 'NN': 3}
    layers_types = {'CNN': CNNLayer, 'NN': NNLayer}
    grow_prob = 0.25
    decrease_prob = 0.25

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

    @staticmethod
    def random_individual():
        max_init_cnn_layers = int(0.4 * ChromosomeCNN.max_layers['CNN'] + 1)
        max_init_nn_layers = int(0.4 * ChromosomeCNN.max_layers['NN'] + 1)
        n_cnn = np.random.randint(0, max_init_cnn_layers)
        n_nn = np.random.randint(0, max_init_nn_layers)
        cnn_layers = [ChromosomeCNN.layers_types['CNN'].random_layer() for _ in range(n_cnn)]
        nn_layers = [ChromosomeCNN.layers_types['NN'].random_layer() for _ in range(n_nn)]
        return ChromosomeCNN(cnn_layers, nn_layers, None)

    def simple_individual(self):
        return ChromosomeCNN([], [], self.evaluator)

    def cross(self, other_chromosome):
        new_cnn_layers = self.cross_layers(self.cnn_layers, other_chromosome.cnn_layers)
        new_nn_layers = self.cross_layers(self.nn_layers, other_chromosome.nn_layers)
        new_chromosome = ChromosomeCNN(new_cnn_layers, new_nn_layers, self.evaluator)
        return new_chromosome

    @staticmethod
    def cross_layers(this_layers, other_layers):
        p = np.random.rand()
        p = 0.8

        new_layers = [l.self_copy() for l in random.choice([this_layers, other_layers])]

        for i in range(min(len(other_layers), len(this_layers))):
            if p > 0.5:
                new_layer = this_layers[i].cross(other_layers[i])
                new_layers[i] = new_layer
            else:
                new_layer = this_layers[-i-1].cross(other_layers[-i-1])
                new_layers[-i-1] = new_layer

        if True:
            return new_layers

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
            if len(this_layers) == 0:
                this_layers.append(self.layers_types[type_].random_layer())
            else:
                index_to_add = int(np.random.randint(len(this_layers)))
                layer_to_copy = this_layers[index_to_add]
                new_layer = layer_to_copy.self_copy()
                new_layer.mutate()
                this_layers.insert(index_to_add + 1, new_layer)
                if type_ == 'CNN':
                    layer_to_copy.maxpool = False
        elif np.random.rand() < self.decrease_prob and len(this_layers) > 0:
            index_to_delete = int(np.random.randint(len(this_layers)))
            this_layers.pop(index_to_delete)

    def __repr__(self):
        rep = ""
        for l in self.cnn_layers + self.nn_layers:
            rep += "%s\n" % l
        return rep

    def fitness(self, test=False):
        return self.evaluator.calc(self, test=test)

    def self_copy(self):
        new_cnn_layers = [layer.self_copy() for layer in self.cnn_layers]
        new_nn_layers = [layer.self_copy() for layer in self.nn_layers]
        return ChromosomeCNN(new_cnn_layers, new_nn_layers, self.evaluator)


class FitnessCNN(Fitness):

    def set_params(self, data, batch_size=128, epochs=100, early_stop=10, reduce_plateau=True, verbose=1,
                   warm_epochs=0, base_lr=0.001, smooth_label=False, cosine_decay=True, find_lr=False, precise_epochs=None):
        self.smooth = smooth_label
        self.warmup_epochs = warm_epochs
        self.learning_rate_base = base_lr
        self.seconds = 0
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.reduce_plateu = reduce_plateau
        self.verb = verbose
        self.find_lr = find_lr
        self.precise_epochs = precise_epochs
        (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = data

        self.input_shape = self.x_train[0].shape
        self.num_clases = self.y_train.shape[1]
        self.cosine_decay = cosine_decay
        # self.set_callbacks = self.set_callbacks()
        if self.smooth > 0:
            self.y_train = smooth_labels(self.y_train, self.smooth)
        return self

    def set_callbacks(self, file_model=None, epochs=None):
        if epochs is None:
            epochs = self.epochs
        callbacks = []
        # Create the Learning rate scheduler.
        total_steps = int(epochs * self.y_train.shape[0] / self.batch_size)
        warm_up_steps = int(self.warmup_epochs * self.y_train.shape[0] / self.batch_size)
        base_steps = total_steps * (not self.cosine_decay)
        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=self.learning_rate_base,
                                                total_steps=total_steps,
                                                warmup_learning_rate=0.0,
                                                warmup_steps=warm_up_steps,
                                                hold_base_rate_steps=base_steps)
        callbacks.append(warm_up_lr)

        if file_model is not None:
            #checkpoint_last = ModelCheckpoint(file_model)
            #checkpoint_loss = ModelCheckpoint(file_model, monitor='val_loss', save_best_only=True)
            checkpoint_acc = ModelCheckpoint(file_model, monitor='val_acc', save_best_only=True)
            callbacks.append(checkpoint_acc)

        if self.early_stop > 0 and keras.__version__ == '2.2.4':
            callbacks.append(EarlyStopping(monitor='val_acc', patience=self.early_stop, restore_best_weights=True))
        elif self.early_stop > 0:
            callbacks.append(EarlyStopping(monitor='val_acc', patience=self.early_stop))
        if self.reduce_plateu:
            callbacks.append(ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                               patience=5, verbose=self.verb))
        return callbacks

    def get_params(self, precise_mode=False):
        if precise_mode and self.precise_epochs is not None:
            return self.precise_epochs
        else:
            return self.epochs

    def get_good_lr(self, model, model_file):
        lr_finder = LRFinder(model, model_file)
        lr = lr_finder.find(self.x_train, self.y_train, start_lr=0.000001, end_lr=10,
                            batch_size=self.batch_size, epochs=2, num_batches=300, return_model=False)
        return lr

    def calc(self, chromosome, test=False, lr=0.001, file_model='./model_acc.hdf5', fp=32, precise_mode=False):
        epochs = self.get_params(precise_mode)
        print("Training...", end=" ")
        if fp == 16 or fp == 160:
            keras.backend.set_floatx("float16")
            keras.backend.set_epsilon(1e-4)
        if not test and not self.find_lr:
            file_model = None
        try:
            ti = time()
            keras.backend.clear_session()
            callbacks = self.set_callbacks(file_model=file_model, epochs=epochs)
            model = self.decode(chromosome, lr=lr, fp=fp)

            if self.find_lr:
                model.save('temp.hdf5')
                lr = self.get_good_lr(model, file_model)
                self.learning_rate_base = lr
                # Set the learning rate
                model = load_model('temp.hdf5', {'BatchNormalizationF16': BatchNormalizationF16})
                keras.backend.set_value(model.optimizer.lr, lr)
                print("Learning Rate founded: %0.5f" % lr)

            h = model.fit(self.x_train, self.y_train,
                          batch_size=self.batch_size,
                          epochs=epochs,
                          verbose=self.verb,
                          validation_data=(self.x_val, self.y_val),
                          callbacks=callbacks)

            if test:
                model = load_model(file_model, {'BatchNormalizationF16': BatchNormalizationF16})
                #model = load_model(file_model)

                test_score = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
                val_score = 1 - np.max(h.history['val_acc'])
                score = (val_score, test_score)
            else:
                score = (1 - np.max(h.history['val_acc']), -1)
        except Exception as e:
            score = [1 / self.num_clases, 1. / self.num_clases]
            if isinstance(e, ResourceExhaustedError):
                print("ResourceExhaustedError\n")
            else:
                print("Some Error!")
                print(e, "\n")
            keras.backend.clear_session()
            sleep(5)
            return 1 - score[1], 1 - score[1]
        if self.verb:
            score_test = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            score_val = 1 - np.max(h.history['val_acc'])
            type_model = ['last', 'best_acc'][test]
            print('Acc -> Val acc: %0.4f,Test (%s) acc: %0.4f' % (score_val, type_model, score_test))
            self.show_result(h, 'acc')
            self.show_result(h, 'loss')
        self.seconds += time() - ti
        print("%0.4f in %0.1f min\n" % (score[0], (time() - ti) / 60))
        return score

    def decode(self, chromosome, lr=0.001, fp=32):

        inp = Input(shape=self.input_shape)
        x = BatchNormalization()(inp)

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
                
            if fp in [320, 160]:
                # fp = 320 to not use BN with FP32 and fp = 160 to not use BN with FP16
                pass
            elif fp == 321:
                x = BatchNormalization()(x)
                x = SpatialDropout2D(chromosome.cnn_layers[i].dropout)(x)
                #x = Dropout(chromosome.cnn_layers[i].dropout)(x)
                if chromosome.cnn_layers[i].maxpool:
                    x = MaxPooling2D(pool_size=3, strides=2)(x)
                    
            elif fp == 32:
                x = BatchNormalization()(x)
                if chromosome.cnn_layers[i].maxpool:
                    x = MaxPooling2D(pool_size=3, strides=2)(x)
                x = SpatialDropout2D(chromosome.cnn_layers[i].dropout)(x)
                # x = Dropout(chromosome.cnn_layers[i].dropout)(x)
                
            elif fp == 322:
                if chromosome.cnn_layers[i].maxpool:
                    x = MaxPooling2D(pool_size=3, strides=2)(x)
                x = BatchNormalization()(x)
                x = SpatialDropout2D(chromosome.cnn_layers[i].dropout)(x)
                # x = Dropout(chromosome.cnn_layers[i].dropout)(x)
                
            else:
                x = BatchNormalizationF16()(x)

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
            x = BatchNormalization()(x)
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
        argmax_val = np.argmax(history.history['val_%s' % metric])
        plt.plot(epochs, history.history['val_%s' % metric], label='validation')
        plt.plot(epochs, history.history[metric], label='train')
        plt.scatter(epochs[argmax_val], history.history['val_%s' % metric][argmax_val],
                    label='max val_%s' % metric, c='r')
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


class FitnessCNNParallel(Fitness):

    def calc(self, chromosome, test=False, precise_mode=False):
        return self.eval_list([chromosome], test=test, precise_mode=precise_mode)[0]

    class Runnable:
        def __init__(self, chromosome_file, fitness_file, command, test=False, fp=32, precise_mode=False):

            self.com_line = '%s -gf %s -ff %s -fp %d' % (command, chromosome_file, fitness_file, fp)
            if test:
                self.com_line += " -t %d" % int(test)
            if precise_mode:
                self.com_line += " -pm %d" % int(precise_mode)
                #self.com_line = '%s -gf %s -ff %s -t %d -fp %d' % \
                #            (self.command, self.chromosome_file, self.fitness_file, int(self.test), self.fp)
            #else:
            #    self.com_line = '%s -gf %s -ff %s -fp %d' % \
            #                    (self.command, self.chromosome_file, self.fitness_file, self.fp)

        def run(self):
            args = shlex.split(self.com_line)
            subprocess.call(args)

    def set_params(self, chrom_files_folder, fitness_file, fp=32, max_gpus=1,
                   main_line='python /home/daniel/proyectos/Tesis/project/GA/NeuroEvolution/train_gen.py', **kwargs):
        self.main_line = main_line
        self.chrom_folder = chrom_files_folder
        self.fitness_file = fitness_file
        self.max_gpus = max_gpus
        self.fp = fp

    def eval_list(self, chromosome_list, test=False, precise_mode=False, **kwargs):
        filenames = [self.write_chromosome(c, i) for i, c in enumerate(chromosome_list)]
        functions = []
        for filename in filenames:
            runnable = self.Runnable(filename, self.fitness_file, self.main_line, test=test, fp=self.fp,
                                     precise_mode=precise_mode)
            functions.append(runnable.run)

        threads_waiting = [threading.Thread(target=f) for f in functions]
        threads_running = []
        threads_finished = []
        simultaneous_threads = self.max_gpus

        while len(threads_waiting) > 0 or len(threads_running) > 0:
            threads_finished += [thr for thr in threads_running if not thr.isAlive()]
            threads_running = [thr for thr in threads_running if thr.isAlive()]

            if len(threads_running) < simultaneous_threads and len(threads_waiting) > 0:
                thr = threads_waiting.pop()
                thr.start()
                sleep(3)
                threads_running.append(thr)
        [thr.join() for thr in threads_finished]

        return [self.read_score("%s_score" % f) for f in filenames]

    def write_chromosome(self, chromosome, id_):
        filename = os.path.join(self.chrom_folder, "gen_%d" % id_)
        chromosome.save(filename)
        return filename
        with open(filename, 'w') as f:
            f.write(chromosome.__repr__())
        return filename

    @staticmethod
    def read_score(filename):
        test_score, val_score = None, None
        with open(filename, 'r') as f:
            for line in f:
                if 'Val_score' in line:
                    val_score = float(line.split(':')[1])
                if 'Test_score' in line:
                    test_score = float(line.split(':')[1])
        if test_score is None:
            return val_score
        else:
            return val_score, test_score



