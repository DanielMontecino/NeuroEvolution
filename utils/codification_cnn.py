from __future__ import print_function

import os
import random
import shlex
import subprocess
import threading
import traceback
from time import sleep
from time import time

import keras
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Conv2D, PReLU, LeakyReLU, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from utils.BN16 import BatchNormalizationF16
from utils.codifications import Layer, Chromosome, Fitness
from utils.lr_finder import LRFinder
from utils.utils import smooth_labels, WarmUpCosineDecayScheduler, EarlyStopByTimeAndAcc, CLRScheduler, lr_schedule
from utils.utils import get_random_eraser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    def __init__(self, cnn_layers=None, nn_layers=None):
        super().__init__()
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
        self.fp = 32

    @classmethod
    def random_individual(cls):
        max_init_cnn_layers = int(0.4 * cls.max_layers['CNN'] + 1)
        max_init_nn_layers = int(0.4 * cls.max_layers['NN'] + 1)
        n_cnn = np.random.randint(0, max_init_cnn_layers)
        n_nn = np.random.randint(0, max_init_nn_layers)
        cnn_layers = [cls.layers_types['CNN'].random_layer() for _ in range(n_cnn)]
        nn_layers = [cls.layers_types['NN'].random_layer() for _ in range(n_nn)]
        return ChromosomeCNN(cnn_layers=cnn_layers, nn_layers=nn_layers)

    def simple_individual(self):
        return ChromosomeCNN(cnn_layers=[], nn_layers=[])

    def cross(self, other_chromosome):
        new_cnn_layers = self.cross_layers(self.cnn_layers, other_chromosome.cnn_layers)
        new_nn_layers = self.cross_layers(self.nn_layers, other_chromosome.nn_layers)
        return ChromosomeCNN(cnn_layers=new_cnn_layers, nn_layers=new_nn_layers)

    @staticmethod
    def cross_layers(this_layers, other_layers):
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

    def self_copy(self):
        new_cnn_layers = [layer.self_copy() for layer in self.cnn_layers]
        new_nn_layers = [layer.self_copy() for layer in self.nn_layers]
        return ChromosomeCNN(cnn_layers=new_cnn_layers, nn_layers=new_nn_layers)

    def get_cnn_layer(self, inp_, activation, filters, k_size):
        if activation in ['relu', 'sigmoid', 'tanh', 'elu']:
            x_ = Conv2D(filters, k_size, activation=activation, padding='same')(inp_)
        elif activation == 'prelu':
            x_ = Conv2D(filters, k_size, padding='same')(inp_)
            x_ = PReLU()(x_)
        else:
            x_ = Conv2D(filters, k_size, padding='same')(inp_)
            x_ = LeakyReLU()(x_)
        return x_

    def get_BN_MP_DROP_block(self, inp_, maxpool, dropout):
        if self.fp == 32:
            x_ = BatchNormalization()(inp_)
            if maxpool:
                print("MAXPOOL")
                x_ = MaxPooling2D()(x_)
            x_ = Dropout(dropout)(x_)
        else:
            x_ = BatchNormalizationF16()(inp_)
            if maxpool:
                print("MAXPOOL")
                x_ = MaxPooling2D()(x_)
            x_ = Dropout(dropout)(x_)
        return x_

    def decode_layer(self, layer, inp_, allow_maxpool=False):
        act_ = layer.activation
        filters_ = layer.filters
        k_size = layer.k_size
        maxpool = layer.maxpool and allow_maxpool
        dropout = layer.dropout
        x_ = self.get_cnn_layer(inp_=inp_, activation=act_, filters=filters_, k_size=k_size)
        x_ = self.get_BN_MP_DROP_block(inp_=x_, maxpool=maxpool, dropout=dropout)
        return x_

    def get_nn_layers(self, x_, num_classes):
        for i in range(self.n_nn):
            act = self.nn_layers[i].activation
            if act in ['relu', 'sigmoid', 'tanh', 'elu']:
                x_ = Dense(self.nn_layers[i].units, activation=act)(x_)
            elif act == 'prelu':
                x_ = Dense(self.nn_layers[i].units)(x_)
                x_ = PReLU()(x_)
            else:
                x_ = Dense(self.nn_layers[i].units)(x_)
                x_ = LeakyReLU()(x_)
            if self.fp == 16:
                x_ = BatchNormalizationF16()(x_)
            else:
                x_ = BatchNormalization()(x_)
            x_ = Dropout(self.nn_layers[i].dropout)(x_)
        x_ = Dense(num_classes, activation='softmax')(x_)
        return x_

    def decode(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):
        self.fp = fp
        inp = Input(shape=input_shape)
        x = inp

        for i in range(self.n_cnn):
            act = self.cnn_layers[i].activation
            filters = self.cnn_layers[i].filters
            ksize = self.cnn_layers[i].k_size
            maxpool = self.cnn_layers[i].maxpool
            dropout = self.cnn_layers[i].dropout
            x = self.get_cnn_layer(x, act, filters, ksize)
            x = self.get_BN_MP_DROP_block(x, maxpool, dropout)

        x = Flatten()(x)
        x = self.get_nn_layers(x, num_classes=num_classes)
        model = Model(inputs=inp, outputs=x)
        if verb:
            model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy'])
        return model
        '''    
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
                x = Dropout(self.cnn_layers[i].dropout)(x)
                if self.cnn_layers[i].maxpool:
                    x = MaxPooling2D()(x)

            elif fp == 32:
                x = BatchNormalization()(x)
                if self.cnn_layers[i].maxpool:
                    x = MaxPooling2D()(x)
                x = Dropout(self.cnn_layers[i].dropout)(x)

            elif fp == 322:
                if self.cnn_layers[i].maxpool:ResNet151
                    x = MaxPooling2D()(x)
                x = BatchNormalization()(x)
                x = Dropout(self.cnn_layers[i].dropout)(x)

            else:
                x = BatchNormalizationF16()(x)

        x = Flatten()(x)

        for i in range(self.n_nn):
            act = self.nn_layers[i].activation
            if act in ['relu', 'sigmoid', 'tanh', 'elu']:
                x = Dense(self.nn_layers[i].units, activation=act)(x)
            elif act == 'prelu':
                x = Dense(self.nn_layers[i].units)(x)
                x = PReLU()(x)
            else:
                x = Dense(self.nn_layers[i].units)(x)
                x = LeakyReLU()(x)
            x = Dropout(self.nn_layers[i].dropout)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inp, outputs=x)
        if verb:
            model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy'])
        return model
        '''


class FitnessCNN(Fitness):

    def __init__(self):
        super().__init__()
        self.smooth = None
        self.warmup_epochs = None
        self.learning_rate_base = None
        self.batch_size = None
        self.epochs = None
        self.early_stop = None
        self.reduce_plateu = None
        self.verb = None
        self.find_lr = None
        self.precise_epochs = None
        (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = [(None, None)] * 3
        self.input_shape = None
        self.num_clases = None
        self.cosine_decay = None
        self.y_train = None
        self.include_time = False
        self.test = False
        self.test_eps = 200
        self.augment = True

    def set_params(self, data, batch_size=128, epochs=100, early_stop=0, reduce_plateau=False, verbose=1,
                   warm_epochs=0, base_lr=0.001, smooth_label=False, cosine_decay=False, find_lr=False,
                   precise_epochs=None, include_time=False, test_eps=200, augment=True):
        self.smooth = smooth_label
        self.warmup_epochs = warm_epochs
        self.learning_rate_base = base_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.reduce_plateu = reduce_plateau
        self.verb = verbose
        self.find_lr = find_lr
        self.precise_epochs = precise_epochs
        self.data = data
        self.input_shape = self.data[0][0][0].shape
        self.num_clases = self.data[0][1].shape[1]
        self.cosine_decay = cosine_decay
        self.include_time = include_time
        self.test_eps = test_eps
        self.augment = augment
        # self.set_callbacks = self.set_callbacks()
        return self

    def set_callbacks(self, file_model=None, epochs=None):
        if epochs is None:
            epochs = self.epochs
        callbacks = []
        # Create the Learning rate scheduler.
        total_steps = int(epochs * self.y_train.shape[0] / self.batch_size)
        warm_up_steps = int(self.warmup_epochs * self.y_train.shape[0] / self.batch_size)
        base_steps = total_steps * (not self.cosine_decay)
        if self.cosine_decay:
            schedule = WarmUpCosineDecayScheduler(learning_rate_base=self.learning_rate_base,
                                                  total_steps=total_steps,
                                                  warmup_learning_rate=0.0,
                                                  warmup_steps=warm_up_steps,
                                                  hold_base_rate_steps=base_steps)
            #schedule = LearningRateScheduler(lr_schedule)
        else:
            schedule = CLRScheduler(max_lr=self.learning_rate_base,
                                    min_lr=0.00002,
                                    total_steps=total_steps)
        callbacks.append(schedule)
        min_val_acc = (1. / self.num_clases) + 0.1
        early_stop = EarlyStopByTimeAndAcc(limit_time=360,
                                           baseline=min_val_acc,
                                           patience=8)
        callbacks.append(early_stop)
        print("No Early stopping")
        # callbacks.append(EarlyStopping(monitor='val_acc', patience=epochs//5, baseline=min_val_acc))
        val_acc = 'val_accuracy' if keras.__version__ == '2.3.1' else 'val_acc'
        if file_model is not None:
            # checkpoint_last = ModelCheckpoint(file_model)
            # checkpoint_loss = ModelCheckpoint(file_model, monitor='val_loss', save_best_only=True)                
            checkpoint_acc = ModelCheckpoint(file_model, monitor=val_acc, save_best_only=True)
            callbacks.append(checkpoint_acc)

        if self.early_stop > 0 and keras.__version__ == '2.2.4':
            callbacks.append(EarlyStopping(monitor=val_acc, patience=self.early_stop, restore_best_weights=True))
        elif self.early_stop > 0:
            callbacks.append(EarlyStopping(monitor=val_acc, patience=self.early_stop))
        if self.reduce_plateu:
            callbacks.append(ReduceLROnPlateau(monitor=val_acc, factor=0.2,
                                               patience=5, verbose=self.verb))

        return callbacks

    def get_params(self, chromosome, precise_mode=False, test=False):
        (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = self.data
        epochs = self.epochs
        if precise_mode and self.precise_epochs is not None:
            epochs = self.precise_epochs
        if test:
            self.x_train = np.concatenate([np.copy(self.x_train), np.copy(self.x_val)])
            self.x_val = np.copy(self.x_test)[0:3000, ...]
            self.y_train = np.concatenate([np.copy(self.y_train), np.copy(self.y_val)])
            self.y_val = np.copy(self.y_test)[0:3000, ...]
            epochs = self.test_eps
        if self.smooth > 0:
            self.y_train = smooth_labels(self.y_train, self.smooth)
            
        return epochs

    def get_good_lr(self, model, model_file):
        lr_finder = LRFinder(model, model_file)
        lr = lr_finder.find(self.x_train, self.y_train, start_lr=0.000001, end_lr=10,
                            batch_size=self.batch_size, epochs=2, num_batches=300, return_model=False)
        return lr

    def test_model(self, model, iterations=100):
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = self.data
        datagen = self.get_datagen(test=True)
        datagen.fit(x_train)
        n_corrects = 0
        for i in range(x_test.shape[0]):
            x = np.expand_dims(x_test[i,...], 0)
            y = np.argmax(y_test[i, ...])
            iterator = datagen.flow(x)
            x_trans = [iterator.next() for _ in range(iterations)]
            x_trans = np.concatenate(x_trans)
            all_y = model.predict(x_trans)
            y_pred = np.argmax(np.max(all_y, axis=0))
            n_corrects += y_pred == y
        return n_corrects / x_test.shape[0]



    def calc(self, chromosome, test=False, file_model=None, fp=32, precise_mode=False):
        #self.verb = True
        self.test = test
        epochs = self.get_params(chromosome, precise_mode, test)
        print("Training...", end=" ")
        if fp == 16 or fp == 160:
            keras.backend.set_floatx("float16")
            keras.backend.set_epsilon(1e-4)
        if (test or self.find_lr) and file_model is None:
            file_model = './model_acc.hdf5'
        try:
            ti = time()
            keras.backend.clear_session()
            callbacks = self.set_callbacks(file_model=file_model, epochs=epochs)
            model = chromosome.decode(num_classes=self.num_clases, input_shape=self.input_shape,
                                      verb=self.verb, fp=fp)

            if self.reduce_plateu:
                model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(),
                      metrics=['accuracy'])
            if self.find_lr:
                model.save('temp.hdf5')
                lr = self.get_good_lr(model, file_model)
                self.learning_rate_base = lr
                # Set the learning rate
                model = load_model('temp.hdf5', {'BatchNormalizationF16': BatchNormalizationF16})
                keras.backend.set_value(model.optimizer.lr, lr)
                print("Learning Rate founded: %0.5f" % lr)

            if self.augment:
                datagen = self.get_datagen(test=test)
                datagen.fit(self.x_train)
                h = model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                        validation_data=(self.x_val, self.y_val),
                                        epochs=epochs, verbose=self.verb, workers=6 if self.test else 4,
                                        callbacks=callbacks,
                                        steps_per_epoch=int(self.x_train.shape[0] / self.batch_size))
            else:
                h = model.fit(self.x_train, self.y_train,
                              batch_size=self.batch_size,
                              epochs=epochs,
                              validation_data=(self.x_val, self.y_val),
                              callbacks=callbacks,
                              verbose=self.verb,
                              shuffle=True)
            if test:
                # model = load_model(file_model, {'BatchNormalizationF16': BatchNormalizationF16})
                # model = load_model(file_model)

                score = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            else:
                key_val_acc = [key for key in h.history.keys() if 'val_acc' in key][0]
                interval = epochs // 3
                val_history = np.array(h.history[key_val_acc][-interval::])
                sorted_ids = np.argsort(val_history)
                print(sorted_ids.shape)
                second_best_id = sorted_ids[-2] if len(sorted_ids)>1 else sorted_ids[-1]
                score = 1 - val_history[second_best_id]

                #score = 1 - np.max(h.history[key_val_acc][-interval::])
                #score    += np.std(h.history[key_val_acc][-interval::])
                if self.include_time:
                    training_time = time() - ti
                    score += np.log(training_time / epochs) / 1000
                   
        except Exception as e:
            score = [1 / self.num_clases, 1. / self.num_clases]
            if isinstance(e, ResourceExhaustedError):
                print("ResourceExhaustedError\n")
            else:
                traceback.print_exc()
                print("Some Error!")
                print(e, "\n")
            keras.backend.clear_session()
            sleep(5)
            return 1.1
        if self.verb:
            score_test = 1 - model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            key_val_acc = [key for key in h.history.keys() if 'val_acc' in key][0]
            score_val = 1 - np.max(h.history[key_val_acc][-10::])
            type_model = ['best_acc', 'last'][test]
            print('Acc -> Val acc: %0.4f,Test (%s) acc: %0.4f' % (score_val, type_model, score_test))
            self.show_result(h, 'acc')
            self.show_result(h, 'loss')
        print("%0.4f in %0.1f min\n" % (score, (time() - ti) / 60))
        return score

    def get_datagen(self, test):
        if self.augment == 'cutout' and test:
            prep_function = get_random_eraser(v_l=np.min(self.x_train), v_h=np.max(self.x_train))
            print("With Cutout augmentation")
        else:
            prep_function = None
            print("Without cutout augmentation")
        return ImageDataGenerator(
                    # featurewise_center=True,
                    width_shift_range=5,
                    height_shift_range=5,
                    fill_mode='nearest',
                    horizontal_flip=True,
                    rotation_range=0,
                    preprocessing_function=prep_function)

    @staticmethod
    def show_result(history, metric='acc'):
        if metric not in history.history.keys() and metric == 'acc':
            metric = 'accuracy'
        epochs = np.linspace(0, len(history.history[metric]) - 1, len(history.history[metric]))
        argmax_val = np.argmax(history.history['val_%s' % metric])
        plt.plot(epochs, history.history['val_%s' % metric], label='validation')
        plt.plot(epochs, history.history[metric], label='train')
        plt.scatter(epochs[argmax_val], history.history['val_%s' % metric][argmax_val],
                    label='max val_%s' % metric, c='r')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.show()


class FitnessCNNParallel(Fitness):

    def __init__(self):
        super().__init__()
        self.main_line = None
        self.chrom_folder = None
        self.fitness_file = None
        self.max_gpus = None
        self.fp = None

    def calc(self, chromosome, test=False, precise_mode=False):
        return self.eval_list([chromosome], test=test, precise_mode=precise_mode)[0]

    class Runnable:
        def __init__(self, chromosome_file, fitness_file, command, test=False, fp=32, precise_mode=False, file_model=None):

            self.com_line = '%s -gf %s -ff %s -fp %d' % (command, chromosome_file, fitness_file, fp)
            if test:
                self.com_line += " -t %d" % int(test)
            if precise_mode:
                self.com_line += " -pm %d" % int(precise_mode)
            if file_model is not None:
                self.com_line += " -fm %s" % file_model

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

    def eval_list(self, chromosome_list, test=False, precise_mode=False, file_model_list=None, **kwargs):
        filenames = [self.write_chromosome(c, i) for i, c in enumerate(chromosome_list)]
        if file_model_list is not None:
            assert len(filenames) == len(file_model_list)
        else:
            file_model_list = [None] * len(filenames)
        functions = []
        for file_model, filename in zip(file_model_list, filenames):
            runnable = self.Runnable(filename, self.fitness_file, self.main_line, test=test, fp=self.fp,
                                     precise_mode=precise_mode, file_model=file_model)
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

    @staticmethod
    def read_score(filename):
        score = None
        with open(filename, 'r') as f:
            for line in f:
                if 'Score' in line:
                    score = float(line.split(':')[1])
        return score
