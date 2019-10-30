import numpy as np
import random

from utils.codification_cnn import NNLayer
from utils.codification_skipc import ChromosomeSkip
from utils.codifications import Layer
from utils.BN16 import BatchNormalizationF16

from keras.layers import Conv2D, PReLU, LeakyReLU, Dropout, MaxPooling2D, Flatten, BatchNormalization, \
    GlobalAveragePooling2D
from keras import Input, Model
from keras.optimizers import Adam


class CNNLayer_WOMP(Layer):
    possible_activations = ['relu', 'sigmoid', 'tanh', 'elu', 'prelu', 'leakyreLu']
    filters_mul_range = [0.5, 1.5]
    possible_k = [1, 3, 5, 7]
    k_prob = 0.2
    filter_prob = 0.2
    act_prob = 0.2
    type = 'CNN'

    def __init__(self, filter_mul, kernel_size, activation):
        assert activation in self.possible_activations
        assert isinstance(filter_mul, float)
        assert isinstance(kernel_size, int)
        self.activation = activation
        self.filter_mul = filter_mul
        self.k_size = kernel_size
        self.dropout = 0.3
        self.maxpool = False

    def cross(self, other_layer):
        new_filter_mul = self.cross_filter_mul(other_layer.filter_mul)
        new_k_size = self.cross_kernel(other_layer.k_size)
        new_activation = self.cross_activation(other_layer.activation)
        return self.create(filter_mul=new_filter_mul, kernel_size=new_k_size,
                           activation=new_activation)

    def cross_filter_mul(self, other_filters_mul):
        b = np.random.rand()
        return self.filter_mul * (1 - b) + other_filters_mul * b

    def cross_kernel(self, other_kernel):
        k = random.choice([self.k_size, other_kernel])
        # b = np.random.rand()
        # k = int(self.k_size * b + (1 - b) * other_kernel)
        return k

    def cross_activation(self, other_activation):
        return random.choice([self.activation, other_activation])

    def mutate(self):
        aleatory = np.random.rand(6)
        if aleatory[0] < self.filter_prob:
            self.filter_mul = self.gauss_mutation(self.filter_mul, self.filters_mul_range[1],
                                                  self.filters_mul_range[0], int_=False)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.k_prob:
            self.k_size = random.choice(self.possible_k)

    def self_copy(self):
        return self.create(filter_mul=self.filter_mul, kernel_size=self.k_size,
                           activation=self.activation)

    @classmethod
    def random_layer(cls):
        filter_mul = np.random.uniform(CNNLayer_WOMP.filters_mul_range[0], CNNLayer_WOMP.filters_mul_range[1])
        k_size = random.choice(cls.possible_k)
        act = random.choice(cls.possible_activations)
        return cls.create(filter_mul=filter_mul, kernel_size=k_size, activation=act)

    def __repr__(self):
        return "CNN|F:%0.1f|K:%d|A:%s" % (self.filter_mul, self.k_size, self.activation)


class ChromosomeFilterGrow(ChromosomeSkip):

    layers_types = {'CNN': CNNLayer_WOMP, 'NN': NNLayer}
    num_blocks = 3
    initial_filters = 32

    def __init__(self, cnn_layers=None, nn_layers=None, connections=None, chromosome_skip=None):
        if chromosome_skip is not None:
            super().__init__(cnn_layers=chromosome_skip.cnn_layers,
                             nn_layers=chromosome_skip.nn_layers,
                             connections=chromosome_skip.connections)
        else:
            super().__init__(cnn_layers=cnn_layers, nn_layers=nn_layers, connections=connections)

    @classmethod
    def random_individual(cls):
        chromosome_cnn = super().random_individual()
        return ChromosomeFilterGrow(chromosome_skip=chromosome_cnn)

    def simple_individual(self):
        chromosome_cnn = super().simple_individual()
        return ChromosomeFilterGrow(chromosome_skip=chromosome_cnn)

    def cross(self, other_chromosome):
        new_chromosome = super().cross(other_chromosome)
        return ChromosomeFilterGrow(chromosome_skip=new_chromosome)

    def self_copy(self):
        chromosome_cnn = super().self_copy()
        return ChromosomeFilterGrow(chromosome_skip=chromosome_cnn)

    @staticmethod
    def decode_layer(layer, inp_, allow_maxpool=False, fp=32):
        act_ = layer.activation
        input_features = inp_._shape_as_list()[-1]
        filters_ = int(layer.filter_mul * input_features)
        k_size = layer.k_size
        maxpool = layer.maxpool and allow_maxpool
        dropout = layer.dropout
        x_ = ChromosomeFilterGrow.get_cnn_layer(inp_=inp_, activation=act_, filters=filters_, k_size=k_size)
        x_ = ChromosomeFilterGrow.get_BN_MP_DROP_block(inp_=x_, maxpool=maxpool, dropout=dropout, fp=fp)
        return x_

    def decode(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):
        inp = Input(shape=input_shape)
        x = BatchNormalization()(inp)
        x = Conv2D(self.initial_filters, 1, activation='relu', padding='same')(x)
        x = Dropout(0.3)(x)
        for i in range(self.num_blocks):
            layers = self.get_cell(x, allow_maxpool=False, fp=fp)
            if len(layers) == 0:
                break
            if i == self.num_blocks - 1:
                x = layers[-1]
            else:
                x = MaxPooling2D()(layers[-1])
        x = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)
        x = self.get_nn_layers(x, num_classes=num_classes)
        model = Model(inputs=inp, outputs=x)
        if verb:
            model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model
