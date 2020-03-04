from keras_preprocessing.image import ImageDataGenerator

from GA.geneticAlgorithm import TwoLevelGA
import random

from utils.codifications import Chromosome
from utils.codification_grew import HyperParams
from utils.datamanager import DataManager
from utils.utils import get_random_eraser
from utils.codification_cnn import FitnessCNN

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import os

#from utils.codification_ops import CNN, Identity
#from utils.codification_grew import CNNGrow, IdentityGrow
#CNN.decode = CNNGrow.decode
#Identity.decode = IdentityGrow.decode


class FitnessCutout(FitnessCNN):

    dataset = 'cifar10'

    # Fitness params
    _epochs = 15
    _batch_size = 128
    _verbose = False
    _redu_plat = False
    _early_stop = 0
    _warm_up_epochs = 0
    _base_lr = 0.05
    _smooth = 0.1
    _cosine_dec = False
    _lr_find = False
    _precise_eps = 75
    _include_time = False

    if dataset == 'cifar10':
        _augment = 'cutout'
        _test_eps = 200
    else:
        _augment = False
        _test_eps = 100

    def __init__(self, experiments_folder='../exp_cifar10_grow_timefit', use_resnet=True):
        super().__init__()
        if use_resnet:
            self.model = ResNet()
        else:
            exp_folder = os.path.join(experiments_folder, 'cifar10')
            self.folder = os.path.join(exp_folder, 'genetic')
            generational = TwoLevelGA.load_genetic_algorithm(folder=self.folder)
            self.model = generational.best_individual['winner']
        num_clases = 100 if self.dataset == 'cifar100' else 10
        dm = DataManager(self.dataset, clases=[], folder_var_mnist='.', num_clases=num_clases)
        data = dm.load_data()
        self.random_eraser = None
        self.set_params(data=data, verbose=self._verbose, batch_size=self._batch_size, reduce_plateau=self._redu_plat,
                        epochs=self._epochs, cosine_decay=self._cosine_dec, early_stop=self._early_stop,
                        warm_epochs=self._warm_up_epochs, base_lr=self._base_lr, smooth_label=self._smooth,
                        find_lr=self._lr_find, precise_epochs=self._precise_eps, include_time=self._include_time,
                        test_eps=self._test_eps, augment=self._augment)

    def calc(self, chromosome, test=False, file_model='./model_acc.hdf5', fp=32, precise_mode=False):
        self.random_eraser = chromosome.decode(v_l=np.min(self.data[0][0]), v_h=np.max(self.data[0][0]))
        return super().calc(self.model, test, file_model, fp, precise_mode)

    def get_datagen(self, test):
        prep_function = self.random_eraser
        return ImageDataGenerator(
                    # featurewise_center=True,
                    width_shift_range=4,
                    height_shift_range=4,
                    # fill_mode='constant',
                    horizontal_flip=True,
                    # rotation_range=15,
                    preprocessing_function=prep_function)


class ChromosomeCutout(Chromosome):
    aux = HyperParams
    mutation_prob = 0.1

    def __init__(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, pixel_level=False):
        super().__init__()
        self.p = p
        self.sl, self.sh = min(s_l, s_h), max(s_l, s_h)  # min, max proportion of erased area against input image
        self.r1, self.r2 = min(r_1, r_2), max(r_1, r_2)  # min, max aspect ratio of erased area
        self.pixel_level = pixel_level

    @classmethod
    def random_individual(cls):
        p, sl, sh, r1, r2 = np.random.rand(5)
        sl, sh = min(sl, sh), max(sl, sh)
        r1, r2 = min(r1, r2), max(r1, r2)
        pixel_level = random.choice([True, False])
        return ChromosomeCutout(p, sl, sh, r1, r2, pixel_level)

    def simple_individual(self):
        ChromosomeCutout()

    def cross(self, other_chromosome):
        p = self.aux.cross_floats(self.p, other_chromosome.p)
        sl = self.aux.cross_floats(self.sl, other_chromosome.sl)
        sh = self.aux.cross_floats(self.sh, other_chromosome.sh)
        r1 = self.aux.cross_floats(self.r1, other_chromosome.r1)
        r2 = self.aux.cross_floats(self.r2, other_chromosome.r2)
        pixel_level = random.choice([self.pixel_level, other_chromosome.pixel_level])
        return ChromosomeCutout(p, sl, sh, r1, r2, pixel_level)

    def mutate(self):
        probs = np.random.rand(6)
        if probs[0] < self.mutation_prob:
            self.p = self.aux.gauss_mutation(self.p, 1, 0, int_=False)
        if probs[0] < self.mutation_prob:
            self.sl = self.aux.gauss_mutation(self.sl, 1, 0, int_=False)
        if probs[0] < self.mutation_prob:
            self.sh = self.aux.gauss_mutation(self.sh, 1, 0, int_=False)
        if probs[0] < self.mutation_prob:
            self.r1 = self.aux.gauss_mutation(self.r1, 1, 0, int_=False)
        if probs[0] < self.mutation_prob:
            self.r2 = self.aux.gauss_mutation(self.r2, 1, 0, int_=False)
        if probs[0] < self.mutation_prob:
            self.pixel_level = random.choice([True, False])

    def __repr__(self):
        s = "P:%0.2f|SL:%0.2f|SH:%0.2f|R1:%0.2f|R2:%0.2f|PL:%s" % (self.p, self.sl, self.sh, self.r1, self.r2,
                                                                   str(self.pixel_level))
        return s

    def self_copy(self):
        return ChromosomeCutout(self.p, self.sl, self.sh, self.r1, self.r2, self.pixel_level)

    def decode(self, v_l, v_h, **kwargs):
        return get_random_eraser(self.p, self.sl, self.sh, self.r1, self.r2, v_l, v_h, self.pixel_level)


class ResNet:
    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------

    def __init__(self, n=3, version=1):

        self.n = n
        self.version = version
        self.depth = n * 6 + 2 if self.version == 1 else n * 9 + 2

    @staticmethod
    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal')
                      #kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(self, input_shape, num_classes):
        if (self.depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters,
                                      strides=strides)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters,
                                      activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def resnet_v2(self, input_shape, num_classes):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (self.depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((self.depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = self.resnet_layer(inputs=inputs,
                              num_filters=num_filters_in,
                              conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters_in,
                                      kernel_size=1,
                                      strides=strides,
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      conv_first=False)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters_in,
                                      conv_first=False)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters_out,
                                      kernel_size=1,
                                      conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters_out,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def decode(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):

        if self.version == 2:
            model = self.resnet_v2(input_shape=input_shape, num_classes=num_classes)
        else:
            model = self.resnet_v1(input_shape=input_shape, num_classes=num_classes)

        if verb:
            model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model


