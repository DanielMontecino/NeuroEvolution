
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, PReLU, LeakyReLU, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import os
from utils.BN16 import BatchNormalizationF16
from utils.codification_cnn import ChromosomeCNN, FitnessCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ChromosomeSkip(ChromosomeCNN):
    skip_prob = 0.1
    allow_maxpool = True
    union_operations = ['concat', 'add']

    def __init__(self, cnn_layers=None, nn_layers=None, connections=None, fitness=None):
        super().__init__(cnn_layers=cnn_layers, nn_layers=nn_layers, fitness=fitness)
        self.connections = connections
        if not isinstance(self.connections, Connections):
            print("Problems with type", type(self.connections))
            print(self.connections)

    @staticmethod
    def random_individual():
        chromosome_cnn = ChromosomeCNN.random_individual()
        n_blocks = len(chromosome_cnn.cnn_layers)
        new_connections = Connections.random_connections(n_blocks)
        return ChromosomeSkip(cnn_layers=chromosome_cnn.cnn_layers,
                              nn_layers=chromosome_cnn.nn_layers,
                              connections=new_connections)

    def simple_individual(self):
        chromosome_cnn = super().simple_individual()
        n_blocks = len(chromosome_cnn.cnn_layers)
        new_connections = Connections.random_connections(n_blocks)
        return ChromosomeSkip(chromosome_cnn.cnn_layers, chromosome_cnn.nn_layers, new_connections)

    def cross(self, other_chromosome):
        new_chromosome = super().cross(other_chromosome)
        new_connections = self.connections.cross(other_chromosome.connections, len(new_chromosome.cnn_layers))
        return ChromosomeSkip(cnn_layers=new_chromosome.cnn_layers,
                              nn_layers=new_chromosome.nn_layers,
                              connections=new_connections)

    def mutate_layers(self, this_layers, type_):
        for i in range(len(this_layers)):
            this_layers[i].mutate()
        if np.random.rand() < self.grow_prob and len(this_layers) < self.max_layers[type_]:
            if len(this_layers) == 0:
                this_layers.append(self.layers_types[type_].random_layer())
            else:
                # Choose a random layer, copy it, mutate it and add it next to the original one
                index_to_add = int(np.random.randint(len(this_layers)))
                layer_to_copy = this_layers[index_to_add]
                new_layer = layer_to_copy.self_copy()
                new_layer.mutate()
                this_layers.insert(index_to_add + 1, new_layer)
                if type_ == 'CNN':
                    self.connections.add_connection(index_to_add)
        elif np.random.rand() < self.decrease_prob and len(this_layers) > 0:
            index_to_delete = int(np.random.randint(len(this_layers)))
            this_layers.pop(index_to_delete)
            if type_ == 'CNN':
                self.connections.delete_connection(index_to_delete)

    def mutate(self):
        super().mutate()
        if np.random.rand() < self.skip_prob:
            self.connections.mutate()

    def __repr__(self):
        return super().__repr__() + self.connections.__repr__()

    def fitness(self, test=False):
        return self.evaluator.calc(self, test=test)

    def self_copy(self):
        chromosome_cnn = super().self_copy()
        new_connections = self.connections.self_copy()
        return ChromosomeSkip(cnn_layers=chromosome_cnn.cnn_layers,
                              nn_layers=chromosome_cnn.nn_layers,
                              connections=new_connections)


class Connections:

    def __init__(self, matrix):
        self.matrix = matrix.astype(np.int32)

    def cross(self, another_connections, n_blocks):
        # Combine the rows between matrices, that is, keep the input connections
        # of each block
        n_blocks = max(n_blocks - 1, 0)
        new_matrix = np.zeros((n_blocks, n_blocks)).astype(np.int32)
        if new_matrix.size == 0:
            return Connections(new_matrix)

        if self.matrix.shape[0] <= another_connections.matrix.shape[0]:
            first_connect = self.matrix
            second_connect = another_connections.matrix
        else:
            first_connect = another_connections.matrix
            second_connect = self.matrix

        min_blocks = min(self.matrix.shape[0], another_connections.matrix.shape[0])
        cross_point = np.random.randint(0, min_blocks + 1)

        new_matrix[0:cross_point, 0:cross_point] = first_connect[0:cross_point, 0:cross_point]
        new_matrix[cross_point:n_blocks, 0:n_blocks] = second_connect[cross_point:n_blocks, 0:n_blocks]

        new_matrix = self.fix_connections(new_matrix)
        return Connections(new_matrix)

    def add_connection(self, index):
        self.matrix = np.insert(self.matrix, index, 0, axis=0)
        self.matrix = np.insert(self.matrix, index, 0, axis=1)
        self.matrix[index, index] = 1

    def delete_connection(self, index):
        if self.matrix.shape[0] == 0:
            return
        deleted_row = max(index - 1, 0)
        deleted_col = min(self.matrix.shape[0] - 1, index)
        new_matrix = np.delete(self.matrix, deleted_row, axis=0)
        new_matrix = np.delete(new_matrix, deleted_col, axis=1)
        new_matrix = self.fix_connections(new_matrix)
        self.matrix = new_matrix

    def mutate(self):
        self.matrix = self.try_to_mutate(self.matrix)
        return

    def __repr__(self):
        s = ""
        n_blocks = self.matrix.shape[0]
        for i in range(n_blocks):
            for j in range(i+1):
                s += str(self.matrix[i, j])
            s += "\n"
        return s

    def try_to_mutate(self, matrix, n_tries=5):
        mutated_matrix = matrix.copy()
        if matrix.shape[0] == 0:
            return matrix
        while (mutated_matrix == matrix).all() and n_tries > 0:
            i = np.random.randint(0, matrix.shape[0])
            j = np.random.randint(0, i + 1)
            mutated_matrix[i, j] = int(not mutated_matrix[i, j])
            n_tries -= 1
            mutated_matrix = self.fix_connections(mutated_matrix)
        return mutated_matrix

    def compare(self, a_connections):
        if self.matrix.shape[0] != a_connections.matrix.shape[0]:
            return False
        return (self.matrix == a_connections.matrix).all()

    def self_copy(self):
        return Connections(self.matrix.copy())

    @staticmethod
    def random_connections(n_blocks):
        # n_blocks include the input block, so the shape's connection matrix has to be n_blocks - 1
        n_blocks = max(n_blocks - 1, 0)
        matrix = np.random.randint(0, 2, size=(n_blocks, n_blocks))
        matrix = Connections.fix_connections(matrix)
        return Connections(matrix)

    @staticmethod
    def fix_connections(matrix):
        # Only the inferior triangle part of the matrix is util.
        # The upper part of the matrix make recurrent connections:
        #
        #      block_0| 1
        #      block_1| 1     0
        # Inpt block_2| 0     1       0
        #      block_3| 0     0       1        1
        #              inp  block_0  block_1 block_2
        #                        Outputs

        matrix = np.tril(matrix)

        # Add INPUT connections to blocks who dont have any one.
        # that is, when its corresponding row is empty
        empty_rows = np.sum(matrix, axis=1) == 0
        for i in range(matrix.shape[0]):
            if empty_rows[i]:
                random_index = np.random.randint(0, i + 1)
                matrix[i, random_index] = 1

        # Add OUTPUT connections to blocks who dont have any one.
        # that is, when its corresponding column is empty
        empty_cols = np.sum(matrix, axis=0) == 0
        for j in range(matrix.shape[0]):
            if empty_cols[j]:
                random_index = np.random.randint(j, matrix.shape[0])
                matrix[random_index, j] = 1
        return matrix


class FitnessSkip(FitnessCNN):

    def decode(self, chromosome, lr=0.001, fp=32):
        connections = chromosome.connections.matrix
        cnn_layers = chromosome.cnn_layers

        def decode_layer(layer, inp_):
            act_ = layer.activation
            filters = layer.filters
            k_size = layer.k_size
            if act_ in ['relu', 'sigmoid', 'tanh', 'elu']:
                x_ = Conv2D(filters, k_size, activation=act_, padding='same')(inp_)
            elif act_ == 'prelu':
                x_ = Conv2D(filters, k_size, padding='same')(inp_)
                x_ = PReLU()(x_)
            else:
                x_ = Conv2D(filters, k_size, padding='same')(inp_)
                x_ = LeakyReLU()(x_)

            if fp == 32:
                x_ = BatchNormalization()(x_)
                if layer.maxpool:
                    x_ = MaxPooling2D(pool_size=3, strides=2)(x_)
            else:
                x_ = BatchNormalizationF16()(x_)
                if layer.maxpool:
                    x_ = MaxPooling2D(pool_size=3, strides=2)(x_)
            x_ = Dropout(layer.dropout)(x_)

            return x_

        inp = Input(shape=self.input_shape)
        x = BatchNormalization()(inp)

        layers = []
        if len(cnn_layers) > 0:
            layers.append(decode_layer(cnn_layers[0], x))

        for block in range(connections.shape[0]):
            input_connections = []
            for input_layer in range(connections.shape[0]):
                if connections[block, input_layer] == 1:
                    input_connections.append(layers[input_layer])

            if len(input_connections) > 1:
                shapes = [l._shape_as_list()[1] for l in input_connections]
                min_shape = np.min(shapes)
                for k in range(len(input_connections)):
                    maxpool_size = int(shapes[k] / min_shape)
                    if maxpool_size > 1:
                        input_connections[k] = MaxPooling2D(maxpool_size)(input_connections[k])
                x = concatenate(input_connections)
            else:
                x = input_connections[0]
            layers.append(decode_layer(cnn_layers[block + 1], x))

        if len(layers) == 0:
            x = Flatten()(x)
        else:
            x = Flatten()(layers[-1])

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
