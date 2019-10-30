import random
import numpy as np
from utils.codifications import Chromosome
from utils.codification_cnn import ChromosomeCNN, CNNLayer
from utils.codifications import Layer

from keras.layers import Conv2D, PReLU, LeakyReLU, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Dense
from keras import Input, Model
from keras.optimizers import Adam


class AbstractGen:
    _type = "AbstractGen"

    def __init__(self, **kwargs):
        pass

    @classmethod
    def type(cls):
        return cls._type

    @classmethod
    def random(cls, **kwargs):
        return cls()

    def cross(self, other_gen):
        pass

    def mutate(self):
        pass

    def __repr__(self):
        return self._type

    def self_copy(self):
        return self.random() # Return a Object of the same class


class Inputs(AbstractGen):
    _type = "Inputs"
    _mutate_prob = 0.5

    def __init__(self, inputs_array, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(inputs_array, np.ndarray)
        assert len(inputs_array.shape) < 2
        self.inputs = inputs_array

    @classmethod
    def random(cls, max_inputs):
        if max_inputs == 0:
            return Inputs(np.array([]))
        inputs = np.random.randint(0, 2, max_inputs)
        if np.sum(inputs) == 0:
            idx = np.random.randint(0, max_inputs)
            inputs[idx] = 1
        return Inputs(inputs)

    def cross(self, other_inputs):
        assert self.inputs.size == other_inputs.inputs.size
        new_inputs = []
        for i in range(self.inputs.size):
            new_inputs.append(random.choice([self.inputs[i], other_inputs.inputs[i]]))
        new_inputs = np.array(new_inputs)
        return Inputs(new_inputs)

    def mutate(self):
        if np.random.rand() < self._mutate_prob and self.inputs.size > 1:
            idx_to_mutate = np.random.randint(0, self.inputs.size)
            self.inputs[idx_to_mutate] = not self.inputs[idx_to_mutate]
            if np.sum(self.inputs) == 0:
                idx = np.random.randint(0, self.inputs.size)
                self.inputs[idx] = 1

    def __repr__(self):
        inputs = list(self.inputs.astype(np.int32))
        repr = ""
        for INPUT in inputs:
            repr += str(INPUT)
        return repr

    def self_copy(self):
        return Inputs(self.inputs.copy())

    def set(self, new_inputs):
        assert isinstance(new_inputs, np.ndarray)
        assert len(new_inputs.shape) < 2
        self.inputs = new_inputs.copy()

    def are_more_than_one(self):
        return True if (np.sum(self.inputs) > 1) else False


class Operation(AbstractGen):
    _type = 'Op'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_operation):
        assert isinstance(other_operation, Operation)
        if self.type() != other_operation.type():
            selected = random.choice([self, other_operation])
            return selected.self_copy()


class Merger(AbstractGen):
    _type = "Merge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_merger):
        assert isinstance(other_merger, Merger)
        if self.type() != other_merger.type():
            selected = random.choice([self, other_merger])
            return selected.self_copy()


class Concatenation(Merger):
    _type = "CAT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return Concatenation()


class Sum(Merger):
    _type = "SUM"

    def __init__(self):
        super().__init__()

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return Sum()


class Identity(Operation):
    _type = "Identity"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return Identity()


class CNN(Operation):
    possible_activations = ['relu', 'sigmoid', 'tanh', 'elu', 'prelu', 'leakyreLu']
    filters_mul_range = [0.5, 1.5]
    dropout_range = [0, 0.8]
    possible_k = [1, 3, 5, 7]
    k_prob = 0.2
    drop_prob = 0.2
    filter_prob = 0.2
    act_prob = 0.2
    _type = 'CNN'

    def __init__(self, filter_mul, kernel_size, activation, dropout, **kwargs):
        super().__init__(**kwargs)
        assert activation in self.possible_activations
        assert isinstance(filter_mul, float)
        assert isinstance(dropout, float)
        assert isinstance(kernel_size, int)
        self.activation = activation
        self.filter_mul = filter_mul
        self.k_size = kernel_size
        self.dropout = dropout

    @classmethod
    def random(cls):
        filter_mul = np.random.uniform(cls.filters_mul_range[0], cls.filters_mul_range[1])
        k_size = random.choice(cls.possible_k)
        act = random.choice(cls.possible_activations)
        dropout = np.random.uniform(cls.dropout_range[0], cls.dropout_range[1])
        return cls(filter_mul=filter_mul, kernel_size=k_size, activation=act, dropout=dropout)

    def cross(self, other_operation):
        super().cross(other_operation)
        new_filter_mul = self.cross_filter_mul(other_operation.filter_mul)
        new_k_size = self.cross_kernel(other_operation.k_size)
        new_activation = self.cross_activation(other_operation.activation)
        new_dropout = self.cross_dropout(other_operation.dropout)
        return CNN(filter_mul=new_filter_mul, kernel_size=new_k_size,
                   activation=new_activation, dropout=new_dropout)

    def cross_dropout(self, other_dropout):
        b = np.random.rand()
        return self.dropout * (1 - b) + b * other_dropout

    def cross_filter_mul(self, other_filters_mul):
        b = np.random.rand()
        return self.filter_mul * (1 - b) + other_filters_mul * b

    def cross_kernel(self, other_kernel):
        k = random.choice([self.k_size, other_kernel])
        return k

    def cross_activation(self, other_activation):
        return random.choice([self.activation, other_activation])

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

    def mutate(self):
        aleatory = np.random.rand(5)
        if aleatory[0] < self.filter_prob:
            self.filter_mul = self.gauss_mutation(self.filter_mul, self.filters_mul_range[1],
                                                  self.filters_mul_range[0], int_=False)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.drop_prob:
            self.dropout = self.gauss_mutation(self.dropout, self.dropout_range[1], self.dropout_range[0], int_=False)
        if aleatory[4] < self.k_prob:
            self.k_size = random.choice(self.possible_k)

    def __repr__(self):
        return "%s|F:%0.2f|K:%d|A:%s|D:%0.3f" % (self.type(), self.filter_mul, self.k_size, self.activation, self.dropout)

    def self_copy(self):
        return CNN(filter_mul=self.filter_mul, kernel_size=self.k_size,
                   activation=self.activation, dropout=self.dropout)


class Block:
    _operations = [Identity, CNN]
    _mergers = [Concatenation, Sum]
    _inputs = Inputs
    _classes = {'CNN': CNNLayer}

    _change_op_prob = 0.1
    _change_concat_prob = 0.1

    def __init__(self, operation_type, ops, concatenation, inputs):
        self.op_type = operation_type
        self.ops = ops
        self.concat = concatenation
        self.inputs = inputs
        assert isinstance(self.op_type, str)
        assert isinstance(self.ops, list)
        assert isinstance(self.concat, Merger)
        assert isinstance(self.inputs, Inputs)
        assert len(Block._operations) == len(self.ops)
        assert self.op_type in [op.type() for op in Block._operations]
        for op in self.ops:
            assert isinstance(op, Operation)

    @staticmethod
    def random_block(max_inputs):
        op = random.choice(Block._operations).type()
        ops = [operation.random() for operation in Block._operations]
        inputs = Block._inputs.random(max_inputs)
        print(inputs)
        concat = random.choice(Block._mergers)()
        return Block(operation_type=op, ops=ops, concatenation=concat, inputs=inputs)

    @staticmethod
    def simple_individual(max_inputs):
        op = random.choice(Block._operations).type()
        ops = [operation.random() for operation in Block._operations]
        inputs = np.zeros(max_inputs)
        inputs[-1] = 1
        inputs = Block._inputs(inputs)
        concat = random.choice(Block._mergers)()
        return Block(operation_type=op, ops=ops, concatenation=concat, inputs=inputs)

    def cross(self, other_chromosome):
        op_type = random.choice([self.op_type, other_chromosome.op_type])
        ops = [self.ops[i].cross(other_chromosome.ops[i]) for i in range(len(Block._operations))]
        concat = self.concat.cross(other_chromosome.concat)
        inputs = self.inputs.cross(other_chromosome.inputs)
        return Block(operation_type=op_type, ops=ops, concatenation=concat, inputs=inputs)

    def mutate(self):
        if np.random.rand() < Block._change_op_prob:
            self.op_type = random.choice(Block._operations).type()
        for op in self.ops:
            op.mutate()
        if np.random.rand() < Block._change_concat_prob:
            self.concat = random.choice(Block._mergers)()
        self.inputs.mutate()
        self.concat.mutate()

    def __repr__(self):
        c = "%s|" % self.op_type
        for op in self.ops:
            if op.type() == self.op_type:
                c = "||%s||" % op.__repr__()
        if self.inputs.are_more_than_one():
            c += "%s||" % self.concat.__repr__()
        else:
            c += "woCAT||"
        c += "%s||\n" % self.inputs.__repr__()
        return c

    def self_copy(self):
        op_type = self.op_type
        ops = [operation.self_copy() for operation in self.ops]
        concat = self.concat.self_copy()
        inputs = self.inputs.self_copy()
        return Block(operation_type=op_type, ops=ops, concatenation=concat, inputs=inputs)

    def get_inputs(self):
        return self.inputs.inputs

    def set_inputs(self, new_inputs):
        self.inputs.set(new_inputs)

    def decode(self, **kwargs):
        pass


class ChromosomeOp(Chromosome):

    _max_initial_blocks = 5
    _block = Block

    _grow_prob = 0.25
    _decrease_prob = 0.25

    initial_filters = 32

    def __init__(self, blocks, n_blocks):
        assert isinstance(blocks, list)
        assert n_blocks == len(blocks)
        for block in blocks:
            assert isinstance(block, Block)
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = blocks
        self.fix_connections()

    def get_matrix_connection(self):
        inputs = [list(block.get_inputs()) for block in self.blocks]
        for INPUT in inputs:
            INPUT += [0] * (self.n_blocks - len(INPUT))
            print("Matrix")
            print(np.array(inputs))
        return np.array(inputs)

    def set_connections_from_matrix(self, matrix):
        shape = matrix.shape
        if matrix.shape[0] == 0:
            print("Empty Inputs!")
            raise NameError
        h = shape[0]
        assert h == len(self.blocks)
        assert h == self.n_blocks
        for k in range(self.n_blocks):
            self.blocks[k].set_inputs(matrix[k, 0:k+1])

    def add_block(self, index_to_add):
        matrix = self.get_matrix_connection()
        matrix.flatten()
        m = np.where(matrix == 1)[0]
        id = m[np.random.randint(0, m.size)]



    def add_connections(self, index_to_add):
        # The new block born in a previous connection, so it adopts the input and the output of this connection
        # (the new block will have only one input and one output)
        n = self.n_blocks
        matrix = self.get_matrix_connection()
        row = np.zeros(n - 1)
        col = np.zeros(n)
        if n <= 1:
            matrix = np.ones(n).astype(np.int32)
        elif index_to_add == n - 1:
            matrix = np.insert(matrix, index_to_add, row, axis=0)
            matrix = np.insert(matrix, index_to_add, col, axis=1)
            matrix[index_to_add, index_to_add] = 1
        else:
            matrix = np.insert(matrix, index_to_add, row, axis=0)
            matrix = np.insert(matrix, index_to_add + 1, col, axis=1)
            matrix[index_to_add, index_to_add] = 1
            if index_to_add < n - 1:
                possible_output = [i for i in range(index_to_add+1, n) if matrix[i, index_to_add]==1]
                output = random.choice(possible_output)
                matrix[output, index_to_add] = 0
                matrix[output, index_to_add + 1] = 1
        self.set_connections_from_matrix(matrix)

    def test(self):
        c = ChromosomeOp.random_individual()
        return c

    def delete_connections(self, index_to_delete):
        matrix = self.get_matrix_connection()

        if matrix.shape[0] == 0:
            return
        deleted_row = max(index_to_delete - 1, 0)
        deleted_col = min(matrix.shape[0] - 1, index_to_delete)
        new_matrix = np.delete(matrix, deleted_row, axis=0)
        new_matrix = np.delete(new_matrix, deleted_col, axis=1)
        self.blocks.pop(index_to_delete)
        self.set_connections_from_matrix(new_matrix)
        self.fix_connections()

    def fix_connections(self):
        matrix = self.get_matrix_connection()
        print("Fixing")
        print(matrix)
        shape = matrix.shape
        if len(shape) == 1:
            matrix = np.ones(self.n_blocks).astype(np.int32)
            self.set_connections_from_matrix(matrix)
            return
        h, w = shape
        assert h == w
        for i in range(h):
            if np.sum(matrix[i, :]) == 0 or np.sum(matrix[:, i]) == 0:
                matrix[i, i] = 1
        self.set_connections_from_matrix(matrix)

    @classmethod
    def random_individual(cls):
        n_blocks = np.random.randint(0, cls._max_initial_blocks)
        print(f"Total blocks: {n_blocks}")
        blocks = []
        for i in range(n_blocks):
            blocks.append(cls._block.random_block(max_inputs=i))
            print(type(blocks[-1]))
            print(blocks[-1].inputs)
        return ChromosomeOp(blocks, n_blocks)

    def simple_individual(self):
        n_blocks = 0
        blocks = []
        return ChromosomeOp(blocks, n_blocks)

    def cross(self, other_chromosome):
        n_blocks = min(self.n_blocks, other_chromosome.N_blocks)
        random_chromosome_blocks = [l.self_copy() for l in random.choice([self.blocks, other_chromosome.blocks()])]
        for i in range(n_blocks):
            random_chromosome_blocks[i] = self.blocks[i].cross(other_chromosome.blocks[i])
        return ChromosomeOp(random_chromosome_blocks, n_blocks)

    def mutate(self):
        # Add a new block after a random one
        if np.random.rand() < self._grow_prob:
            if self.n_blocks == 0:
                self.blocks.append(self._block.random_block(max_inputs=0))
            else:
                # Choose a random block, copy it and add it next to the original one in one of its connections.
                # The last block doesn't have output connections, so a block can be added after it.
                index_to_add = int(np.random.randint(self.n_blocks))
                block_to_copy = self.blocks[index_to_add]
                new_block = block_to_copy.self_copy()
                self.add_connections(index_to_add)
                self.blocks.insert(index_to_add + 1, new_block)
            self.n_blocks += 1
            assert self.n_blocks == len(self.blocks)

        elif np.random.rand() < self._decrease_prob and self.n_blocks > 0:
            index_to_delete = int(np.random.randint(self.n_blocks))
            self.delete_connections(index_to_delete)
        # mutate each block
        for block in self.blocks:
            block.mutate()
        return

    def __repr__(self):
        r = ''
        for block in self.blocks:
            r += block.__repr__()
        return r

    def self_copy(self):
        blocks = [block.self_copy() for block in self.blocks]
        assert len(blocks) == self.n_blocks
        return ChromosomeOp(blocks, self.n_blocks)

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
            x_ = BatchNormalization()(x_)
            x_ = Dropout(self.nn_layers[i].dropout)(x_)
        x_ = Dense(num_classes, activation='softmax')(x_)
        return x_

    def get_cell(self, inp_, allow_maxpool, fp=32):
        return


    def decode(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):
        inp = Input(shape=input_shape)
        x = BatchNormalization()(inp)
        x = Conv2D(self.initial_filters, 1, activation='relu', padding='same')(x)
        x = Dropout(0.3)(x)
        for i in range(self.n_blocks):
            layers = self.get_cell(x, allow_maxpool=False, fp=fp)
            if len(layers) == 0:
                break
            if i == self.n_blocks - 1:
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

class IdentityLayer(Layer):
    def mutate(self):
        pass

    def self_copy(self):
        pass

    @classmethod
    def random_layer(cls):
        return IdentityLayer()

    def __repr__(self):
        return "Identity"

    def cross(self, other_layer):
        return random.choice([self, other_layer])

    def get_layer(self, x):
        return x


class OpCoding:
    _operations = ['CNN', 'Identity']
    _joints = ['concatenation', 'sum', 'average']
    _CNN_params = {'growing_factor_type': float,
                   'kernel_size_type': int,
                   'activation_type': str,
                   'min_growing_factor': 0.5,
                   'max_growing_factor': 2.0,
                   'kernel_sizes': [1, 3, 5, 7],
                   'activations': ['relu', 'sigmoid', 'tanh', 'elu', 'prelu', 'leakyreLu']}

    def __init__(self, operation, cnn_grow_factor, cnn_kernel_size, cnn_activation, join_type, inputs):
        assert operation in OpCoding._operations
        assert join_type in OpCoding._joints
        assert cnn_kernel_size in OpCoding._CNN_params['kernel_sizes']
        assert cnn_activation in OpCoding._CNN_params['activations']
        assert isinstance(cnn_grow_factor, float)
        assert isinstance(inputs, list)
        assert cnn_grow_factor >= OpCoding._CNN_params['min_growing_factor']
        assert cnn_grow_factor <= OpCoding._CNN_params['max_growing_factor']

        self.op = operation
        self.cnn_grow_factor = cnn_grow_factor
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_activation = cnn_activation
        self.join = join_type
        self.inputs = inputs

    def get_code(self):
        CODE = []
        op_type = OpCoding._operations.index(self.op)
        cnn_grow_factor = self.cnn_grow_factor
        cnn_kernel_size = self.cnn_kernel_size
        cnn_activation = OpCoding._CNN_params['activations'].index(self.cnn_activation)
        join = OpCoding._joints.index(self.join)
        inputs = self.inputs
        CODE.append(op_type)
        CODE.append(cnn_grow_factor)
        CODE.append(cnn_kernel_size)
        CODE.append(cnn_activation)
        CODE.append(join)
        CODE += inputs
        return CODE


class ChromosomeGrow(ChromosomeCNN):
    skip_prob = 0.1
    allow_maxpool = True
    union_operations = ['concat', 'add']

    def __init__(self, cnn_layers, nn_layers, connections, concatenations=None):
        assert isinstance(cnn_layers, list)
        assert isinstance(nn_layers, list)
        assert isinstance(concatenations, list)
        assert len(cnn_layers) == len(concatenations)
        super().__init__()
        self.connections = connections
        self.cnn_layers = cnn_layers
        self.nn_layers = nn_layers
        self.concatenations = concatenations

        if not isinstance(self.connections, Connections):
            print("Problems with type", type(self.connections))
            print(self.connections)

    def verify_codification(self):
        matrix_connections = self.connections.matrix
        nodes_with_concat = np.sum(matrix_connections, axis=1) > 1

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


class ChromosomeCoding(Chromosome):

    _initial_max_operations = 4

    def __init__(self, operations_codings):
        super().__init__()
        assert isinstance(operations_codings, list)

    @staticmethod
    def random_individual():
        n_operations = np.random.randint(0, ChromosomeCoding._initial_max_operations + 1)
        #ops = [OpCoding.random_code() for i in range(n_operations)]
        #matrix_connection = ChromosomeCoding.get_matrix_connections(ops)
        #matrix_connection = ChromosomeCoding.veryfy_and_fix_connections(matrix_connection)

    def simple_individual(self):
        pass

    def cross(self, other_chromosome):
        pass

    def mutate(self):
        pass

    def __repr__(self):
        pass

    def fitness(self, test=False):
        pass

    def self_copy(self):
        pass
