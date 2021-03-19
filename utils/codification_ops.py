import random
import numpy as np

from config import config
from utils.codifications import Chromosome
from utils.codification_cnn import ChromosomeCNN
from utils.codifications import Layer

from keras.layers import PReLU, LeakyReLU
from keras import Input, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, BatchNormalization, add, GlobalAveragePooling2D, SeparableConv2D
from keras.layers.merge import concatenate
from keras.regularizers import l2


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
        raise NotImplementedError

    def mutate(self):
        pass

    def __repr__(self):
        return self._type

    def self_copy(self):
        return self.random()

    def decode(self, **kgars):
        raise NotImplementedError

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


class Inputs(AbstractGen):
    _type = "Inputs"
    _mutate_prob = config.inputs_mutate_prob

    def __init__(self, inputs_array, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(inputs_array, np.ndarray)
        assert len(inputs_array.shape) <= 1
        self.inputs = inputs_array

    @classmethod
    def random(cls, max_inputs):
        if max_inputs == 0:
            return Inputs(np.array([]))
        inputs = np.random.randint(0, 2, max_inputs)  # build a vector with "max_inputs" possible inputs
        if np.sum(inputs) == 0:  # There must be at least one input
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
        _repr = ""
        for INPUT in inputs:
            _repr += str(INPUT)
        return _repr

    def self_copy(self):
        return Inputs(self.inputs.copy())

    def set(self, new_inputs):
        assert isinstance(new_inputs, np.ndarray)
        assert len(new_inputs.shape) < 2
        self.inputs = new_inputs.copy()

    def are_more_than_one(self):
        return np.sum(self.inputs) > 1

    def decode(self, tensors_list):
        input_tensors = []
        for i in range(len(self.inputs)):
            if self.inputs[i] == 1:
                input_tensors.append(tensors_list[i])
        return input_tensors


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
    _conv_type = [Conv2D, SeparableConv2D][1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def projection(self, input_tensor, n_features):
        init = None #l2(1e-4)
        x_ = self._conv_type(n_features, 1, padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=init)(input_tensor)
        x_ = BatchNormalization()(x_)
        return x_

    def cross(self, other_merger):
        assert isinstance(other_merger, Merger)
        if self.type() != other_merger.type():
            selected = random.choice([self, other_merger])
            return selected


class Concatenation(Merger):
    _type = "CAT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return Concatenation()

    def decode(self, input_tensors):        
        if len(input_tensors) <= 1:
            return input_tensors[0]
        return concatenate(input_tensors)
        # List with number of channels of each input tensor
        channels_count = [input_tensor._shape_as_list()[-1] for input_tensor in input_tensors]
        #min_channels = int(np.min(channels_count))
        #channels = int(min_channels / )
        projected_tensors = []
        for i, input_tensor in enumerate(input_tensors):
            channels = int(channels_count[i] / 2)
            projected_tensors.append(self.projection(input_tensor, channels))
        # projected_tensors = [self.projection(input_tensor, channels) for input_tensor in input_tensors]
        return concatenate(projected_tensors)


class Sum(Merger):
    _type = "SUM"

    def __init__(self):
        super().__init__()

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return Sum()

    def decode(self, input_tensors):
        if len(input_tensors) <= 1:
            return input_tensors[0]
        # List with number of channels of each input tensor
        channels_count = [input_tensor._shape_as_list()[-1] for input_tensor in input_tensors]
        m_channels = int(np.mean(channels_count))
        projected_tensors = []
        for i, input_tensor in enumerate(input_tensors):
            if channels_count[i] != m_channels:
                projected_tensors.append(self.projection(input_tensor, m_channels))
            else:
                projected_tensors.append(input_tensor)
        # projected_tensors = [self.projection(input_tensor, min_channels) for input_tensor in input_tensors]
        return add(projected_tensors)


class Identity(Operation):
    _type = "Identity"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return self.__class__()

    def decode(self, input_tensors):
        return input_tensors


class MaxPooling(Operation):
    _type = 'Maxpool'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return MaxPooling()

    def decode(self, input_tensor):
        return MaxPooling2D(3, 1, padding='same')(input_tensor)


class CNN(Operation):
    possible_activations = config.possible_activations ['relu', 'elu', 'prelu']
    filters_mul_range = config.filters_mul_range
    dropout_range = config.dropout_range
    _dropout_precision = 0.05
    _possible_drops = np.arange(dropout_range[0], dropout_range[1] + 1e-7, _dropout_precision)
    # possible_k = [1, 3, 5, 7]
    possible_k = config.possible_k
    k_prob = config.k_prob
    drop_prob = config.drop_prob
    filter_prob = config.filter_prob
    act_prob = config.act_prob
    _type = 'CNN'
    _conv_type = [Conv2D, SeparableConv2D][0]
    _conv_type5 = [Conv2D, SeparableConv2D][1]

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

    def round_dropout(self, dropout):
        return self._possible_drops[np.argmin(np.abs(dropout - self._possible_drops))]

    def cross(self, other_operation):
        super().cross(other_operation)
        new_filter_mul = self.cross_filter_mul(other_operation.filter_mul)
        new_k_size = self.cross_kernel(other_operation.k_size)
        new_activation = self.cross_activation(other_operation.activation)
        new_dropout = self.cross_dropout(other_operation.dropout)
        return self.__class__(filter_mul=new_filter_mul, kernel_size=new_k_size,
                              activation=new_activation, dropout=new_dropout)

    def cross_dropout(self, other_dropout):
        b = np.random.rand()
        new_dropout = self.dropout * (1 - b) + b * other_dropout
        return new_dropout

    def cross_filter_mul(self, other_filters_mul):
        b = np.random.rand()
        return self.filter_mul * (1 - b) + other_filters_mul * b

    def cross_kernel(self, other_kernel):
        k = random.choice([self.k_size, other_kernel])
        return k

    def cross_activation(self, other_activation):
        return random.choice([self.activation, other_activation])

    def mutate(self):
        aleatory = np.random.rand(5)
        if aleatory[0] < self.filter_prob:
            self.filter_mul = self.gauss_mutation(self.filter_mul, self.filters_mul_range[1],
                                                  self.filters_mul_range[0], int_=False)
        if aleatory[1] < self.act_prob:
            self.activation = random.choice(self.possible_activations)
        if aleatory[2] < self.drop_prob:
            new_dropout = self.gauss_mutation(self.dropout, self.dropout_range[1], self.dropout_range[0], int_=False)
            self.dropout = new_dropout
        if aleatory[4] < self.k_prob:
            self.k_size = random.choice(self.possible_k)

    def __repr__(self):
        dropout = self.round_dropout(self.dropout)
        return "%s|F:%0.1f|K:%d|A:%s|D:%0.2f" % (self.type(), self.filter_mul, self.k_size, self.activation, dropout)

    def self_copy(self):
        return self.__class__(filter_mul=self.filter_mul, kernel_size=self.k_size,
                              activation=self.activation, dropout=self.dropout)

    def decode(self, input_tensor):
        activation = self.activation
        filter_mul = self.filter_mul
        k_size = self.k_size
        dropout = self.round_dropout(self.dropout)

        input_features = input_tensor._shape_as_list()[-1]
        filters = int(filter_mul * input_features)
        init = None  # l2(1e-4)

        # Return a Conv-Activation-BN-Dropout layer
        if activation in ['relu', 'sigmoid', 'tanh', 'elu']:
            x = self._conv_type(filters, k_size, activation=activation, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=init)(input_tensor)
        elif activation == 'prelu':
            x = self._conv_type(filters, k_size, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=init)(input_tensor)
            x = PReLU()(x)
        else:
            x = self._conv_type(filters, k_size, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=init)(input_tensor)
            x = LeakyReLU()(x)

        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        return x


class OperationBlock:
    _operations = [Identity, CNN, MaxPooling]
    _mergers = [Concatenation, Sum]
    _inputs = Inputs

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
        assert len(OperationBlock._operations) == len(self.ops)
        assert self.op_type in [op.type() for op in OperationBlock._operations]
        for op in self.ops:
            assert isinstance(op, Operation)

    @staticmethod
    def random_block(max_inputs):
        op = random.choice(OperationBlock._operations).type()
        ops = [operation.random() for operation in OperationBlock._operations]
        inputs = OperationBlock._inputs.random(max_inputs)
        #print(inputs)
        concat = random.choice(OperationBlock._mergers)()
        return OperationBlock(operation_type=op, ops=ops, concatenation=concat, inputs=inputs)

    @staticmethod
    def simple_individual(max_inputs):
        op = random.choice(OperationBlock._operations).type()
        ops = [operation.random() for operation in OperationBlock._operations]
        inputs = np.zeros(max_inputs)
        inputs[-1] = 1
        inputs = OperationBlock._inputs(inputs)
        concat = random.choice(OperationBlock._mergers)()
        return OperationBlock(operation_type=op, ops=ops, concatenation=concat, inputs=inputs)

    def cross(self, other_chromosome):
        op_type = random.choice([self.op_type, other_chromosome.op_type])
        ops = [self.ops[i].cross(other_chromosome.ops[i]) for i in range(len(OperationBlock._operations))]
        concat = self.concat.cross(other_chromosome.concat)
        inputs = self.inputs.cross(other_chromosome.inputs)
        return self.__class__(operation_type=op_type, ops=ops, concatenation=concat, inputs=inputs)

    def mutate(self):
        if np.random.rand() < self.__class__._change_op_prob:
            self.op_type = random.choice(self.__class__._operations).type()
        for op in self.ops:
            op.mutate()
        if np.random.rand() < self.__class__._change_concat_prob:
            self.concat = random.choice(self.__class__._mergers)()
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
        return OperationBlock(operation_type=op_type, ops=ops, concatenation=concat, inputs=inputs)

    def get_inputs(self):
        return self.inputs.inputs

    def set_inputs(self, new_inputs):
        self.inputs.set(new_inputs)

    def get_operation(self):
        for op in self.ops:
            if op.type() == self.op_type:
                return op

    def decode(self, all_input_tensors, **kwargs):
        input_tensors = self.inputs.decode(all_input_tensors)  # Select the inputs from all the possible inputs
        merged_tensors = self.concat.decode(input_tensors)  # Merge the input tensors and return a single tensor
        operation = self.get_operation()  # Return the output tensor of the selected operation
        return operation.decode(merged_tensors)

    def decode_with_single_input_tensor(self, input_tensor):
        operation = self.get_operation()  # Return the output tensor of the selected operation
        return operation.decode(input_tensor)


class ChromosomeOp(Chromosome):
    _max_initial_blocks = 5
    _block = OperationBlock

    _grow_prob = 0.15
    _decrease_prob = 0.25

    INITIAL_FILTERS = 45
    CELLS_PER_BLOCK = 2
    N_BLOCKS = 2

    def __init__(self, blocks, n_blocks):
        assert isinstance(blocks, list)
        assert n_blocks == len(blocks)
        for block in blocks:
            assert isinstance(block, OperationBlock)
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = blocks
        self.fix_connections()

    def get_matrix_connection(self):
        inputs = [list(block.get_inputs()) for block in self.blocks]
        for INPUT in inputs:
            INPUT += [0] * (self.n_blocks - len(INPUT))
            # print("Matrix")
            # print(np.array(inputs))
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

    def add_block(self, out_block_index, new_block):
        # Add a block before the given out_block_index
        matrix = self.get_matrix_connection()
        n = matrix.shape [0]
        row = np.zeros(n)
        col = np.zeros(n + 1)

        inputs = matrix[out_block_index, :]  # possible inputs of the out_block
        inputs_index = np.where(inputs == 1)[0]  # activated inputs of the out_block
        random_input_index = np.random.choice(inputs_index)  # random input of the out_block

        matrix = np.insert(matrix, out_block_index, row, axis=0)  # add the row (inputs) associated to new block
        matrix = np.insert(matrix, out_block_index + 1, col, axis=1)  # add the col (outputs) associated to new block
        matrix[out_block_index + 1, out_block_index + 1] = 1  # the new block is connected to the out_block

        # connect the previous selected input to the new block, and disconnect from the out_block
        matrix[out_block_index, random_input_index] = 1
        matrix[out_block_index + 1, random_input_index] = 0
        self.blocks.insert(out_block_index, new_block)
        self.n_blocks = len(self.blocks)
        self.set_connections_from_matrix(matrix)

    def delete_block(self):
        matrix = self.get_matrix_connection()
        # First, try to delete a block that has only one input and one output, or just one input
        # And replace it with a connection
        total_inputs = np.sum(matrix, axis=1)
        total_outputs = np.sum(matrix, axis=0)[1::]
        total_outputs = np.append(total_outputs, 1)  # The last block doest have outputs, so virtual adding it
        available_to_delete = total_outputs * total_inputs  # only one input and one output, or just one input
        available_id = np.where(available_to_delete == 1)[0]
        if available_id.size > 0:
            id_to_delete = random.choice(available_id)  # block to delete
            if id_to_delete < matrix.shape[0]:  # If the selected block is not the last one
                matrix[id_to_delete + 1, id_to_delete] = 1  # Connect the input and the output of the block to delete
                matrix = np.delete(matrix, id_to_delete + 1, axis=1)
                matrix = np.delete(matrix, id_to_delete, axis=0)
            else:
                matrix = np.delete(matrix, id_to_delete, axis=1)
                matrix = np.delete(matrix, id_to_delete, axis=0)

            self.blocks.pop(id_to_delete)
            self.n_blocks = len(self.blocks)
            self.set_connections_from_matrix(matrix)
        else:
            #  If all blocks have more than one input or more than one connections
            self.delete_connections()

    def add_connections(self, index_to_add):
        # The new block born in a previous connection, so it adopts the input and the output of this connection
        # (the new block will have only one input and one output)
        n = self.n_blocks
        matrix = self.get_matrix_connection()
        row = np.zeros(n)
        col = np.zeros(n + 1)
        if n <= 1:
            matrix = np.ones(n).astype(np.int32)
        elif index_to_add == n:
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

    def delete_connections(self):
        # Delete a random connection
        matrix = self.get_matrix_connection()
        all_connections = np.where(matrix == 1)
        n_connections = len(all_connections[0])
        ids = np.arange(n_connections)
        np.random.shuffle(ids)
        for i in ids:
            row = all_connections[0][i]
            col = all_connections[1][i]
            if np.sum(matrix[row, :]) > 1 and np.sum(matrix[:, col]) > 1:
                matrix[row, col] = 0
                self.set_connections_from_matrix(matrix)
                return

    def fix_connections(self):
        matrix = self.get_matrix_connection()
        # print("Fixing")
        # print(matrix)
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
        n_blocks = np.random.randint(1, cls._max_initial_blocks)
        # print(f"Total blocks: {n_blocks}")
        blocks = []
        for i in range(n_blocks):
            blocks.append(cls._block.random_block(max_inputs=i))
            # print(type(blocks[-1]))
            # print(blocks[-1].inputs)
        return ChromosomeOp(blocks, n_blocks)

    def simple_individual(self):
        n_blocks = 0
        blocks = []
        return ChromosomeOp(blocks, n_blocks)

    def cross(self, other_chromosome):
        n_blocks = min(self.n_blocks, other_chromosome.n_blocks)
        random_chromosome_blocks = [l.self_copy() for l in random.choice([self.blocks, other_chromosome.blocks])]
        for i in range(n_blocks):
            random_chromosome_blocks[i] = self.blocks[i].cross(other_chromosome.blocks[i])
        return ChromosomeOp(random_chromosome_blocks, len(random_chromosome_blocks))

    def mutate(self):
        # Add a new block after a random one
        if np.random.rand() < self._grow_prob:
            if self.n_blocks == 0:
                self.blocks.append(self._block.random_block(max_inputs=0))
            else:
                # Choose a random block, copy it and add it back to the original one in one of its inputs.
                # The last first doesn't have input connections, so a block cant be added before it.
                index_to_add = int(np.random.randint(self.n_blocks))
                block_to_copy = self.blocks[index_to_add]
                new_block = block_to_copy.self_copy()
                self.add_block(index_to_add, new_block)
            assert self.n_blocks == len(self.blocks)

        elif np.random.rand() < self._decrease_prob and self.n_blocks > 0:
            index_to_delete = int(np.random.randint(self.n_blocks))
            self.delete_connections()
        # mutate each block
        for block in self.blocks:
            block.mutate()
        self.fix_connections()
        #print(self.get_matrix_connection())
        return

    def __repr__(self):
        r = ''
        for block in self.blocks:
            r += block.__repr__()
        return r

    def decode(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):
        inp = Input(shape=input_shape)
        x = BatchNormalization()(inp)
        x = Conv2D(self.INITIAL_FILTERS, 3, activation='relu', padding='same')(inp)

        for block_i in range(self.N_BLOCKS):
            if block_i > 0:
                x = MaxPooling2D(2, 2)(x)
            for cell_i in range(self.CELLS_PER_BLOCK):
                input_tensors = [x]
                for block in self.blocks:
                    input_tensors.append(block.decode(input_tensors))
                x = input_tensors[-1]

        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inp, outputs=x)
        if verb:
            model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model

    def self_copy(self):
        blocks = [block.self_copy() for block in self.blocks]
        assert len(blocks) == self.n_blocks
        return ChromosomeOp(blocks, self.n_blocks)

    def decode_deprecated(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):
        inp = Input(shape=input_shape)
        x = BatchNormalization()(inp)
        x = Conv2D(self.INITIAL_FILTERS, 1, activation='relu', padding='same')(x)
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





