import random
import numpy as np

import keras
from keras.callbacks import ModelCheckpoint
from utils.utils import LinearScheduler, EarlyStopByTimeAndAcc
from utils.codifications import Chromosome
from utils.codification_cnn import FitnessCNN
from utils.codification_ops import AbstractGen, Inputs, Operation, CNN, Identity
from utils.utils import lr_schedule

from keras.layers import PReLU, LeakyReLU, AveragePooling2D
from keras import Input, Model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, BatchNormalization, add, GlobalAveragePooling2D

from keras.layers.merge import concatenate


class HyperParams(AbstractGen):
    _type = "HyperParams"
    _GROW_RATE_LIMITS = [2., 5.]
    _N_CELLS = [1, 2]
    _N_BLOCKS = [2]
    _STEM = [16, 32, 45]
    _LR_LIMITS = [-9, -2]
    _MAX_WU = 0.2
    mutation_prob = 0.2
    lr = -4.32
    warmup = 0.5

    def __init__(self, grow_rate, n_cells, n_blocks, stem, lr, warmup, **kwargs):
        super().__init__()
        self.grow_rate = grow_rate
        self.n_cells = n_cells
        self.n_blocks = n_blocks
        self.stem = stem
        self.lr = lr
        self.warmup = min(warmup, self._MAX_WU)

    def cross(self, other_gen):
        grow_rate = self.cross_floats(self.grow_rate, other_gen.grow_rate)
        lr = self.cross_floats(self.lr, other_gen.lr)
        n_cells = self.choice(self.n_cells, other_gen.n_cells)
        n_blocks = self.choice(self.n_blocks, other_gen.n_blocks)
        stem = self.choice(self.stem, other_gen.stem)
        warmup = self.cross_floats(self.warmup, other_gen.warmup)
        return HyperParams(grow_rate, n_cells, n_blocks, stem, lr, warmup)

    @staticmethod
    def cross_floats(value1, value2):
        b = np.random.rand()
        return b * value1 + (1 - b) * value2

    @staticmethod
    def choice(a, b=None):
        return random.choice(a) if b is None else random.choice([a, b])

    def decode(self, **kwars):
        lr = 2. ** self.lr
        return self.grow_rate, self.n_cells, self.n_blocks, self.stem, lr, self.warmup

    @classmethod
    def random(cls, **kwargs):
        grow_rate = cls.random_float(cls._GROW_RATE_LIMITS)
        lr = cls.random_float(cls._LR_LIMITS)
        w = np.random.rand() * cls._MAX_WU
        return HyperParams(grow_rate, cls.choice(cls._N_CELLS), cls.choice(cls._N_BLOCKS), cls.choice(cls._STEM), lr, w)

    @staticmethod
    def random_float(limits):
        y1, y2 = limits
        x1, x2 = 0., 1.
        x = np.random.rand()
        return x * (y2 - y1) + y1  # x * (y2 - y1) / (x2 - x1) + y1

    def mutate(self):
        if np.random.rand() < self.mutation_prob:
            self.grow_rate = self.gauss_mutation(self.grow_rate, self._GROW_RATE_LIMITS[1], self._GROW_RATE_LIMITS[0],
                                                 int_=False)
        if np.random.rand() < self.mutation_prob:
            self.lr = self.gauss_mutation(self.lr, self._LR_LIMITS[1], self._LR_LIMITS[0], int_=False)
        if np.random.rand() < self.mutation_prob:
            self.n_cells = self.choice(self._N_CELLS)
        if np.random.rand() < self.mutation_prob:
            self.n_blocks = self.choice(self._N_BLOCKS)
        if np.random.rand() < self.mutation_prob:
            self.stem = self.choice(self._STEM)
        if np.random.rand() < self.mutation_prob:
            self.warmup = np.random.rand() * self._MAX_WU

    def __repr__(self):
        _, _, _, _, lr, _ = self.decode()
        return "HP->|GR:%0.2f|CELL:%d|BLOCK:%d|STEM:%d|LR:%0.4f|WU:%0.1f\n" % \
               (self.grow_rate, self.n_cells, self.n_blocks, self.stem, lr, self.warmup)

    def self_copy(self):
        return HyperParams(self.grow_rate, self.n_cells, self.n_blocks, self.stem, self.lr, self.warmup)


class Merger(AbstractGen):
    _type = "Merge"
    _projection_type = ['normal', 'extend', 'zero-pad'][1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def projection_extend(self, input_tensor, n_features):
        input_features = input_tensor._shape_as_list()[-1]
        if n_features == input_features:
            return input_tensor
        elif n_features > input_features:
            features_diff = n_features - input_features
            features = self.projection_normal(input_tensor, features_diff)
            return concatenate([input_tensor, features])
        else:  # n_features < input_features
            return self.projection_normal(input_tensor, n_features)

    # TODO: Implement Zero padding channel
    def zero_pading(self, input_tensor, n_features):
        input_features = input_tensor._shape_as_list()[-1]
        if n_features == input_features:
            return input_tensor
        elif n_features > input_features:
            features_diff = n_features - input_features
            features = self.projection_normal(input_tensor, features_diff)
            raise NotImplementedError
            return concatenate([input_tensor, features])
        else:  # n_features < input_features
            return self.projection_normal(input_tensor, n_features)

    def projection_normal(self, input_tensor, n_features):
        init = l2(1e-5)
        x_ = BatchNormalization()(input_tensor)
        x_ = Conv2D(n_features, 1, padding='same', activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=init)(x_)
        return x_

    def projection(self, input_tensor, n_features):
        if self._projection_type == 'normal':
            return self.projection_normal(input_tensor, n_features)
        elif self._projection_type == 'extend':
            return self.projection_extend(input_tensor, n_features)
        else:
            raise NotImplementedError

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

    def decode(self, input_tensors, features_out):
        if len(input_tensors) <= 1:
            return input_tensors[0]
        else:
            return concatenate(input_tensors)


class Sum(Merger):
    _type = "SUM"

    def __init__(self):
        super().__init__()

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        return Sum()

    def decode(self, input_tensors, features_out):
        if len(input_tensors) <= 1:
            return input_tensors[0]
            return self.projection(input_tensors[0], features_out)
        # else
        # List with number of channels of each input tensor
        channels_count = [input_tensor._shape_as_list()[-1] for input_tensor in input_tensors]
        projected_tensors = []
        for i, input_tensor in enumerate(input_tensors):
            if channels_count[i] != features_out:
                projected_tensors.append(self.projection(input_tensor, features_out))
            else:
                projected_tensors.append(input_tensor)
        # projected_tensors = [self.projection(input_tensor, min_channels) for input_tensor in input_tensors]
        return add(projected_tensors)


class MaxPooling(Operation):
    _type = 'MP'
    _admited_sizes = [2, 3, 5]

    def __init__(self, size=2, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else
        new_size = random.choice([self.size, other_operation.size])
        return MaxPooling(new_size)

    def decode(self, input_tensor,  features_out):
        return MaxPooling2D(self.size, 1, padding='same')(input_tensor)

    def __repr__(self):
        return "%s|%d" % (self._type, self.size)

    def mutate(self):
        return random.choice(self._admited_sizes)

    def self_copy(self):
        return MaxPooling(self.size)

    @classmethod
    def random(cls):
        new_size = random.choice(cls._admited_sizes)
        return MaxPooling(new_size)


class AvPooling(Operation):
    _type = 'AP'
    _admited_sizes = [2, 3, 5]

    def __init__(self, size=2, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def cross(self, other_operation):
        # if self.type() != other_merger.type():
        super().cross(other_operation)
        # else            n_inputs = len(input_tensors)

        new_size = random.choice([self.size, other_operation.size])
        return AvPooling(new_size)

    def decode(self, input_tensor,  features_out):
        return AveragePooling2D(self.size, 1, padding='same')(input_tensor)

    def __repr__(self):
        return "%s|%d" % (self._type, self.size)

    def self_copy(self):
        return AvPooling(self.size)

    @classmethod
    def random(cls):
        new_size = random.choice(cls._admited_sizes)
        return AvPooling(new_size)


class IdentityGrow(Identity):
    def decode(self, input_tensors, features_out):
        return input_tensors


class CNNGrow(CNN):
    def decode(self, input_tensor, features_out):
        activation = self.activation
        filters = int(self.filter_mul * features_out)
        k_size = self.k_size
        dropout = self.round_dropout(self.dropout)
        init = l2(1e-5)

        # Return a BN-Dropout-Conv-Activation layer
        x = BatchNormalization()(input_tensor)
        x = Dropout(dropout)(x)
        if k_size > 3 and False:
            conv_type = self._conv_type5
        else:
            conv_type = self._conv_type
        if activation in ['relu', 'sigmoid', 'tanh', 'elu']:
            x = conv_type(filters, k_size, activation=activation, padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=init)(x)
        else:  # activation == 'prelu':
            x = conv_type(filters, k_size, padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=init)(x)
            x = PReLU()(x)
        return x


class OperationBlock(AbstractGen):
    _operations = [CNNGrow, IdentityGrow]  # , MaxPooling, AvPooling]
    _mergers = [Concatenation, Sum]
    _inputs = Inputs

    _change_op_prob = 0.1
    _change_concat_prob = 0.1

    def __init__(self, operation_type, ops, concatenation, inputs):
        super().__init__()
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
        return OperationBlock(operation_type=op_type, ops=ops, concatenation=concat, inputs=inputs)

    def mutate(self):
        if np.random.rand() < OperationBlock._change_op_prob:
            self.op_type = random.choice(OperationBlock._operations).type()
        for op in self.ops:
            op.mutate()
        if np.random.rand() < OperationBlock._change_concat_prob:
            self.concat = random.choice(OperationBlock._mergers)()
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

    def decode(self, all_input_tensors, features_out, **kwargs):
        input_tensors = self.inputs.decode(all_input_tensors)  # Select the inputs from all the possible inputs
        merged_tensors = self.concat.decode(input_tensors, features_out)
        operation = self.get_operation()  # Return the output tensor of the selected operation
        return operation.decode(merged_tensors, features_out)

    def decode_with_single_input_tensor(self, input_tensor):
        operation = self.get_operation()  # Return the output tensor of the selected operation
        return operation.decode(input_tensor)


class ChromosomeGrow(Chromosome):
    _max_initial_blocks = 4
    _block = OperationBlock
    _HYPERPARAMS = HyperParams

    _grow_prob = 0.15
    _decrease_prob = 0.25

    def __init__(self, blocks, n_blocks, hparams):
        assert isinstance(blocks, list)
        assert isinstance(hparams, self._HYPERPARAMS)
        assert n_blocks == len(blocks)
        for block in blocks:
            assert isinstance(block, OperationBlock)
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = blocks
        self.hparams = hparams
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
            self.blocks[k].set_inputs(matrix[k, 0:k + 1])

    def add_block(self, out_block_index, new_block):
        # Add a block before the given out_block_index
        matrix = self.get_matrix_connection()
        n = matrix.shape[0]
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
        # The new block born in a previous connectiself.grow_rate, self.n_cells, self.n_blocks, self.stemon, so it adopts the input and the output of this connection
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
                possible_output = [i for i in range(index_to_add + 1, n) if matrix[i, index_to_add] == 1]
                output = random.choice(possible_output)
                matrix[output, index_to_add] = 0
                matrix[output, index_to_add + 1] = 1
        self.set_connections_from_matrix(matrix)

    def test(self):
        c = self.__class__.random_individual()
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
        n_blocks = np.random.randint(cls._max_initial_blocks - 2, cls._max_initial_blocks)
        # print(f"Total blocks: {n_blocks}")
        blocks = []
        for i in range(n_blocks):
            blocks.append(cls._block.random_block(max_inputs=i))
            # print(type(blocks[-1]))
            # print(blocks[-1].inputs)
        hparams = cls._HYPERPARAMS.random()
        return cls(blocks, n_blocks, hparams)

    def simple_individual(self):
        n_blocks = 0
        blocks = []
        hparams = self._HYPERPARAMS.random()
        return self.__class__(blocks, n_blocks, hparams)

    def cross(self, other_chromosome):
        n_blocks = min(self.n_blocks, other_chromosome.n_blocks)
        random_chromosome_blocks = [l.self_copy() for l in random.choice([self.blocks, other_chromosome.blocks])]
        for i in range(n_blocks):
            random_chromosome_blocks[i] = self.blocks[i].cross(other_chromosome.blocks[i])
        hparams = self.hparams.cross(other_chromosome.hparams)
        return self.__class__(random_chromosome_blocks, len(random_chromosome_blocks), hparams)

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
        self.hparams.mutate()
        # print(self.get_matrix_connection())
        return

    def __repr__(self):
        r = ''
        for block in self.blocks:
            r += block.__repr__()
        r += self.hparams.__repr__()
        return r

    def learning_rate(self):
        _, _, _, _, lr, _ = self.hparams.decode()
        return lr

    def warmup_epochs(self, total_epochs):
        _, _, _, _, _, warmup = self.hparams.decode()
        return np.round(warmup * total_epochs).astype(np.int32)

    def decode(self, input_shape, num_classes=10, verb=False, fp=32, **kwargs):
        grow_rate, n_cells, n_blocks, stem, lr, warmup = self.hparams.decode()
        total_grow = grow_rate ** n_blocks

        inp = Input(shape=input_shape)
        x = Conv2D(stem, 3, activation='relu', padding='same')(inp)

        total_ops = self.n_blocks * n_cells * n_blocks
        op_i = 1
        for block_i in range(n_blocks):
            if block_i > 0:
                x = MaxPooling2D(2, 2)(x)
                pass
            for cell_i in range(n_cells):
                input_tensors = [x]
                for block in self.blocks:
                    features_out = int(np.round(stem * total_grow ** (op_i / total_ops)))
                    input_tensors.append(block.decode(input_tensors, features_out))
                    op_i += 1
                x = input_tensors[-1]
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
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
        return self.__class__(blocks, self.n_blocks, self.hparams.self_copy())


#class ChromosomeOp(ChromosomeGrow):
#    def __init(self, blocks, n_blocks, hparams):
#        super().__init__(blocks, n_blocks, hparams)

class FitnessGrow(FitnessCNN):

    def set_callbacks(self, file_model=None, epochs=None):
        #epochs = self.epochs
        # Create the Learning rate scheduler.
        total_steps = np.round(epochs * self.y_train.shape[0] / self.batch_size).astype(np.int32)
        warmup_steps = np.round(self.warmup_epochs * self.y_train.shape[0] / self.batch_size).astype(np.int32)
        schedule = LinearScheduler(max_lr=self.learning_rate_base,
                                   min_lr=0.00002,
                                   total_steps=total_steps,
                                   warmup_steps=warmup_steps,
                                   verbose=False)
        if self.reduce_plateu:
            schedule = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=True)
        min_val_acc = (1. / self.num_clases) + 0.1
        callbacks = [schedule]
        if not self.test:
            early_stop = EarlyStopByTimeAndAcc(limit_time=900,
                                           baseline=min_val_acc,
                                           patience=epochs//2)
            callbacks.append(early_stop)
        val_acc = 'val_accuracy' if keras.__version__ == '2.3.1' else 'val_acc'        
        if file_model is not None:
            return callbacks
            checkpoint_acc = ModelCheckpoint(file_model, monitor=val_acc, save_best_only=True)
            callbacks.append(checkpoint_acc)
        return callbacks

    def get_params(self, chromosome, precise_mode=False, test=False):
        #epochs = super().get_params(chromosome, precise_mode, test)
        if hasattr(chromosome, 'learning_rate'):
            self.learning_rate_base = chromosome.learning_rate()
        if hasattr(chromosome, 'warmup_epochs'):
            epochs = self.epochs
            self.warmup_epochs = chromosome.warmup_epochs(epochs)
        epochs = super().get_params(chromosome, precise_mode, test)
        print("epochs: %d. warmup epochs: %d" % (epochs, self.warmup_epochs))
        return epochs
