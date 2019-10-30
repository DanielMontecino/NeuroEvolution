import random
import numpy as np
from utils.codifications import Fitness, Chromosome
import sys
sys.path.append('../../nasbench')
from nasbench import api


class ChromosomeNASBench(Chromosome):
    INPUT = 'input'
    OUTPUT = 'output'
    CONV3X3 = 'conv3x3-bn-relu'
    CONV1X1 = 'conv1x1-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'

    _operations = [CONV3X3, CONV1X1, MAXPOOL3X3]
    _change_op_prob = 0.1
    _grow_prob = 0.08
    _decrease_prob = 0.1
    _connection_prob = 0.5
    MAX_VERTICES = 7
    MAX_EDGES = 9
    """
    Connections are in the form:
    
                  INP    | _0_ |  _1_  |  _1_  |  _0_  |  _1_  |  _0_
                  BLOCK1 | ___ |  _0_  |  _0_  |  _1_  |  _1_  |  _0_
     OUTPUTS      BLOCK2 | ___ |  ___  |  _0_  |  _0_  |  _1_  |  _0_
                  ...    | ___ |  ___  |  ___  |  _0_  |  _1_  |  _1_
                  BLOCKN | ___ |  ___  |  ___  |  ___  |  _0_  |  _0_
                  OUT    | ___ |  ___  |  ___  |  ___  |  ___  |  _0_
                           INP   BLOCK1  BLOCK2  ...     BLOCKN   OUT
                           
                                     INPUTS
    
    """

    def decode(self, **kwargs):
        pass

    def __init__(self, operations, matrix):
        super().__init__()
        self.operations = operations
        self.matrix = matrix
        assert isinstance(matrix, np.ndarray)
        assert isinstance(operations, list)
        assert len(self.matrix.shape) == 2
        assert self.matrix.shape[0] == self.matrix.shape[1]
        self.n_ops = self.matrix.shape[0]
        assert self.n_ops >= 2
        assert len(self.operations) == self.n_ops
        assert self.operations[0] == self.INPUT
        assert self.operations[-1] == self.OUTPUT
        for op in self.operations[1:-1]:
            assert op in self._operations
        self.blocks_added = 0
        self.blocks_deleted = 0
        self.conn_added = 0
        self.conn_deleted = 0

        self.blocks_try_added = 0
        self.blocks_try_delete = 0
        self.fix_connections()

    def show_stats(self):
        print(f"Blocks added: {self.blocks_added}")
        print(f"Blocks deleted: {self.blocks_deleted}")
        print(f"Connections added: {self.conn_added}")
        print(f"Connections deleted: {self.conn_deleted}")
        print(f"Times a block should be added: {self.blocks_try_added}")
        print(f"Times a block should be deleted: {self.blocks_try_delete}")

    def fix_connections(self):
        blocks = self.matrix.shape[0]
        indices_lower_triangular = np.tril_indices(blocks)
        self.matrix[indices_lower_triangular] = 0  # All the elements in the lower triangle part of the matrix are zero
        for i in range(blocks - 1):
            if np.sum(self.matrix[i, :]) == 0 or np.sum(self.matrix[:, i + 1]) == 0:
                self.matrix[i, i + 1] = 1
        c = 0
        while len(self.operations) > self.MAX_VERTICES:
            self.delete_block()
            c += 1
            if c > 100:
                print("Error in while!")
                raise AssertionError
        c = 0
        conns = np.sum(self.matrix)
        while conns > self.MAX_EDGES:
            self.delete_connections()
            if np.sum(self.matrix) == conns:
                col = np.random.randint(2, blocks)
                self.matrix[0, col] = 0
                self.matrix[col - 1, col] = 1
            else:
                conns = np.sum(self.matrix)
            c += 1
            if c > 100:
                print("Error in while!")
                raise AssertionError

    def select_random_connection(self):
        ones_indices = np.where(self.matrix == 1)
        selected_id = np.random.randint(0, len(ones_indices[0]))
        output_block = ones_indices[0][selected_id]
        input_block = ones_indices[1][selected_id]
        return output_block, input_block  # row, col

    def add_block(self):
        output_block, input_block = self.select_random_connection()
        new_row = np.zeros(self.matrix.shape[0])
        new_col = np.zeros(self.matrix.shape[0] + 1)
        self.matrix = np.insert(self.matrix, input_block, new_row, axis=0)
        self.matrix = np.insert(self.matrix, input_block, new_col, axis=1)
        self.matrix[input_block, input_block + 1] = 1
        self.matrix[output_block, input_block] = 1
        self.matrix[output_block, input_block + 1] = 0

        new_operation = random.choice(self._operations)
        self.operations.insert(input_block, new_operation)
        self.blocks_added += 1

    def add_connection(self):
        aux_matrix = self.matrix.copy()
        tli = np.tril_indices(aux_matrix.shape[0])
        aux_matrix[tli] = 2
        zero_indices = np.where(aux_matrix == 0)
        if len(zero_indices[0]) == 0:
            return
        random_id = np.random.randint(0, len(zero_indices[0]))
        self.matrix[zero_indices[0][random_id], zero_indices[1][random_id]] = 1
        self.conn_added += 1

    def delete_block(self):
        if len(self.operations) == 2:
            return
        # First, try to delete a block that has only one input and one output, and replace it with a connection
        total_inputs = np.sum(self.matrix, axis=0)
        total_outputs = np.sum(self.matrix, axis=1)
        available_to_delete = total_outputs * total_inputs
        available_id = np.where(available_to_delete == 1)[0]
        if available_id.size > 0:
            id_to_delete = random.choice(available_id)
            inputs = np.where(self.matrix[:, id_to_delete] == 1)[0]
            output = np.where(self.matrix[id_to_delete, :] == 1)[0]
            self.matrix[inputs, output] = 1
            self.matrix = np.delete(self.matrix, id_to_delete, axis=0)
            self.matrix = np.delete(self.matrix, id_to_delete, axis=1)
            self.operations.pop(id_to_delete)
            self.blocks_deleted += 1
        else:
            #  If all blocks have more than one input or more than one connections
            self.delete_connections()

    def delete_connections(self):
        ones_indices = np.where(self.matrix == 1)
        n_ones = len(ones_indices[0])

        random_ids = np.arange(n_ones)
        np.random.shuffle(random_ids)

        for ID in random_ids:
            row = ones_indices[0][ID]
            col = ones_indices[1][ID]
            if np.sum(self.matrix[:, col]) > 1 and np.sum(self.matrix[row, :]) > 1:
                self.matrix[row, col] = 0
                n_edges = np.sum(self.matrix)
                if self.MAX_EDGES < n_edges == np.sum(self.matrix[0, 0:-1]) + np.sum(self.matrix[0:-1, -1]):
                    self.matrix[row, col] = 1
                else:
                    self.conn_deleted += 1
                    return

    @classmethod
    def random_individual(cls, max_blocks=None):
        if max_blocks is None:
            max_blocks = cls.MAX_VERTICES
        n_blocks = np.random.randint(0, max_blocks - 1)
        ops = [cls.INPUT] + random.choices(cls._operations, k=n_blocks) + [cls.OUTPUT]
        matrix = np.random.randint(0, 2, size=(n_blocks + 2, n_blocks + 2))
        return ChromosomeNASBench(operations=ops, matrix=matrix)

    @staticmethod
    def simple_individual():
        ops = ['INPUT', 'OUTPUT']
        matrix = np.zeros((2, 2))
        matrix[1, 1] = 1
        return ChromosomeNASBench(operations=ops, matrix=matrix)

    def cross(self, other_chromosome):
        copied_chromosome = random.choice([self, other_chromosome]).self_copy()
        min_len = min(len(self.operations), len(other_chromosome.operations))
        for i in range(min_len - 1):
            copied_chromosome.operations[i] = random.choice([self.operations[i], other_chromosome.operations[i]])
            inputs_1 = self.matrix[0:min_len, i + 1]
            inputs_2 = other_chromosome.matrix[0:min_len, i + 1]
            new_inputs = np.array([random.choice([inputs_1[k], inputs_2[k]]) for k in range(min_len)])
            copied_chromosome.matrix[0:min_len, i + 1] = new_inputs
        copied_chromosome.fix_connections()
        return copied_chromosome

    def mutate(self):
        if np.random.random() < self._grow_prob and len(self.operations) < self.MAX_VERTICES \
                and np.sum(self.matrix) < self.MAX_EDGES:  # increase blocks
            self.add_block()
            self.blocks_try_added += 1
        if np.random.random() < self._decrease_prob:  # decrease blocks
            self.delete_block()
            self.blocks_try_delete += 1
        if np.random.random() < self._connection_prob:
            if np.random.random() < 0.5 and np.sum(self.matrix) < self.MAX_EDGES:  # Add connection
                self.add_connection()
            else:                         # Delete connections
                self.delete_connections()
        if np.random.random() < self._change_op_prob and len(self.operations) > 2:  # Change operation
            id_to_change = np.random.randint(1, len(self.operations) - 1)
            self.operations[id_to_change] = random.choice(self._operations)

    def __repr__(self):
        string = str(self.matrix)
        string += '\n'
        string += str(self.operations)
        return string

    def self_copy(self):
        ops = self.operations.copy()
        matrix = self.matrix.copy()
        return ChromosomeNASBench(operations=ops, matrix=matrix)


class FitnessNASBench(Fitness):
    ALLOWED_EPOCHS = [4, 12, 36, 108]

    def __init__(self, records_path='../../nasbench/nasbench_full.tfrecord'):
        super().__init__()
        self.api = api.NASBench(records_path)
        self.precise_epochs = None
        self.epochs = None
        self.total_training_time = 0
        self.best_m = np.array([[0, 1, 1, 0, 0, 1, 1],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        self.best_op = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3',
                         'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
        self.best_nasbench = ChromosomeNASBench(self.best_op, self.best_m)
        self.best_mean_acc = self.get_test_mean_acc(self.best_nasbench)

    def reset(self):
        self.total_training_time = 0

    def calc(self, chromosome, test=False, precise_mode=False):
        cell = api.ModelSpec(matrix=chromosome.matrix, ops=chromosome.operations)
        fixed_stats, computed_stats = self.api.get_metrics_from_spec(cell)
        epochs = self.precise_epochs if precise_mode else self.epochs
        set_to_eval = 'final_test_accuracy' if test else 'final_validation_accuracy'
        stats = computed_stats[epochs]
        random_iteration = np.random.randint(0, 3)
        stat = stats[random_iteration]
        score = stat[set_to_eval]
        self.total_training_time += stat['final_training_time']
        return 1 - score

    def get_test_mean_acc(self, chromosome):
        cell = api.ModelSpec(matrix=chromosome.matrix, ops=chromosome.operations)
        fixed_stats, computed_stats = self.api.get_metrics_from_spec(cell)
        stats = computed_stats[108]
        mean_score = [stats[i]['final_test_accuracy'] for i in range(3)]
        return np.mean(mean_score)

    def get_test_regret(self, chromosome):
        score = self.get_test_mean_acc(chromosome)
        return self.best_mean_acc - score

    def set_params(self, epochs, precise_epochs):
        assert precise_epochs in self.ALLOWED_EPOCHS
        assert epochs in self.ALLOWED_EPOCHS
        self.precise_epochs = precise_epochs
        self.epochs = epochs

    def get_params(self, chromosome):
        cell = api.ModelSpec(matrix=chromosome.matrix, ops=chromosome.operations)
        data = self.api.query(cell)
        print(data)
        return data['trainable_parameters'], data


