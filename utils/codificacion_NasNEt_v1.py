import numpy as np
import random

N_BLOCKS = 5


class Op(object):
    OPS = ['identity', 'conv_sep', 'conv_dilated', 'avg_pool', 'max_pool', 'conv', 'conv_depth_sep']
    SIZES = [1, 3, 5, 7]  # Only squared operations
    MAX_FILTERS = 512

    OP_DIGITS = np.ceil(np.log(len(OPS)) / np.log(2)).astype(np.int32)
    SIZE_DIGITS = np.ceil(np.log(len(SIZES)) / np.log(2)).astype(np.int32)
    F_DIGITS = np.ceil(np.log(MAX_FILTERS) / np.log(2)).astype(np.int32)
    LEN_CODE = OP_DIGITS + SIZE_DIGITS + F_DIGITS

    def __init__(self, operation, size, filters):
        assert operation in self.OPS
        assert size in self.SIZES
        assert filters <= self.MAX_FILTERS

        self.operation = operation
        self.size = size
        self.filters = filters
        self.code = self.encode(self.operation, self.size, self.filters)

    def __repr__(self):
        return "(%s_%dx%d_%d)" % (self.operation, self.size, self.size, self.filters)

    @staticmethod
    def encode(operation, size, filters):
        code = []
        code += Op.__encode_binary(Op.OPS.index(operation), Op.OP_DIGITS)
        code += Op.__encode_binary(Op.SIZES.index(size), Op.SIZE_DIGITS)
        code += Op.__encode_binary(filters, Op.F_DIGITS)
        return code

    @staticmethod
    def random_op():
        op = random.choice(Op.OPS)
        size = random.choice(Op.SIZES)
        filters = np.random.randint(1, Op.MAX_FILTERS)
        return Op(op, size, filters)

    @staticmethod
    def random_op_code():
        return Op.random_op().code

    @staticmethod
    def decode_op(code):
        assert len(code) == Op.LEN_CODE
        assert Op.verify_code(code)
        op_code = code[0:Op.OP_DIGITS]
        size_code = code[Op.OP_DIGITS:Op.OP_DIGITS + Op.SIZE_DIGITS]
        filters_code = code[-Op.F_DIGITS::]

        op = Op.OPS[Op.__binary_to_decimal(op_code)]
        size = Op.SIZES[Op.__binary_to_decimal(size_code)]
        filters = Op.__binary_to_decimal(filters_code)
        return Op(op, size, filters)

    @staticmethod
    def __encode_binary(values, digits):
        encoding = []
        if not isinstance(values, list):
            values = [values]
        for value in values:
            encode = Op.__binarizer(value)
            delta_digits = digits - len(encode)
            encode = [0 for _ in range(delta_digits)] + encode
            encoding += encode
        return encoding

    @staticmethod
    def __binarizer(decimal):
        binary = []
        while decimal // 2 != 0:
            binary.append(decimal % 2)
            decimal = decimal // 2
        binary.append(decimal)
        return [binary[-i - 1] for i in range(len(binary))]

    @staticmethod
    def __binary_to_decimal(binary):
        return np.sum([binary[-1 - i] * 2 ** i for i in range(len(binary))])

    @staticmethod
    def verify_code(code):
        if len(code) != Op.LEN_CODE:
            return False
        op_code = code[0:Op.OP_DIGITS]
        size_code = code[Op.OP_DIGITS:Op.OP_DIGITS + Op.SIZE_DIGITS]
        filters_code = code[-Op.F_DIGITS::]

        # To decimals
        op_code = Op.__binary_to_decimal(op_code)
        size_code = Op.__binary_to_decimal(size_code)
        filters_code = Op.__binary_to_decimal(filters_code)

        if op_code >= len(Op.OPS) or size_code >= len(Op.SIZES):
            return False
        if filters_code > Op.MAX_FILTERS or filters_code == 0:
            return False
        return True


class Cell(object):
    def __init__(self, operations=None, connections=None, n_blocks=5, n_connections=2):
        self.n_blocks = n_blocks
        self.n_connect = n_connections
        self.n_ops = self.n_blocks * self.n_connect
        if (operations is not None) and (connections is not None):
            assert len(operations) == self.n_ops
            assert len(connections) == n_blocks * (n_connections + 1)
            self.connects = connections
            self.ops = operations
        else:
            self.connects = self.random_connections()
            self.ops = self.random_operations()


    def encode_ops(self, operations):
        code = []
        for op in operations:
            code += op.encode()
        return code


    def random_operations(self):
        ops = [Op.random_op() for _ in range(self.n_ops)]
        return ops

    def random_connections(self):
        connections = []
        for i in range(self.n_blocks):
            connections += random.choices(range(i+2), k=self.n_connect)
        return connections

    def onehot_connections(self, connections):
        connections_onehot = []
        for i in range(self.n_blocks):
            current_connection = connections[i * self.n_connect: (i + 1) * self.n_connect]
            connections_onehot += self.encode_to_onehot(current_connection, i + 2)
        return connections_onehot

    def encode_to_onehot(self, values, list_values):
        encoding = []
        if not isinstance(values, list):
            values = [values]
        for value in values:
            index = list_values.index(value)
            encoding += list(to_categorical(index, len(list_values)))
        return encoding


        op = random.choice(OPS)
        size = random.choices(sizes, k=2)
        filters = random.choice(range(2 ** max_digits_F))
        print()

        self.operations = []
        print(op)
        print(self.encode_to_onehot(op, OPS), end='\n\n')
        print(size)
        print(self.encode_to_onehot(size, sizes), end='\n\n')
        print(filters)
        print(self.__encode_binary(filters, max_digits_F))



