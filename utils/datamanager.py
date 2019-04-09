from __future__ import print_function

from keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from keras.utils import to_categorical
import numpy as np


class DataManager(object):
    '''
    This database administrator allows to use a subset of the complete database, that is, to select a more
     limited number of training examples or a smaller number of classes.
    '''

    def __init__(self, name='mnist', max_examples=None, clases=[], num_clases=10, train_split=0.8):
        assert name in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
        self.name = name
        if len(clases) > 0:
            self.num_clases = len(clases)
        else:
            self.num_clases = num_clases
        self.max_examples = max_examples
        self.clases = clases
        self.train_split = train_split

    def load_data(self):
        if self.name == 'mnist':
            data = mnist.load_data()
        elif self.name == 'fashion_mnist':
            data = fashion_mnist.load_data()
        elif self.name == 'cifar10':
            data = cifar10.load_data()
        else:
            data = cifar100.load_data()

        train_data, test_data = self.select_clases(data)
        x_train, y_train = self.limit_examples(train_data)
        x_test, y_test = test_data

        del data, test_data, train_data

        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

        y_train, y_test = self.encode(y_train, y_test)
        y_train = to_categorical(y_train, self.num_clases)
        y_test = to_categorical(y_test, self.num_clases)

        (x_train, y_train), (x_val, y_val) = self.split(x_train, y_train, self.train_split)
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        print(x_train.shape, 'train samples')
        print(x_val.shape, 'validation samples')
        print(x_test.shape, 'test samples')
        return (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val)

    def encode(self, y_train, y_test):
        self.encoder = {}
        self.decoder = {}
        clases = sorted(self.count_clases(y_train))
        for i in range(len(clases)):
            self.encoder[clases[i]] = i
            self.decoder[i] = clases[i]

        y_train = [self.encoder[l] for l in y_train]
        y_test = [self.encoder[l] for l in y_test]
        return y_train, y_test

    def limit_examples(self, data):
        examples = len(data[1])
        if self.max_examples is None or examples < self.max_examples:
            return data
        ids = np.random.permutation(examples)
        return (data[0][ids[:self.max_examples]], data[1][ids[:self.max_examples]])

    def select_clases(self, data):
        data_clases = self.count_clases(data[0][1])
        if len(data_clases) <= self.num_clases:
            self.num_clases = len(data_clases)
            return data
        if len(self.clases) == 0:
            all_clases = np.random.permutation(len(data_clases))
            sel_id = all_clases[:self.num_clases]
            sel = data_clases[sel_id]
        else:
            sel = self.clases
        idx_train = np.array([i for i in range(len(data[0][1])) if data[0][1][i] in sel])
        idx_test = np.array([i for i in range(len(data[1][1])) if data[1][1][i] in sel])
        return (data[0][0][idx_train], data[0][1][idx_train]), (data[1][0][idx_test], data[1][1][idx_test])

    def count_clases(self, labels):
        classes = []
        for label in labels:
            if not label in classes:
                classes.append(label)
        return np.array(classes)

    def decode(self, onehot_labels):
        decoded_labels = []
        if onehot_labels.ndim == 2:
            n_ex, n_clases = onehot_labels.shape
            for i in range(n_ex):
                decoded_labels.append(self.decoder[np.argmax(onehot_labels[i])])
        else:
            decoded_labels.append(self.decoder[np.argmax(onehot_labels)])
        return np.array(decoded_labels)

    def split(self, data, labels, train_split):
        s = int(train_split * data.shape[0])
        return (data[:s], labels[:s]), (data[s:], labels[s:])