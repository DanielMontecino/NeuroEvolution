import operator
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd

import sys
sys.path.append('../')


def combine_and_sort_individuals_from_dicts(sorted_dicts_list, maximize=False):
    combined_dict = dict(sorted_dicts_list[0])
    for d in sorted_dicts_list[1::]:
        combined_dict.update(dict(d))
    sorted_and_combined = sorted(combined_dict.items(), key=operator.itemgetter(1), reverse=maximize)
    return sorted_and_combined


def get_sorted_individuals_from_evolutions(dataset, parent_dir, runs, TwoLevelGAObject, maximize=False):
    '''
    Extract all trained individuals from a list of evolutions.
    return: individual list ordered by fitness (increasing) from the first and second level
    '''
    all_h, all_ph = get_dict_individuals_from_evolution(dataset, parent_dir, runs, TwoLevelGAObject)
    sorted_h = sorted(all_h.items(), key=operator.itemgetter(1), reverse=maximize)
    sorted_ph = sorted(all_ph.items(), key=operator.itemgetter(1), reverse=maximize)
    return sorted_h, sorted_ph


def get_best_individuals(x, y, err_):
    if err_ > 1 or err_ < 0:
        m = err_
        if err_ < 0:
            m = int(abs(err_) * len(y))
        y_lower_max_error = y[0:m]
        x_lower_max_error = x[0:m]

        y_higher_max_error = y[m::]
        x_higher_max_error = x[m::]
    else:
        y_lower_max_error = [y_i for y_i in y if y_i <= err_]
        x_lower_max_error = [x[i] for i in range(len(y_lower_max_error))]

        y_higher_max_error = [y_i for y_i in y if y_i > err_]
        x_higher_max_error = [x[i] for i in range(len(y_lower_max_error), len(y))]
    return (x_lower_max_error, y_lower_max_error), (x_higher_max_error, y_higher_max_error)


def plot_individuals(x_lower_max_error, y_lower_max_error, x, y, scatter, x_axis, y_axis, title, hist=False, savefig=None):
    two_plots = False
    if isinstance(y, tuple):
        x2, y2 = x[1], y[1]
        x, y = x[0], y[0]
        two_plots = True
    max_error = np.max(y_lower_max_error)
    k = 0.8  # just a margin to plot figures
    if not hist:
        plt.figure(figsize=(15, 5))
        plot = plt.scatter if scatter else plt.plot
        plt.subplot(1, 2, 1)
        plt.plot(x_lower_max_error, max_error * np.ones(len(x_lower_max_error)), color='red', linewidth=1)
        plt.plot(x_lower_max_error, min(y) * k * np.ones(len(x_lower_max_error)), color='red', linewidth=1)
        plt.plot(2 * [min(x_lower_max_error)], [min(y) * k, max_error], color='red', linewidth=1)
        plt.plot(2 * [max(x_lower_max_error)], [min(y) * k, max_error], color='red', linewidth=1)
        plot(x, y, marker='.', label='first level')
        if two_plots:
            plot(x2, y2, marker='.', label='second level')
            plt.legend()
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(title)

        plt.subplot(1, 2, 2)
        plot(x_lower_max_error, y_lower_max_error, marker='.', label='first level')
        if two_plots:
            data = np.array([[x_, y_] for x_, y_ in zip(x2, y2) if x_ in x_lower_max_error]).T
            plot(data[0, :], data[1, :] - np.log(5) / 1000, marker='.', label='second level')
            plt.legend()
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.ylim([min(y) * k, max_error])
        plt.title(title)
        if savefig is not None:
            plt.savefig("%s_%s.png" % (savefig, x_axis))
        plt.show()

    else:
        hist_data = [np.array(x, dtype=np.float32), np.array(x_lower_max_error, dtype=np.float32)]
        labels = ['all', 'best %d' % len(x_lower_max_error)]
        if two_plots:
            hist_data = [np.array(x_lower_max_error, dtype=np.float32), data[0, :]]
            labels = ['first level', 'second level']
        plt.hist(hist_data, label=labels, alpha=0.85, density=True)
        plt.legend()
        plt.xlabel(x_axis)
        plt.title(title)
        if savefig is not None:
            plt.savefig("%s_hist_%s.png" % (savefig, x_axis))
        plt.show()


def plot_column(table_, x_axis, y_axis='error', scatter=True, hist=False, title='', err_=100, savefig=None):
    '''
    Given a table with features as columns, this function selects two of them on x_axis and y_axis, and makes
    a plot (or scatter plot). Also show a histogram if hist=True.
    It makes two plots, one with the total population, and the other with K-best of them

    Params:
        table_: pd.Dataframe Table with individuals in rows and features in columns
        x_axis: feature to use in x_axis
        y_axis: feature to use in y_axis
        scatter: if use catter plot or a line plot
        hist: if show a histogram for the x_axis.
        title: title of the plot
        err_: Select the K best individuals from table_.
              If err_ \in (0, 1), select individuals with y_axis value <= err_
              elif err_ > 1, select err_ individuals
              elif err_ < 0, select |err_| * len(individuals) individuals (a proportion)

    '''
    x = table_[x_axis]
    y = table_[y_axis]
    (x_lower_max_error, y_lower_max_error), (x_higher_max_error, y_higher_max_error) = get_best_individuals(x, y, err_)
    if not hist:
        plot_individuals(x_lower_max_error, y_lower_max_error, x, y, scatter, x_axis, y_axis, title, hist, savefig)
    else:
        plot_individuals(x_lower_max_error, y_lower_max_error, x_higher_max_error, y_higher_max_error, scatter, x_axis,
                         y_axis, title, hist, savefig)


def plot_two_tables(table1, table2, y_axis='error', scatter=True, hist=False, title='', err_=100, savefig=None):
    '''
    Similar to plot_column, it but shows the first level and the second level in the same graph.
    It plots individuals by their first level id, that is, with the first level id in the x_axis
    Inputs:
        table1: table associated to the first level
        table2: table associated to the second level
    '''
    x_axis = 'id'
    x1 = table1[x_axis]
    y1 = table1[y_axis]

    x2, y2 = [], []
    for code2, y in zip(table2['code'], table2[y_axis]):
        for x, code1 in zip(x1, table1['code']):
            if code1 == code2:
                x2.append(x)
                y2.append(y)
                break
    (x_lower_max_error, y_lower_max_error), ( _, _ ) = get_best_individuals(x1, y1, err_)
    plot_individuals(x_lower_max_error, y_lower_max_error, (x1,x2), (y1,y2), scatter, x_axis, y_axis, title,hist, savefig)


def get_input_with_N_outs(code, N=3):
    blocks = code.split('\n')[0:-2]
    matrix = np.zeros((len(blocks), len(blocks)))
    connections = [block.split('||')[-2] for block in blocks]
    for i in range(len(blocks)):
        node_outputs = np.array([int(c) for c in connections[i]])
        matrix[i, 0:len(node_outputs)] = node_outputs
    n_cons = np.sum(matrix, axis=0)
    return n_cons[0] >= N > np.max(n_cons[1::])


def input_has_more_outputs(code, N=3):
    blocks = code.split('\n')[0:-2]
    matrix = np.zeros((len(blocks), len(blocks)))
    connections = [block.split('||')[-2] for block in blocks]
    for i in range(len(blocks)):
        node_outputs = np.array([int(c) for c in connections[i]])
        matrix[i, 0:len(node_outputs)] = node_outputs
    n_cons = np.sum(matrix, axis=0)
    return n_cons[0] == np.max(n_cons)


def final_node_is_conv(code):
    blocks = code.split('\n')[0:-2]
    return 'CNN' in blocks[-1]


def penultimate_node_is_conv(code):
    block = code.split('\n')[0:-2][-2]
    return 'CNN' in block


def penultimate_node_is_conv3(code):
    block = code.split('\n')[0:-2][-2]
    for param in block.split('|'):
        if 'K' in param:
            return int(param.split(':')[-1])==3
    return False


def input_connected_with_output(code):
    block = code.split('\n')[0:-2][-1]
    connections = block.split('||')[-2]
    return connections[0] == '1'


def second_node_is_conv3(code):
    blocks = code.split('\n')[0:-2]
    block = blocks[0]
    if 'Identity' in block:
        return 0
    return 'CNN' in block and 'K:3' in block


def conv3_followed_by_conv5(code):
    blocks = code.split('\n')[0:-2]
    block1 = blocks[0]
    block2 = blocks[1]
    connections = [block.split('||')[-2] for block in blocks]
    # return ('CNN' in block2 and 'K:5' in block2)
    if ('CNN' in block1 and 'K:3' in block1):
        return ('CNN' in block2 and 'K:5' in block2)
    return None

    #
    if connections[1][1] != '1':
        return 0

    return ('CNN' in block1 and 'K:3' in block1) and ('CNN' in block2 and 'K:5' in block2)


def final_node_conv(code):
    blocks = code.split('\n')[0:-2]
    block = blocks[-1]
    for param in block.split('|'):
        if 'K' in param:
            return int(param.split(':')[-1])
    return 0


def final_node_is_conv1(code):
    return final_node_conv(code) == 1


def is_there_prelu(code):
    blocks = code.split('\n')[0:-2]
    for block in blocks:
        for param in block.split('|'):
            if 'A' in param:
                if param.split(':')[-1] == 'prelu':
                    return 0
    return 1


def n_prelu(code):
    blocks = code.split('\n')[0:-2]
    prelus = 0
    for block in blocks:
        for param in block.split('|'):
            if 'A' in param:
                if param.split(':')[-1] == 'prelu':
                    prelus += 1
    return prelus > 0


def large_skip_c(code):
    blocks = code.split('\n')[0:-2]
    matrix = np.zeros((len(blocks), len(blocks)))
    connections = [block.split('||')[-2] for block in blocks]
    for i in range(len(blocks)):
        node_outputs = np.array([int(c) for c in connections[i]])
        matrix[i, 0:len(node_outputs)] = node_outputs

    for i in range(len(blocks)):
        matrix[i, i] = 0
        matrix[i, 0] = 0
    for i in range(1, len(blocks)):
        matrix[i, i - 1] = 0
    return np.sum(matrix)


def get_n_layers(code):
    return len(code.split('\n')) - 2


def get_hparam(code, hparam):
    hparams = code.split('\n')[-2].split('|')
    for param in hparams:
        if hparam in param:
            return float(param.split(':')[-1])


def get_param_func(func, param):
    def final_func(code):
        return func(code, param)

    return final_func


def get_connections(code):
    blocks = code.split('\n')[0:-2]
    connections = [block.split('||')[-2] for block in blocks]
    c = [sum([1 for s in connection if s == '1']) for connection in connections]
    return sum(c)


def get_skip_connections(code):
    blocks = code.split('\n')[0:-2]
    connections = [block.split('||')[-2] for block in blocks]
    c = [sum([1 for s in connection if s == '1']) - 1 for connection in connections]
    return sum(c)


def get_rel_connections(code):
    blocks = code.split('\n')[0:-2]
    connections = [block.split('||')[-2] for block in blocks]
    c = [sum([1 for s in connection if s == '1']) for connection in connections]
    return sum(c) / get_n_layers(code)


def get_sums(code):
    blocks = code.split('\n')[0:-2]
    sums = [1 for block in blocks if 'SUM' in block]
    return sum(sums)


def get_cats(code):
    blocks = code.split('\n')[0:-2]
    wo_cat = [1 for block in blocks if 'wo' in block]
    return get_n_layers(code) - len(wo_cat) - get_sums(code)


def get_rel_sums_cats(code):
    s = get_sums(code)
    c = get_cats(code)
    if s + c == 0:
        return 0
    return c / (s + c)


def get_f_stat(code, stat):
    blocks = code.split('\n')[0:-2]
    f = []
    for block in blocks:
        for param in block.split('|'):
            if 'F' in param:
                f.append(float(param.split(':')[-1]))
    if len(f) == 0:
        return 0
    if stat == 'mean':
        return np.mean(f)
    elif stat == 'max':
        return np.max(f)
    elif stat == 'min':
        return np.min(f)
    return np.std(f)


def get_d_stat(code, stat):
    blocks = code.split('\n')[0:-2]
    f = []
    for block in blocks:
        for param in block.split('|'):
            if 'D' in param:
                f.append(float(param.split(':')[-1]))
    if len(f) == 0:
        return 0
    if stat == 'mean':
        return np.mean(f)
    elif stat == 'max':
        return np.max(f)
    elif stat == 'min':
        return np.min(f)
    return np.std(f)


def n_identities(code):
    blocks = code.split('\n')[0:-2]
    identities = [1 for block in blocks if 'Identity' in block]
    return len(identities)


def n_convs(code):
    blocks = code.split('\n')[0:-2]
    identities = [1 for block in blocks if 'CNN' in block]
    return len(identities)


def convs_less_identity(code):
    return n_convs(code) - n_identities(code)


def get_k_stat(code, k):
    blocks = code.split('\n')[0:-2]
    f = 0
    for block in blocks:
        for param in block.split('|'):
            if 'K' in param:
                k_i = int(param.split(':')[-1])
                if k == k_i:
                    f += 1
    return f


def final_node_with_join(code):
    blocks = code.split('\n')[0:-2]
    connections = [block.split('||')[-2] for block in blocks][-1]
    inputs = [1 for i in connections if i == '1']
    return len(inputs) >= 2


def get_joins(code):
    blocks = code.split('\n')[0:-2]
    connections = [[int(inp) for inp in block.split('||')[-2]] for block in blocks]
    n_inputs = [sum(connection) for connection in connections]
    joins = [inputs >= 2 for inputs in n_inputs]
    n_joins = sum(joins)
    return n_joins


def connections_relative_join(code):
    connections = get_connections(code)
    skip_c = get_skip_connections(code)
    joins = get_joins(code)
    return connections - joins


def get_util_functions():
    util_functions = {'connections-joins': connections_relative_join,
                      'joins': get_joins,
                      'final_node_with_join': final_node_with_join,
                      'convs_less_identity': convs_less_identity,
                      'n_convs': n_convs,
                      'n_identities': n_identities,
                      'n_rel_cats': get_rel_sums_cats,
                      'n_sums': get_sums,
                      'n_cats': get_cats,
                      'rel_connections': get_rel_connections,
                      'n_connections': get_connections,
                      'n_skip_connections': get_skip_connections,
                      'n_layers': get_n_layers,
                      'large_skip_c': large_skip_c,
                      'n_prelu': n_prelu,
                      'is_there_prelu': is_there_prelu,
                      'final_node_conv': final_node_conv,
                      'conv3_followed_by_conv5': conv3_followed_by_conv5,
                      'second_node_is_conv3': second_node_is_conv3,
                      'final_node_is_conv': final_node_is_conv,
                      'input_has_more_outputs': input_has_more_outputs,
                      'has_3_or_more': get_input_with_N_outs,
                      'penultimate_node_is_conv':penultimate_node_is_conv,
                      'final_node_is_conv1': final_node_is_conv1,
                      'penultimate_node_is_conv3':penultimate_node_is_conv3,
                      'input_connected_with_output':input_connected_with_output
                      }
    for hparam in ['GR', 'CELL', 'BLOCK', 'STEM', 'LR', 'WU']:
        util_functions[hparam] = get_param_func(get_hparam, hparam)
    for val in ['mean', 'max', 'min', 'std']:
        value = 'filter_mul_%s' % val
        util_functions[value] = get_param_func(get_f_stat, val)
        value = 'dropout_%s' % val
        util_functions[value] = get_param_func(get_d_stat, val)
    for val in [1, 3, 5]:
        value = 'kernel_%d' % val
        util_functions[value] = get_param_func(get_k_stat, val)
    return util_functions


def get_sorted_tables(dataset=None, parent_dir=None, runs=None, sorted_h=None, sorted_ph=None):
    dict_h, dict_ph = {}, {}
    if sorted_h is None or sorted_ph is None:
        assert dataset is not None
        assert parent_dir is not None
        assert runs is not None
        sorted_h, sorted_ph = get_sorted_individuals_from_evolutions(dataset, parent_dir, runs)
    for i in range(len(sorted_h)):
        dict_h[i] = {'id': i, 'error': sorted_h[i][1], 'code': sorted_h[i][0]}
    for i in range(len(sorted_ph)):
        dict_ph[i] = {'id': i, 'error': sorted_ph[i][1], 'code': sorted_ph[i][0]}
    for value, function in get_util_functions().items():
        for i in range(len(sorted_h)):
            dict_h[i][value] = function(sorted_h[i][0])
        for i in range(len(sorted_ph)):
            dict_ph[i][value] = function(sorted_ph[i][0])

    table_h = pd.DataFrame(dict_h).T
    table_ph = pd.DataFrame(dict_ph).T
    return table_h, table_ph


class FilterNets:
    def __init__(self, util_functions):
        self.util_functions = util_functions
        self.function_limits = self.get_function_limits()

    def get_function_limits(self):
        function_limits = {}
        function_limits['input_has_more_outputs'] = [True]
        function_limits['final_node_is_conv'] = [True]
        function_limits['conv3_followed_by_conv5'] = [True, None]
        function_limits['final_node_conv'] = [1, 3, 5]
        function_limits['is_there_prelu'] = [True]
        function_limits['n_prelu'] = [0]
        function_limits['large_skip_c'] = [0, 1]
        function_limits['n_layers'] = [3, 4]  # [3, 4, 5]
        function_limits['GR'] = (2.5, 4.5)
        function_limits['CELL'] = [2]
        # function_limits['STEM'] = [45]
        function_limits['LR'] = (0, 0.03)  # (0.0, 0.04)
        function_limits['WU'] = (0.2, 0.4)
        function_limits['n_connections'] = (4, 7)  # (0, 9)
        function_limits['n_skip_connections'] = (1, 3)  # (0, 4)
        # function_limits['filter_mul_mean'] = (0.6, 0.8)  # (0.5, 0.9)
        function_limits['filter_mul_max'] = (0.7, 1)  # (0.6, 1)
        function_limits['filter_mul_min'] = (0.3, 0.7)  # (0.2, 0.8)
        # function_limits['filter_mul_std'] = (0, 0.3)  # (0.075, 0.3)
        # function_limits['dropout_mean'] = (0.2, 0.4)  # (0.2, 0.4)
        function_limits['dropout_max'] = (0.1, 0.55)  # (0.2, 0.5)   (0.1, 0.55)
        function_limits['dropout_min'] = (0.1, 0.3)  # (0.05, 0.3)
        # function_limits['dropout_std'] = (0, 0.18)  # (0, 0.15)
        function_limits['n_identities'] = [0, 1, 2]  # [0, 1, 2, 3]
        function_limits['n_convs'] = [2, 3, 4]  # [1, 2, 3, 4, 5]
        function_limits['kernel_1'] = [0, 1]
        function_limits['kernel_3'] = [0, 1, 2]
        function_limits['kernel_5'] = [1, 2]
        function_limits['joins'] = (1, 3)  # (0, 3)
        function_limits['connections-joins'] = (3, 5)  # (3, 7)
        return function_limits

    def condition_all(self, hi, fitness_lim=None):
        code, fit = hi
        final_condition = True
        for value, limits in self.function_limits.items():
            final_condition = final_condition and self.condition_unique(hi, value)
        if fitness_lim is not None:
            final_condition = final_condition and (fit <= fitness_lim)
        return final_condition

    def condition_unique(self, hi, value):
        code, fit = hi
        func = self.util_functions[value]
        limits = self.function_limits[value]
        if isinstance(limits, list):
            return func(code) in limits
        elif isinstance(limits, tuple):
            f = func(code)
            return limits[0] <= f <= limits[1]
        else:
            raise NotImplementedError

    def filter_all_conditions(self, h, fitness_lim=None):
        return [hi for hi in h if self.condition_all(hi, fitness_lim)]

    def filter_one_condition(self, h, value):
        assert value in self.util_functions.keys()
        if value not in self.function_limits.keys():
            return h
        return [hi for hi in h if self.condition_unique(hi, value)]


def filter_set(h, fitness_lim=None):
    def get_hparams(code):
        hparams = code.split('\n')[-2].split('|')
        hparams_dict = {}
        for hparam in hparams:
            if ':' in hparam:
                hp, value = hparam.split(':')
                hparams_dict[hp] = float(value)
        return hparams_dict

    def filter_code(hparams_dict, hparam, condition):
        return condition(hparams_dict[hparam])

    def filter_pipeline(hi):
        code, fitness = hi
        hparams_dict = get_hparams(code)
        is_valid = filter_code(hparams_dict, 'LR', lambda x: x < 0.07)
        is_valid = is_valid and filter_code(hparams_dict, 'GR', lambda x: x >= 2.5)
        is_valid = is_valid and filter_code(hparams_dict, 'GR', lambda x: x <= 4.5)
        is_valid = is_valid and filter_code(hparams_dict, 'CELL', lambda x: x == 2)
        # is_valid = is_valid and filter_code(hparams_dict, 'STEM', lambda x: x==45)
        if fitness_lim is not None:
            is_valid = is_valid and fitness < fitness_lim
        return is_valid

    return [hi for hi in h if filter_pipeline(hi)]


def make_percentiles(h, divs, div_mode):
    assert div_mode in ['pop', 'fit']
    percentiles_dict = {}
    for i in range(divs):
        percentiles_dict['b%d' % (i + 1)] = []
    if div_mode == 'pop':
        divisions = np.linspace(0, len(h), divs + 1)
        for i in range(len(h)):
            k_p = np.argmax((i <= divisions[1::]))
            percentiles_dict['b%d' % (k_p + 1)].append(h[i])

    else:  # div_mode == 'fit':
        max_fit = np.max([h[0][1], h[-1][1]])
        min_fit = np.min([h[0][1], h[-1][1]])
        divisions = np.linspace(min_fit, max_fit, divs + 1)
        for i in range(len(h)):
            fit = h[i][1]
            k_p = np.argmax((fit <= divisions[1::]))
            percentiles_dict['b%d' % (k_p + 1)].append(h[i])

    percentiles_dict['all'] = h
    return percentiles_dict, divisions


def color_names():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    names = []
    for i, name in enumerate(sorted_names):
        col = i % 4
        if col == 0:
            names.append(name)
    return [names[i] for i in [6, 14, 18, 20, 26, 31, 36, 5, 32, 2, 9]] + names


def plot_percentiles(percentiles_dict, divisions, div_mode, save=None, show_param=False):
    xi = 0
    plot_info = {}
    for k, v in percentiles_dict.items():
        plot_info[k] = {}
        plot_info[k]['y'] = [p[1] for p in v]
        plot_info[k]['code'] = [p[0] for p in v]
        plot_info[k]['x'] = np.arange(xi, len(plot_info[k]['y']) + xi, 1)
        xi += len(plot_info[k]['y'])

    i = 0
    if show_param:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    for k, v in plot_info.items():
        if k == 'all' or len(v['x']) == 0:
            i += 1
            continue
        if show_param:
            ax.scatter(v['x'], v['y'], label="%s (%d)" % (k, len(v['x'])), color=color_names()[i], marker='.')
            for ii in range(len(v['y'])):
                ax.annotate('%s' % get_util_functions()[show_param](v['code'][ii]),
                            xy=(v['x'][ii], v['y'][ii]), xytext=(v['x'][ii] + 0.0, v['y'][ii] + 0.005),
                            arrowprops=dict(arrowstyle='->'))
        else:
            plt.scatter(v['x'], v['y'], label="%s (%d)" % (k, len(v['x'])), color=color_names()[i], marker='.')
        i += 1
    if div_mode == 'pop':
        plt.xticks(divisions.astype(np.int32))
        plt.grid(axis='x')
    elif div_mode == 'fit':
        plt.yticks(divisions)
        plt.grid(axis='y')
    plt.legend()
    plt.xlabel('Id')
    plt.ylabel('Fitness')
    if save is not None:
        plt.savefig(save)
    plt.show()
    return plot_info


def get_dict_individuals_from_evolution(dataset, parent_dir, runs, TwoLevelGAObject):
    all_h = {}
    all_ph = {}
    for run in runs:
        genetic_path = os.path.join(parent_dir, str(run), dataset, 'genetic')
        genetic = TwoLevelGAObject.load_genetic_algorithm(folder=genetic_path)
        h = genetic.history_fitness
        all_h.update(h)
        h = genetic.history_precision_fitness
        all_ph.update(h)
    return all_h, all_ph