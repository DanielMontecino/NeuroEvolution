import os

import numpy as np
import keras
from keras import backend as K
import subprocess
import re
import time

from GA.geneticAlgorithm import TwoLevelGA
from utils.codification_grew import ChromosomeGrow, Merger, HyperParams, OperationBlock, CNNGrow


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4,  r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    if pixel_level:
        print("Cutout WITH pixel-level noise")
    else:
        print("Cutout WITHOUT pixel-level noise")

    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0.1
    if epoch > 150:
        lr *= 0.1
    elif epoch > 225:
        lr *= 0.001
    #elif epoch > 120:
    #    lr *= 1e-2
    #elif epoch > 80:
    #    lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr


def linear_decay_with_warmup(global_step,
                             max_lr,
                             min_lr,
                             total_steps,
                             warmup_steps=1):

    x1, y1 = 0, min_lr
    x2, y2 = warmup_steps, max_lr
    x3, y3 = total_steps, min_lr

    if global_step < warmup_steps:
        learning_rate = global_step * (y2 - y1) / (x2 - x1) + y1
    else:
        learning_rate = (global_step * (y3 - y2) + (y2 * x3 - y3 * x2)) / (x3 - x2)

    return np.where(global_step > total_steps, 0.0, learning_rate)


def clr_decay_with_warmup(global_step,
                          max_lr,
                          min_lr,
                          total_steps,
                          cycles=1):

    step_size = int(total_steps / (2 * cycles))
    cycle = np.floor(1 + global_step / (2 * step_size))
    x = np.abs(global_step / step_size - 2 * cycle + 1)

    learning_rate = min_lr + (max_lr - min_lr) * np.max((0.0, 1.0 - x))

    return np.where(global_step > total_steps, 0.0, learning_rate)


class LinearScheduler(keras.callbacks.Callback):

    def __init__(self,
                 max_lr,
                 min_lr,
                 total_steps,
                 warmup_steps=1,
                 global_step_init=0,
                 verbose=0):

        super(LinearScheduler, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = max(warmup_steps, 1)
        self.global_step = global_step_init
        self.verbose = verbose
        self.learning_rates = []
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        loss = logs['loss']
        acc = [logs[key] for key in logs.keys() if 'acc' in key][0]
        self.accs.append(acc)
        self.losses.append(loss)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = linear_decay_with_warmup(global_step=self.global_step,
                                      max_lr=self.max_lr,
                                      min_lr=self.min_lr,
                                      total_steps=self.total_steps,
                                      warmup_steps=self.warmup_steps)

        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class CLRScheduler(keras.callbacks.Callback):

    def __init__(self,
                 max_lr,
                 min_lr,
                 total_steps,
                 cycles=1,
                 global_step_init=0,
                 verbose=0):

        super(CLRScheduler, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.cycles = cycles
        self.global_step = global_step_init
        self.verbose = verbose
        self.learning_rates = []
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        loss = logs['loss']
        acc = [logs[key] for key in logs.keys() if 'acc' in key][0]
        self.accs.append(acc)
        self.losses.append(loss)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = clr_decay_with_warmup(global_step=self.global_step,
                                   max_lr=self.max_lr,
                                   min_lr=self.min_lr,
                                   total_steps=self.total_steps,
                                   cycles=self.cycles)

        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class EarlyStopByTimeAndAcc(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 limit_time=150,
                 baseline=0.15,
                 patience=2,
                 verbose=0):
        super(EarlyStopByTimeAndAcc, self).__init__()
        self.limit_time = limit_time
        self.baseline = baseline
        self.patience = patience
        self.epochs_without_improve = 0
        self.verbose = verbose
        self.learning_rates = []
        self.initial_time = 0

    def on_epoch_end(self, batch, logs=None):
        # Verifying if the batch processing took to much time
        elapsed_batch_time = time.time() - self.initial_time
        if elapsed_batch_time > self.limit_time:
            self.model.stop_training = True
            if self.verbose > 0:
                print('\nBatch processing took too much time! %0.2f.' % (elapsed_batch_time / 60.0))
            return

        # Verifying if the validation accuracy doesn't improve over the baseline.
        val_acc = [logs[key] for key in logs.keys() if 'val_acc' in key][0]
        if val_acc <= self.baseline:
            self.epochs_without_improve += 1
        else:
            self.epochs_without_improve = 0

        if self.epochs_without_improve >= self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                print('\nModel is not learning!.')
            return

    def on_epoch_begin(self, batch, logs=None):
        self.initial_time = time.time()


def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_maps(mode='free_fraction'):
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi -q  -d MEMORY")
    atacched_gpus = 0
    for row in output.split('\n'):
        if 'Attached GPUs' in row:
            atacched_gpus = int(row.split(':')[1])
    info_gpu = output.split('GPU ')[1:]
    assert len(info_gpu) == atacched_gpus
    final_gpu_dict = {}
    for gpu_id, info in enumerate(info_gpu):
        gpu_dict = {}
        line = info.split('BAR')[0].split('FB')[1].split('\n')
        line = [l.replace(" ", "") for l in line if ':' in l]
        for sub_line in line:
            key, val = sub_line.split(':')
            gpu_dict[key] = int(val.split('MiB')[0])
        if mode == 'free_fraction':
            final_gpu_dict[gpu_id] = gpu_dict['Free'] * 1. / gpu_dict['Total']
        elif mode == 'used_fraction':
            final_gpu_dict[gpu_id] = gpu_dict['Used'] * 1. / gpu_dict['Total']
        elif mode == 'free_memory':
            final_gpu_dict[gpu_id] = gpu_dict['Free']
        elif mode == 'all_memory':
            final_gpu_dict[gpu_id] =  gpu_dict['Total']
    return final_gpu_dict


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def verify_free_gpu_memory(min_frac=0.8):
    gpu_fractions = gpu_memory_maps()
    gpu_ids, fractions = list(gpu_fractions.keys()), list(gpu_fractions.values())
    k = int(np.argmax(fractions))
    id, max_frac = gpu_ids[k], fractions[k]
    if max_frac < min_frac:
        print("Waiting for a GPU... free memory fractions: %0.4f" % max_frac)
        return False, -1
    return True, id


def get_n_best_str(generational_, N=3):
    x = generational_.history_precision_fitness
    a = sorted(x.items(), key=lambda item: item[1])
    best_str = [a[k][0] for k in range(N)]
    print("Scores selected:", [a[k][1] for k in range(N)])
    return best_str


def get_n_best(generational_, N=3):
    best_str = get_n_best_str(generational_, N)
    all_gens = generational_.population_history['2-level']
    winner, best_fit = generational_.get_best()
    best_n = [winner]
    added_str = [winner.__repr__()]

    for gen in all_gens.values():
        for indiv in gen:
            if indiv.__repr__() not in added_str and indiv.__repr__() in best_str:
                best_n.append(indiv)
                added_str.append(indiv.__repr__())
    return best_n


def get_n_best_from_exp(experiments_folder, N=3, dataset='cifar10'):
    exp_folder = os.path.join(experiments_folder, dataset)
    folder = os.path.join(exp_folder, 'genetic')
    generational = TwoLevelGA.load_genetic_algorithm(folder=folder)
    return get_n_best(generational, N)


def get_param_description(generations, population_first_level, population_second_level, epochs, batch_size, smooth,
                          precise_eps, include_time, test_eps, augment):
    params = "\nParameters\n"

    params += "initial blocks: %d  \n" % ChromosomeGrow._max_initial_blocks
    params += "grow prob: %0.2f  \n" % ChromosomeGrow._grow_prob
    params += "decrease prob: %0.2f  \n" % ChromosomeGrow._decrease_prob
    params += "projection: %s  \n" % Merger._projection_type
    params += "grow rate limits: %s  \n" % HyperParams._GROW_RATE_LIMITS
    params += "n cells: %s  \n" % HyperParams._N_CELLS
    params += "n blocks: %s  \n" % HyperParams._N_BLOCKS
    params += "stems: %s  \n" % HyperParams._STEM
    params += "hyperparam mutation prob: %0.2f  \n" % HyperParams.mutation_prob
    params += "change op prob: %0.2f  \n" % OperationBlock._change_op_prob
    params += "change concat prob: %0.2f  \n" % OperationBlock._change_concat_prob
    params += "learning rate limits: %s\n" % HyperParams._LR_LIMITS
    params += "filters range: %s  \n" % CNNGrow.filters_mul_range
    params += "activations: %s  \n" % CNNGrow.possible_activations
    params += "dropout range: %s  \n" % CNNGrow.dropout_range
    params += "possible kernels sizes: %s  \n" % CNNGrow.possible_k
    params += "kernel mutation prob: %0.2f  \n" % CNNGrow.k_prob
    params += "dropout mutation prob: %0.2f  \n" % CNNGrow.drop_prob
    params += "filter mutation prob: %0.2f  \n" % CNNGrow.filter_prob
    params += "activation mutation prob: %0.2f  \n" % CNNGrow.act_prob

    # genetic algorithm params:
    params += "generations: %d  \n" % generations
    params += "population first level: %d  \n" % population_first_level
    params += "population second level: %d  \n" % population_second_level

    # Fitness params
    params += "epochs: %d  \n" % epochs
    params += "batch8 size: %d  \n" % batch_size
    params += "smooth: %0.2f  \n" % smooth
    params += "precise epochs: %0.2f  \n" % precise_eps
    params += "include time: %s  \n" % include_time
    params += "test epochs: %d \n" % test_eps
    params += "augment: %s  \n" % augment
    return params