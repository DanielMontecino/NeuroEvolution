import numpy as np
import keras
from keras import backend as K
import subprocess, re
import time


def clr_decay_with_warmup(global_step,
                             max_lr,
                             min_lr,
                             total_steps,
                             cycles=1):

    step_size = int(total_steps / (2 * cycles) )
    cycle = np.floor(1 + global_step / (2 * step_size))
    x = np.abs(global_step / step_size - 2 * cycle + 1)

    learning_rate = min_lr + (max_lr - min_lr) * np.max((0.0, 1.0 - x))

    return np.where(global_step > total_steps, 0.0, learning_rate)


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
        acc = logs['acc']
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
        val_acc = logs['val_acc']
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
