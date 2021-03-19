

# Chromosome parameters
from utils.codification_grew import CNNGrow, IdentityGrow, MaxPooling

operations = [CNNGrow, IdentityGrow, MaxPooling]
max_initial_blocks = 5
grow_prob = 0.15
decrease_prob = 0.25
projection_type = ['normal', 'extend'][1]

GROW_RATE_LIMITS = [2, 4.5]
N_CELLS = [1, 2]
N_BLOCKS = [1, 2]
STEM = [32, 45]
hp_mutation_prob = 0.2
MAX_WU = 0.5
LR_LIMITS = [-9, -2] # [-9, -3]

change_op_prob = 0.15
change_concat_prob = 0.15
filters_mul_range = [0.2, 1.2]
possible_activations = ['relu', 'elu']
dropout_range = [0.0, 0.5]
possible_k = [1, 3, 5]
k_prob = 0.2
drop_prob = 0.2
filter_prob = 0.2
act_prob = 0.23

inputs_mutate_prob = 0.3

# Fitness parameters
smooth_label = 0
