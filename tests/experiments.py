import os
from time import time
import numpy as np
import sys
sys.path.append('../')

from utils.codification_cnn import FitnessCNNParallel
from utils.codification_grew import FitnessGrow, ChromosomeGrow, HyperParams, Merger
from utils.codification_grew import Inputs, MaxPooling, AvPooling, OperationBlock, CNNGrow, IdentityGrow
from utils.datamanager import DataManager
from GA.geneticAlgorithm import TwoLevelGA

# Chromosome parameters
ChromosomeGrow._max_initial_blocks = 5
ChromosomeGrow._grow_prob = 0.15
ChromosomeGrow._decrease_prob = 0.25


Merger._projection_type = ['normal', 'extend'][1]

HyperParams._GROW_RATE_LIMITS = [1.5, 5.]
HyperParams._N_CELLS = [1, 2]
HyperParams._N_BLOCKS = [2]
HyperParams._STEM = [32, 45]

OperationBlock._change_op_prob = 0.15
OperationBlock._change_concat_prob = 0.15

CNNGrow.filters_mul_range = [0.1, 1.2]
CNNGrow.possible_activations = ['relu', 'elu', 'prelu']
CNNGrow.dropout_range = [0, 0.7]
CNNGrow.possible_k = [1, 3, 5]
CNNGrow.k_prob = 0.2
CNNGrow.drop_prob = 0.2
CNNGrow.filter_prob = 0.2
CNNGrow.act_prob = 0.2

Inputs._mutate_prob = 0.5

    
data_folder = '../../datasets/MNIST_variations'
command = 'python3 ../train_gen.py'
verbose = 0

gpus = 2

# dataset params:
data_folder = data_folder
classes = []

# genetic algorithm params:
generations = 30
population_first_level = 20
population_second_level = 8
training_hours = 60
save_progress = True
maximize_fitness = False
statistical_validation = False
frequency_second_level = 3
start_level2 = 1


# Fitness params
epochs = 15
batch_size = 128
verbose = verbose
redu_plat = False
early_stop = 0
warm_up_epochs = 0
base_lr = 0.05
smooth = 0.1
cosine_dec = False
lr_find = False
precise_eps = 75

include_time = True
test_eps = 200
augment = 'cutout'

datasets = ['fashion_mnist', 'MB', 'MBI', 'MRB', 'MRD', 'MRDBI']

for evolve_maxpool in [True, False]:
    if evolve_maxpool:
        OperationBlock._operations = [CNNGrow, IdentityGrow, MaxPooling, AvPooling]
    else:
        OperationBlock._operations = [CNNGrow, IdentityGrow]
        
    fitness_cnn = FitnessGrow()    
    c = ChromosomeGrow.random_individual()   
    experiments_folder = '../../exp_finals_pool' if evolve_maxpool else '../../exp_finals'
    description = "Grow V2 Maxpool and AvgPool" if evolve_maxpool else "Grow V2"
    
    experiments_folder = experiments_folder
    os.makedirs(experiments_folder, exist_ok=True)
    for dataset in datasets:
        if dataset == 'cifar10':
            test_eps = 200
            augment = 'cutout'
        print("\nEVOLVING IN DATASET %s ...\n" % dataset)
        exp_folder = os.path.join(experiments_folder, dataset)
        folder = os.path.join(exp_folder, 'genetic')
        fitness_folder = exp_folder
        fitness_file = os.path.join(fitness_folder, 'fitness_example')   
        os.makedirs(folder, exist_ok=True)

        try:
            generational = TwoLevelGA.load_genetic_algorithm(folder=folder)
            # Load data
            num_clases = 100 if dataset == 'cifar100' else 10
            dm = DataManager(dataset, clases=classes, folder_var_mnist=data_folder, num_clases=num_clases) #, max_examples=8000)
            data = dm.load_data()
            fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=redu_plat,
                           epochs=epochs, cosine_decay=cosine_dec, early_stop=early_stop, 
                           warm_epochs=warm_up_epochs, base_lr=base_lr, smooth_label=smooth, find_lr=lr_find,
                           precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps, augment=augment)

            fitness_cnn.save(fitness_file)
        except:
            # Load data
            num_clases = 100 if dataset == 'cifar100' else 10
            dm = DataManager(dataset, clases=classes, folder_var_mnist=data_folder, num_clases=num_clases) #, max_examples=8000)
            data = dm.load_data()
            fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=redu_plat,
                           epochs=epochs, cosine_decay=cosine_dec, early_stop=early_stop, 
                           warm_epochs=warm_up_epochs, base_lr=base_lr, smooth_label=smooth, find_lr=lr_find,
                           precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps,  augment=augment)

            fitness_cnn.save(fitness_file)

            del dm, data

            fitness = FitnessCNNParallel()
            fitness.set_params(chrom_files_folder=fitness_folder, fitness_file=fitness_file, max_gpus=gpus,
                           fp=32, main_line=command)
            generational = TwoLevelGA(chromosome=c,
                                      fitness=fitness,
                                      generations=generations,
                                      population_first_level=population_first_level,
                                      population_second_level=population_second_level,
                                      training_hours=training_hours,
                                      save_progress=save_progress,
                                      maximize_fitness=maximize_fitness,
                                      statistical_validation=statistical_validation,
                                      folder=folder,
                                      start_level2=start_level2,
                                      frequency_second_level=frequency_second_level)
            generational.print_genetic(description)


        ti_all = time()
        print(generational.generation)
        print(generational.num_generations)
        if generational.generation < generational.num_generations:
            winner, best_fit, ranking = generational.evolve()
        print("Total elapsed time: %0.3f" % (time() - ti_all))

