import os
from time import time
import numpy as np
import sys
sys.path.append('../')

from utils.codification_cnn import FitnessCNNParallel
from utils.codification_grew import FitnessGrow, ChromosomeGrow, HyperParams, Merger
from utils.codification_grew import Inputs, MaxPooling, AvPooling, OperationBlock, CNNGrow, IdentityGrow
from utils.datamanager import DataManager
from GA.geneticAlgorithm import GenerationalGA

# Chromosome parameters
ChromosomeGrow._max_initial_blocks = 5
ChromosomeGrow._grow_prob = 0.15
ChromosomeGrow._decrease_prob = 0.25


Merger._projection_type = ['normal', 'extend'][1]

HyperParams._GROW_RATE_LIMITS = [2, 4.5]
HyperParams._N_CELLS = [1, 2]
HyperParams._N_BLOCKS = [2]
HyperParams._STEM = [32, 45]
HyperParams.mutation_prob = 0.2
HyperParams._MAX_WU = 0.5
HyperParams._LR_LIMITS = [-9, -2]

OperationBlock._change_op_prob = 0.15
OperationBlock._change_concat_prob = 0.15

CNNGrow.filters_mul_range = [0.2, 1.2]
CNNGrow.possible_activations = ['relu', 'elu']
CNNGrow.dropout_range = [0, 0.7]
CNNGrow.possible_k = [1, 3, 5]
CNNGrow.k_prob = 0.2
CNNGrow.drop_prob = 0.2
CNNGrow.filter_prob = 0.2
CNNGrow.act_prob = 0.23

Inputs._mutate_prob = 0.3

    
data_folder = '../../datasets/MNIST_variations'
command = 'python3 ../train_gen.py'
verbose = 0
gpus = 4

# dataset params:
classes = []

# genetic algorithm params:
generations = 20
population = 23
training_hours = 60
save_progress = True
maximize_fitness = False
statistical_validation = False

# Fitness params
batch_size = 128
verbose = verbose
redu_plat = False
early_stop = 0
warm_up_epochs = 0
base_lr = 0.05
smooth = 0.1
cosine_dec = False
lr_find = False
precise_eps = None

include_time = True
test_eps = 90
augment = False

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
params += "population: %d  \n" % population
params += "hours: %d  \n" % training_hours

# Fitness params
params += "batch8 size: %d  \n" % batch_size
params += "smooth: %0.2f  \n" % smooth
params += "include time: %s  \n" % include_time
params += "test epochs: %d \n" % test_eps
params += "augment: %s  \n" % augment


dataset = 'MRDBI'

for n in range(5):
    for epochs in [18, 54]:
        OperationBlock._operations = [CNNGrow, IdentityGrow]
        
        fitness_cnn = FitnessGrow()    
        c = ChromosomeGrow.random_individual()   
        experiments_folder = '../../experiments/1level_20gen/%d_epochs/%d' % (epochs, n)
        description = "Testing One Level Genetic Algorithm with 20 generations"
        os.makedirs(experiments_folder, exist_ok=True)
    
        print("\nEVOLVING IN DATASET %s ...\n" % dataset)
        exp_folder = os.path.join(experiments_folder, dataset)
        folder = os.path.join(exp_folder, 'genetic')
        fitness_folder = exp_folder
        fitness_file = os.path.join(fitness_folder, 'fitness_example')   
        os.makedirs(folder, exist_ok=True)
        try:
            generational = GenerationalGA.load_genetic_algorithm(folder=folder)
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
            generational = GenerationalGA(chromosome=c,
                                      fitness=fitness,
                                      generations=generations,
                                      num_population=population,
                                      training_hours=training_hours,
                                      save_progress=save_progress,
                                      maximize_fitness=maximize_fitness,
                                      statistical_validation=statistical_validation,
                                      folder=folder)
            generational.print_genetic(description)


        ti_all = time()
        print(generational.generation)
        print(generational.num_generations)
        if generational.generation < generational.num_generations:
            winner, best_fit, ranking = generational.evolve(show=False)
        print("Total elapsed time: %0.3f" % (time() - ti_all))

       
