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
from utils.net_classification import ModelFilter

# Chromosome parameters
ChromosomeGrow._max_initial_blocks = 6
ChromosomeGrow._grow_prob = 0.15
ChromosomeGrow._decrease_prob = 0.25


Merger._projection_type = ['normal', 'extend'][1]

HyperParams._GROW_RATE_LIMITS = [2, 4.5]
HyperParams._N_CELLS = [1, 2]
HyperParams._N_BLOCKS = [2]
HyperParams._STEM = [32, 45]
HyperParams.mutation_prob = 0.2
HyperParams._MAX_WU = 0.5
HyperParams._LR_LIMITS = [-9, -2] # [-9, -3]

OperationBlock._change_op_prob = 0.15
OperationBlock._change_concat_prob = 0.15
CNNGrow.filters_mul_range = [0.2, 1.2]
CNNGrow.possible_activations = ['relu', 'elu']
CNNGrow.dropout_range = [0.0, 0.7]
CNNGrow.possible_k = [1, 3, 5]
CNNGrow.k_prob = 0.2
CNNGrow.drop_prob = 0.2
CNNGrow.filter_prob = 0.2
CNNGrow.act_prob = 0.2

Inputs._mutate_prob = 0.3

    
data_folder = '../../datasets/MNIST_variations'
command = 'python3 ../train_gen.py'
verbose = 0

gpus = 4


# dataset params:
data_folder = data_folder
classes = []

# genetic algorithm params:
generations = 10
population_first_level = 30
population_second_level = 15
training_hours = 48
save_progress = True
maximize_fitness = False
statistical_validation = False
frequency_second_level = 3
start_level2 = 2
model_filter = ModelFilter('model_filter2', TwoLevelGA)


# Fitness params
epochs = 18
batch_size = 128
verbose = verbose
redu_plat = False
early_stop = 0
warm_up_epochs = 0
base_lr = 0.05
smooth = 0.1
cosine_dec = False
lr_find = False
precise_eps = 54

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
params += "population first level: %d  \n" % population_first_level
params += "population second level: %d  \n" % population_second_level
params += "hours: %d  \n" % training_hours
params += "frequency second level: %d  \n" % frequency_second_level
params += "start evaluating second level: %d  \n" % start_level2

# Fitness params
params += "epochs: %d  \n" % epochs
params += "batch8 size: %d  \n" % batch_size
params += "smooth: %0.2f  \n" % smooth
params += "precise epochs: %0.2f  \n" % precise_eps
params += "include time: %s  \n" % include_time 
params += "test epochs: %d \n" % test_eps 
params += "augment: %s  \n" % augment 


datasets = ['MB','MBI', 'MRB', 'MRD', 'MRDBI', 'fashion_mnist']
datasets = ['fashion_mnist']
datasets = ['MB', 'MRDBI', 'MBI', 'MRB', 'MRD']
datasets = ['fashion_mnist']
repetitions = 5
init = 0
for n in range(init, init + repetitions):
    if False:
        OperationBlock._operations = [CNNGrow, IdentityGrow, MaxPooling]
    else:
        OperationBlock._operations = [CNNGrow, IdentityGrow]
    generations = 10 - 2 * n
    for dataset in datasets:        
        description = "Testing %s with filter models and %d generations (using sep conv5)" %(dataset,  generations)
        fitness_cnn = FitnessGrow()    
        c = ChromosomeGrow.random_individual()   
        experiments_folder = '../../experiments/model_filter_mrdbi/%d' % n
        os.makedirs(experiments_folder, exist_ok=True)

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
            dm = DataManager(dataset, clases=classes, folder_var_mnist=data_folder, num_clases=num_clases)  #, max_examples=12000)
            data = dm.load_data()
            fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=redu_plat,
                           epochs=epochs, cosine_decay=cosine_dec, early_stop=early_stop,
                           warm_epochs=warm_up_epochs, base_lr=base_lr, smooth_label=smooth, find_lr=lr_find,
                           precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps,  augment=augment)

        except:
            # Load data
            num_clases = 100 if dataset == 'cifar100' else 10
            dm = DataManager(dataset, clases=classes, folder_var_mnist=data_folder, num_clases=num_clases)  #,  max_examples=12000)
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
                                      frequency_second_level=frequency_second_level,
                                      model_filter=model_filter)
            generational.print_genetic(description)
            generational.print_genetic(params)

        ti_all = time()
        print(generational.generation)
        print(generational.num_generations)
        if generational.generation < generational.num_generations - 1:
            winner, best_fit, ranking = generational.evolve(show=False)
        print("Total elapsed time: %0.3f" % (time() - ti_all))
        winner, best_fit = generational.get_best()
        print("\nFinal test")
        print(winner)
        lr = winner.hparams.lr
        fitness_cnn.verb = True
        
        for eps in [epochs, precise_eps, test_eps]:
            if dataset == 'fashion_mnist':
                continue
            fitness_cnn.test_eps = eps
            fitness_cnn.save(fitness_file)
            generational.fitness_evaluator.set_params(chrom_files_folder=fitness_folder, fitness_file=fitness_file, max_gpus=gpus,
                           fp=32, main_line=command)
            score = generational.fitness_evaluator.calc(winner, test=True)
            generational.print_genetic("\nTesting the winner with %d epochs" % eps)
            generational.print_genetic("Test scroe: %0.4f" % score)
        fitness_cnn.test_eps = test_eps
        fitness_cnn.save(fitness_file) 
        for stem in [32, 45]:
            continue
            for cells in [2]:
                winner.hparams.stem = stem
                #winner.hparams.n_cells = cells
                score = generational.fitness_evaluator.calc(winner, test=True)
                generational.print_genetic("\n\nCells: %d, Stem: %d" %(cells, stem))
                generational.print_genetic("Score: %0.4f" % score)
        for aug in [True, 'cutout']:
            continue
            #winner.hparams.stem = 32
            #winner.hparams.n_cells = 
            fitness_cnn.augment = aug
            fitness_cnn.save(fitness_file)
            score = generational.fitness_evaluator.calc(winner, test=True)
            generational.print_genetic("\n\nStem: %d, augment: %s, epochs: %d" %(winner.hparams.stem, aug, fitness_cnn.test_eps))
            generational.print_genetic("Score: %0.4f" % score)
        
       # dm = DataManager(dataset, clases=classes, folder_var_mnist=data_folder, num_clases=num_clases) #, max_examples=8000)
       # data = dm.load_data()
       # fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=redu_plat,
       #                    epochs=epochs, cosine_decay=cosine_dec, early_stop=early_stop,
       #                    warm_epochs=warm_up_epochs, base_lr=base_lr, smooth_label=smooth, find_lr=lr_find,
       #                    precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps,  augment=augment)

       # fitness_cnn.save(fitness_file)
        for aug in [False, True, 'cutout']:
            if dataset != 'fashion_mnist':
                continue
            #winner.hparams.stem = 32
            #winner.hparams.n_cells = 
            fitness_cnn.augment = aug
            fitness_cnn.save(fitness_file)
            score = generational.fitness_evaluator.calc(winner, test=True)
            generational.print_genetic("\n\nTesting with all data.\nStem: %d, augment: %s, epochs: %d" %(winner.hparams.stem, aug, fitness_cnn.test_eps))
            generational.print_genetic("Score: %0.4f" % score)


        fitness_cnn.augment = augment
        fitness_cnn.verb = False
        fitness_cnn.test_eps = test_eps
        fitness_cnn.save(fitness_file)

#Changue: experiments_folder, max_samples when loading and creating GA, final testing with all data, batch_size, description and repteitions ids.
