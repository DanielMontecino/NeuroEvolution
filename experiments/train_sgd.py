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


data_folder = '../../datasets/MNIST_variations'
command = 'python3 ../train_gen.py'
verbose = 0

gpus = 4


# dataset params:
data_folder = data_folder
classes = []


# Fitness params
epochs = 5
batch_size = 256
verbose = True
redu_plat = True
early_stop = 0
warm_up_epochs = 0
base_lr = 0.05
smooth = 0
cosine_dec = False
lr_find = False
precise_eps = 300

include_time = True
test_eps = 300
augment = 'cutout'

params = "\nParameters\n"
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
repetitions = 5
for n in range(0, repetitions):
    if False:
        OperationBlock._operations = [CNNGrow, IdentityGrow, MaxPooling]
        description = "Validation with 20 1en, 30 and 15 indvs. And with maxpooling. Fitness with std and time"
    else:
        OperationBlock._operations = [CNNGrow, IdentityGrow]
        description = "Validation with 20 1en, 30 and 15 indvs. And witouth maxpooling. Fitness with std and time"

    for dataset in datasets:        
        fitness_cnn = FitnessGrow()    
        c = ChromosomeGrow.random_individual()   
        experiments_folder = '../../experiments/test_validation3/%d' % n
    
        # description = "Testing parameters: initial blocks, cells, warmup limit, stem"   
        experiments_folder = experiments_folder
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
            dm = DataManager(dataset, clases=classes, folder_var_mnist=data_folder, num_clases=num_clases) #, max_examples=8000)
            data = dm.load_data()
            fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=redu_plat,
                           epochs=epochs, cosine_decay=cosine_dec, early_stop=early_stop,
                           warm_epochs=warm_up_epochs, base_lr=base_lr, smooth_label=smooth, find_lr=lr_find,
                           precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps,  augment=augment)

            fitness_cnn.save(fitness_file)

            fitness = FitnessCNNParallel()
            fitness.set_params(chrom_files_folder=fitness_folder, fitness_file=fitness_file, max_gpus=gpus,
                           fp=32, main_line=command)

        except:
            print("Cant load Geneticlgortihm")
                        
        generational.print_genetic('Trainig with 300 epochs and with SGD')
           
        ti_all = time()
        print(generational.generation)
        print(generational.num_generations)
       
       
        winner, best_fit = generational.get_best()
        print("\nFinal test")
        print(winner)
        lr = winner.hparams.lr
        #wu = winner.hparams.warmup
        # winner.hparams.lr = lr - np.log(5)/np.log(2)
        fitness_cnn.verb = True
        #winner.hparams.lr = lr
        score = fitness.calc(winner, test = True)
        generational.print_genetic("\nTesting the winner with %d epochs" % fitness_cnn.test_eps)
        generational.print_genetic("Test scroe: %0.4f" % score)
       
       

