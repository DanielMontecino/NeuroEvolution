import os
from time import time
import sys
import argparse
from config import config
from utils.utils import get_param_description
#sys.path.append('../')

from utils.codification_cnn import FitnessCNNParallel
from utils.codification_grew import FitnessGrow, ChromosomeGrow
from utils.datamanager import DataManager
from GA.geneticAlgorithm import TwoLevelGA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Evolving 2LGA.''')
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--pop_1l', type=int, default=20, help='population of the first the level')
    parser.add_argument('--pop_2l', type=int, default=8, help='population of the second the level')
    parser.add_argument('--freq_2l', type=int, default=3, help='frequency of second level evaluation')
    parser.add_argument('--eps_1l', type=int, default=18, help='epochs to train the first level population')
    parser.add_argument('--eps_2l', type=int, default=54, help='epochs to train the second level population')
    parser.add_argument('--eps_test', type=int, default=90, help='epochs to train the winner individual')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--data-aug', type=bool, default=False, help='If use random flip and random crop in the '
                                                                     'evolution and test')
    parser.add_argument('--time', type=bool, default=True, help='If include a time component in fitness')
    parser.add_argument('--runs', type=int, default=5, help='number of independent evolutions to perform')
    parser.add_argument('--dataset', type=str, default='MRDBI', help='dataset to evolve ')
    parser.add_argument('--gpus', type=int, default=4, help='gpus to use in parallel')
    parser.add_argument('--verb', type=bool, default=True, help='If show progress and each individual scoring')
    parser.add_argument('--exp_folder', type=str, default='./experiments/', help='path to save progress')

    args = parser.parse_args()

    generations = args.generations
    population_first_level = args.pop_1l
    population_second_level = args.pop_2l
    frequency_second_level = args.freq_2l

    # Fitness params
    epochs = args.eps_1l
    batch_size = args.batch_size
    smooth = config.smooth_label
    precise_eps = args.eps_2l
    test_eps = args.eps_test
    include_time = args.time
    augment = args.data_aug

    repetitions = args.runs
    dataset = args.dataset
    verbose = args.verb
    gpus = args.gpus
    exp_folder = args.exp_folder

    data_folder = './MNIST_variations'
    os.makedirs(data_folder, exist_ok=True)

    params = get_param_description(generations, population_first_level, population_second_level, epochs, batch_size,
                                   smooth, precise_eps, include_time, test_eps, augment)

    for n in range(repetitions):
        fitness_cnn = FitnessGrow()
        c = ChromosomeGrow.random_individual()
        experiments_folder = os.path.join(exp_folder, "run_{}".format(n))
        os.makedirs(experiments_folder, exist_ok=True)

        print("\nEVOLVING IN DATASET %s ...\n" % dataset)
        exp_folder = os.path.join(experiments_folder, dataset)
        folder = os.path.join(exp_folder, 'genetic')
        fitness_folder = exp_folder
        fitness_file = os.path.join(fitness_folder, 'fitness_example')
        os.makedirs(folder, exist_ok=True)

        try:
            generational = TwoLevelGA.load_genetic_algorithm(folder=folder)
            generational.num_generations = generations
        except:
            # Load data
            num_clases = 100 if dataset == 'cifar100' else 10
            dm = DataManager(dataset, clases=[], folder_var_mnist=data_folder, train_split=0.8, num_clases=num_clases,
                             normalize=True)  #,  max_examples=15000)
            data = dm.load_data()
            fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=False,
                           epochs=epochs, cosine_decay=False, early_stop=0,
                           warm_epochs=0, base_lr=0.001, smooth_label=smooth, find_lr=False,
                           precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps,  augment=augment)

            fitness_cnn.save(fitness_file)

            del dm, data

            fitness = FitnessCNNParallel()
            fitness.set_params(chrom_files_folder=fitness_folder, fitness_file=fitness_file, max_gpus=gpus,
                               fp=32, main_line='python3 ../train_gen.py')
            generational = TwoLevelGA(chromosome=c,
                                      fitness=fitness,
                                      generations=generations,
                                      population_first_level=population_first_level,
                                      population_second_level=population_second_level,
                                      training_hours=100,
                                      save_progress=True,
                                      maximize_fitness=False,
                                      statistical_validation=False,
                                      folder=folder,
                                      start_level2=frequency_second_level,
                                      frequency_second_level=frequency_second_level,
                                      perform_evo=None)

            generational.print_genetic(params)

        ti_all = time()
        print(generational.generation)
        print(generational.num_generations)
        if generational.generation < generational.num_generations - 1:
            winner, best_fit, ranking = generational.evolve(show=False)
            print("Total elapsed time: %0.3f" % (time() - ti_all))
        else:
            winner, best_fit = generational.get_best()

        


