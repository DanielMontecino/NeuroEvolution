import os
import numpy as np
import sys
sys.path.append('../')

from GA.geneticAlgorithm import TwoLevelGA
from utils.codification_grew import FitnessGrow


def load_genetic(experiments_folder, dataset):
    exp_folder = os.path.join(experiments_folder, dataset)
    folder = os.path.join(exp_folder, 'genetic')
    if os.path.isdir(exp_folder):
        print("Exists")
        return TwoLevelGA.load_genetic_algorithm(folder=folder)
    
def show_results(experiments_folder, dataset):
    generational = load_genetic(experiments_folder, dataset)
    print("Evolved til generation", generational.generation)
    winner = generational.best_individual['winner']
    print("Winner:\n", winner)
    fit = generational.best_individual['best_fit']
    print("Val accuracy: ",1 - fit)
    winner = generational.best_individual['winner']
    level1_fit = generational.history_fitness[winner.__repr__()]
    print("Val accuracy first level: ",1 - level1_fit)
    level2_fit = generational.history_precision_fitness[winner.__repr__()]
    print("Val accuracy second level: ",1 - level2_fit)
    test = generational.best_individual["test"]
    print("Test accuracy:", 1 - test)
    return generational

def train_bads(experiments_folder):
    experiments = os.listdir(experiments_folder)
    for exp in experiments:
        exp_folder = os.path.join(experiments_folder, exp)
        datasets = os.listdir(exp_folder)
        for dataset in datasets:
            generational = load_genetic(exp_folder, dataset)
            winner = generational.best_individual['winner']
            print("Winner:\n", winner)
            fit = generational.best_individual['best_fit']
            print("Val accuracy: ",1 - fit)
            winner = generational.best_individual['winner']
            level1_fit = generational.history_fitness[winner.__repr__()]
            print("Val accuracy first level: ",1 - level1_fit)
            level2_fit = generational.history_precision_fitness[winner.__repr__()]
            print("Val accuracy second level: ",1 - level2_fit)
            test = generational.best_individual["test"]
            print("Test accuracy:", test)

            if test > 0.1:
                print("\nTraining model")
                print("Folder:", exp_folder)
                print("Dataset:", dataset)
                fitness_file = generational.fitness_evaluator.fitness_file
                fitness = FitnessGrow.load(fitness_file)
                fitness.verb = True
                score = fitness.calc(winner, test=False, precise_mode=True)
                
folder = sys.argv[1]
train_bads(folder)
