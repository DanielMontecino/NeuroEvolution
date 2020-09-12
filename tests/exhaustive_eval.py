import os
from time import time
import numpy as np
import sys
sys.path.append('../')

from utils.codification_cnn import FitnessCNNParallel
from utils.codification_grew import FitnessGrow, Merger, ChromosomeGrow, Concatenation, Sum, IdentityGrow, AvPooling
from utils.codification_grew import Inputs, MaxPooling, AvPooling, OperationBlock, CNNGrow, IdentityGrow, HyperParams

from utils.datamanager import DataManager
from GA.geneticAlgorithm import TwoLevelGA
from keras.models import load_model


data_folder =  '../../datasets/MNIST_variations'
command = 'python3 ../train_gen.py'
verbose = 0

dataset = 'MRDBI'
experiments_folder = '../../experiments/test_validation3/'

gpus = 4
batch = 40

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



# Chromosome parameters
Merger._projection_type = ['normal', 'extend'][1]

HyperParams._GROW_RATE_LIMITS = [2, 4.5]
HyperParams._N_CELLS = [1, 2]
HyperParams._N_BLOCKS = [2]
HyperParams._STEM = [32, 45]
HyperParams._MAX_WU = 0.5
HyperParams._LR_LIMITS = [-9, -2] # [-9, -3]

CNNGrow.filters_mul_range = [0.2, 1.2]
CNNGrow.possible_activations = ['relu', 'elu']
CNNGrow.dropout_range = [0.0, 0.7]
CNNGrow.possible_k = [1, 3, 5]


def get_params(gen_list):
    params = {}
    for gen in gen_list:
        if ':' in gen:
            name, val = gen.split(':')
            if not val.isalpha():
                val = float(val) if '.' in val else int(val)
            else:
                pass
            val = np.log(val) / np.log(2) if name == 'LR' else val
            params[name] = val
    return params
    
def get_hp_from_line(_line):
    params = _line.split('|')    
    p_dict = get_params(params)
    hp = HyperParams(grow_rate=p_dict['GR'], n_cells=p_dict['CELL'], n_blocks=p_dict['BLOCK'], 
                     stem=p_dict['STEM'], lr=p_dict['LR'], warmup=p_dict['WU'])
    return hp

def get_operation_from_line(line):
    '||CNN|F:1.2|K:1|A:relu|D:0.05||woCAT||1||'
    op, merge, inputs = line.split('||')[1:-1]
    
    # encoding inputs
    inputs = np.array([int(c) for c in inputs])
    inputs = Inputs(inputs_array=inputs)
    
    # encoding merging
    merge = Concatenation() if 'CAT' in merge else Sum()
    
    # encoding operation
    op_type = op.split('|')[0]
    op_params = op.split('|')[1::]     
        
    if 'AP' == op_type:
        val = op_params[0]
        op = AvPooling(size=int(val))
    elif 'MP' == op_type:
        val = op_params[0]
        op = MaxPooling(size=int(val))
    elif 'CNN' == op_type:
        p_dict = get_params(op_params)
        op = CNNGrow(filter_mul=p_dict['F'], kernel_size=p_dict['K'], activation=p_dict['A'], dropout=p_dict['D'])
    else:
        op = IdentityGrow()
                    
    ops = [operation.random() if operation._type != op_type else op for operation in OperationBlock._operations]
    
    final_op = OperationBlock(operation_type=op_type, ops=ops, concatenation=merge, inputs=inputs)
    return final_op

def str_to_chromosome(string):
    lines = string.split('\n')
    nodes = []
    hp = None
    for line in lines:
        if line.replace(' ', '') == '':
            continue
        if 'HP' in line:
            hp = line
            break
        else:
            nodes.append(line)
    hp = get_hp_from_line(hp)
    ops = [get_operation_from_line(line) for line in nodes]
    return ChromosomeGrow(blocks=ops, n_blocks=len(ops), hparams=hp)



exps = os.listdir(experiments_folder)
for exp in exps:
    exp_folder = os.path.join(experiments_folder, exp, dataset)
    folder = os.path.join(exp_folder, 'genetic')
    fitness_folder = exp_folder
    fitness_file = os.path.join(fitness_folder, 'fitness_example') 
    
    if not os.path.isfile(fitness_file):
        dm = DataManager(dataset, clases=[], folder_var_mnist=data_folder, num_clases=10)  #,  max_examples=15000)
        data = dm.load_data()
        fitness_cnn = FitnessGrow() 
        fitness_cnn.set_params(data=data, verbose=verbose, batch_size=batch_size, reduce_plateau=redu_plat,
                       epochs=epochs, cosine_decay=cosine_dec, early_stop=early_stop, 
                       warm_epochs=warm_up_epochs, base_lr=base_lr, smooth_label=smooth, find_lr=lr_find,
                       precise_epochs=precise_eps, include_time=include_time, test_eps=test_eps,  augment=augment)

        fitness_cnn.save(fitness_file)
    
    models_folder = os.path.join(fitness_folder, 'all_models')
    os.makedirs(models_folder, exist_ok=True)
    fitness = FitnessCNNParallel()
    fitness.set_params(chrom_files_folder=fitness_folder, fitness_file=fitness_file, max_gpus=gpus,
                           fp=32, main_line=command)
    
    try:
        generational = TwoLevelGA.load_genetic_algorithm(folder=folder) 
    except:
        print("Genetic Model not found!")
        continue
    original_filename = generational.filename
    filename = original_filename.split('genetic')[-1]
    generational.filename = folder + filename
    if not hasattr(generational,'exhaustive_eval'):
        generational.exhaustive_eval = {}
    
    for k, individual in enumerate(generational.history_fitness.keys()):
        if not individual in generational.exhaustive_eval.keys():
            print("adding individual %d" % k)
            chromosome = str_to_chromosome(individual)
            file_model = os.path.join(models_folder, "%d.hdf5" % k)
            tmp_dict = {'n':k, 'chromosome':chromosome, 'test':None, 'val':None, 'file_model':file_model}
            generational.exhaustive_eval[individual] = tmp_dict
            
    generational.print_genetic("\n\nStarting exhaustive evaluation.\n\n")
    generational.print_genetic("Evaluating %d model" % len(generational.exhaustive_eval.keys()))
    generational.maybe_save_genetic_algorithm()
    
    ti = time()
    models_to_eval = []
    file_models = []
    ids = []
    for individual, tmp_dict in generational.exhaustive_eval.items():
        if tmp_dict['val'] is not None and os.path.isfile(tmp_dict['file_model']):
            continue
        models_to_eval.append(tmp_dict['chromosome'])        
        file_models.append(tmp_dict['file_model'])
        print("Saving at", tmp_dict['file_model'])
        ids.append(tmp_dict['n'])        
        if len(models_to_eval) == batch:
            scores = fitness.eval_list(chromosome_list=models_to_eval, test=False, precise_mode=True,
                               file_model_list=file_models)
            for i in range(len(models_to_eval)):
                indiv_i = models_to_eval[i].__repr__()
                score_i = scores[i]
                generational.exhaustive_eval[indiv_i]['val'] = score_i
            generational.print_genetic("Elapsed time: %0.2f" % ((time() - ti) / 60))
            generational.maybe_save_genetic_algorithm()
            models_to_eval = []
            file_models = []
            ids = []
        
    scores = fitness.eval_list(chromosome_list=models_to_eval, test=False, precise_mode=True,
                               file_model_list=file_models)
    for i in range(len(models_to_eval)):
        indiv_i = models_to_eval[i].__repr__()
        score_i = scores[i]
        generational.exhaustive_eval[indiv_i]['val'] = score_i
        
    generational.print_genetic("Elapsed time: %0.2f" % ((time() - ti) / 60))
    generational.maybe_save_genetic_algorithm()
        
    # Evaluate test
    
    fitness = FitnessGrow.load(fitness_file)
    print(type(fitness.data[1][0]), fitness.data[1][0].shape)
    print(type(fitness.data[1][1]), fitness.data[1][1].shape)
    for indiv in generational.exhaustive_eval.keys():
        tmp_dict = generational.exhaustive_eval[indiv]
        file_model = tmp_dict['file_model']
        model = load_model(file_model)
        score = 1 - model.evaluate(fitness.data[1][0], fitness.data[1][1], verbose=0)[1]
        generational.exhaustive_eval[indiv]['test'] = score
        print("\nModel N°%d" % tmp_dict['n'])
        print(indiv)
        print("Val: %0.2f. Test: %0.2f" % (tmp_dict['val']*100, tmp_dict['test']*100))

    generational.print_genetic("Finish in time: %0.2f" % ((time() - ti) / 60))
    generational.filename = original_filename
    generational.maybe_save_genetic_algorithm()
        
