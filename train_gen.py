import argparse
from ast import literal_eval
from utils.utils import verify_free_gpu_memory
from utils.codification_cnn import CNNLayer, NNLayer, ChromosomeCNN, FitnessCNN
from time import sleep
import os


parser = argparse.ArgumentParser(description='Train a gen of a CNN.')

parser.add_argument('-gf', '--gen_file', type=str, required=True,
                    help='text file who contains the genetic encoding of the CNN to train')
parser.add_argument('-ff', '--fitness_file', type=str, required=True,
                    help='file that contains the fitness object to use in the training and evaluating process')
parser.add_argument('-t', '--test', type=bool, default=False,
                    help="If use the test dataset to evaluate the model trained")

parser.add_argument('-fp', '--float_precision', type=int, default=32,
                    help='Bits to use in float precision. FP32 is more accurate, but FP is faster and use less memory')

parser.add_argument('-pm', '--precise_mode', type=bool, default=False,
                    help="Train the gen with a secondary configuration, in order to make a more precise calculation"
                         " of the fitness")

args = vars(parser.parse_args())


def get_chromosome_from_file(filename):
    cnn_layers = []
    nn_layers = []
    with open(filename, 'r') as f:
        for line in f:
            params = line.split('|')            
            if 'CNN' == params[0]:
                filters = int(params[1].split(':')[1])
                kernel = literal_eval(params[2].split(':')[1])
                activation = params[3].split(':')[1]
                dropout = float(params[4].split(':')[1])
                maxpool = bool(int(params[5].split(':')[1]))
                cnn_layers.append(CNNLayer(filters, kernel, activation, dropout, maxpool))
            if 'NN' == params[0]:
                units = int(params[1].split(':')[1])
                activation = params[2].split(':')[1]
                dropout = float(params[3].split(':')[1])
                nn_layers.append(NNLayer(units, activation, dropout))
    return ChromosomeCNN(cnn_layers, nn_layers)


chromosome = get_chromosome_from_file(args['gen_file'])
print(chromosome)
fitness = FitnessCNN.load(args['fitness_file'])

while not verify_free_gpu_memory()[0]:
    sleep(3)
    print("Waiting 3 seconds for a gpu...")
gpu_id = verify_free_gpu_memory()[1]
print("GPU AVAILABLE: :/GPU %d" % gpu_id)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

score = fitness.calc(chromosome, test=args['test'], file_model='./model_acc_gpu%d.hdf5' % gpu_id,
                     fp=args['float_precision'], precise_mode=args['precise_mode'])
print()
with open(args['gen_file'], 'a') as f:
    f.write("\nScore: %0.6f" % score)
print("Score: %0.4f" % score)


