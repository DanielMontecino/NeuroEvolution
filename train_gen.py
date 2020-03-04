import argparse
from utils.utils import verify_free_gpu_memory
from utils.codifications import Chromosome, Fitness
from time import sleep, time
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
abs_ti = time()

chromosome = Chromosome.load(args['gen_file'])
print(chromosome)
fitness = Fitness.load(args['fitness_file'])

while not verify_free_gpu_memory()[0]:
    sleep(3)
    print("Waiting 3 seconds for a gpu...")
gpu_id = verify_free_gpu_memory()[1]
print("GPU AVAILABLE: :/GPU %d" % gpu_id)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_time = time()
try:
    score = fitness.calc(chromosome, test=args['test'], file_model='./model_acc_gpu%d.hdf5' % gpu_id,
                         fp=args['float_precision'], precise_mode=args['precise_mode'])
except:
    score = 1
training_time = (time() - training_time) / 60.
print()
with open("%s_score" % args['gen_file'], 'w') as f:
    f.write("\nScore: %0.6f" % score)

abs_ti = (time() - abs_ti) / 60.
hours = abs_ti // 60
minutes = abs_ti % 60
work_directory = os.path.split(args['gen_file'])[0]
record_file = os.path.join(work_directory, 'RECORD')
with open(record_file, 'a') as f:
    f.write("-" * 40 + "\n")
    f.write(f"{chromosome.__repr__()}\n")
    if abs_ti > 10:
        f.write("Taking too much time\n")
    f.write(f"Precision:\t{args['precise_mode']}\n")
    f.write(f"Score:\t\t{score.__format__('2.4')}\n")
    f.write("Training time:\t%d:%d\n" % (hours, minutes))
print("Score: %0.4f" % score)


