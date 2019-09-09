"""
SA NN training on HTRU2 data
"""

import sys

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule
import opt.SimulatedAnnealing as SimulatedAnnealing
from func.nn.activation import RELU
from base import *

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 11
HIDDEN_LAYER1 = 50
HIDDEN_LAYER2 = 50
HIDDEN_LAYER3 = 50
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


def main(CE):
    """Run this experiment"""
    training_ints = initialize_instances(TRAIN_DATA_FILE)
    testing_ints = initialize_instances(TEST_DATA_FILE)
    validation_ints = initialize_instances(VALIDATE_DATA_FILE)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(0.001, 50, 0.000001)
    oa_name = "SA_{}".format(CE)
    with open(OUTFILE.format(oa_name), 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, HIDDEN_LAYER3, OUTPUT_LAYER], relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = SimulatedAnnealing(1E10, CE, nnop)
    train(oa, classification_network, oa_name, training_ints, validation_ints, testing_ints, measure, TRAINING_ITERATIONS, OUTFILE.format(oa_name))


if __name__ == "__main__":
    for CE in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        main(CE)
