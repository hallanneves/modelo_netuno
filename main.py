import tensorflow as tf
import numpy as np
import sys
import getopt
import Architectures
import architecture
import Datasets
import dataset
import DatasetManagers
import dataset_manager
import Losses
import loss
def get_implementation(parent_class, child_class_name):
    for child_class in parent_class.__subclasses__():
        if child_class.__name__ == child_class_name:
            return child_class()
    return None

def arg_validation(arg, cla):
    if get_implementation(cla, arg) != None:
        return arg
    else:
        print(str(arg)+" is not a valid " + cla.__module__ + " name.")
        sys.exit(2)


def process_args(argv):

    try:
        long_opts = ["help", "architecture=", "dataset=",
                     "dataset_manager=", "loss="]
        opts, _ = getopt.getopt(argv, "ha:d:m:g:l:", long_opts)
        if opts == []:
            print('main.py -a <architecture> -d <dataset> -g <dataset_manager>')
            sys.exit(2)
    except getopt.GetoptError:
        print('main.py -a <architecture> -d <dataset> -g <dataset_manager>')
        sys.exit(2)
    opt_values = {}
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('main.py -a <architecture> -d <dataset> -g <dataset_manager>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            if arg in ('train', 'evaluate', 'restore'):
                opt_values['execution_mode'] = arg
            else:
                print(arg + 'is not a possible execution mode')

                sys.exit(2)
        elif opt in ("-a", "--architecture"):
            opt_values['architecture_name'] = \
                                arg_validation(arg, architecture.Architecture)
        elif opt in ("-d", "--dataset"):
            opt_values['dataset_name'] = arg_validation(arg, dataset.Dataset)
        elif opt in ("-g", "--dataset_manager"):
            opt_values['dataset_manager_name'] = \
                        arg_validation(arg, dataset_manager.DatasetManager)
        elif opt in ("-l", "--loss"):
            opt_values['loss'] = \
                        arg_validation(arg, loss.Loss)
    return opt_values

if __name__ == "__main__":
    OPT_VALUES = process_args(sys.argv[1:])
    ARCH_NM = OPT_VALUES['architecture_name']
    DATASET_NM = OPT_VALUES['dataset_name']
    # DATASET_MAN_NM = OPT_VALUES['dataset_manager_name']
    # EXECUTION_MODE = OPT_VALUES['execution_mode']
    # LOSS_NM = OPT_VALUES['loss']
    ARCHITECTURE = get_implementation(architecture.Architecture, ARCH_NM)
    DATASET = get_implementation(dataset.Dataset, DATASET_NM)
    # DATASET_MANAGER = get_implementation(dataset_manager.DatasetManager,
    #                                      DATASET_MAN_NM)
    # LOSS = get_implementation(loss.Loss, LOSS_NM)
    # print(EXECUTION_MODE)
    RES = ARCHITECTURE.prediction('dUMMY', True)
    print(RES)
    RES = DATASET.next_batch(batch_size=24)
    print(RES)
    # RES = DATASET_MANAGER.load_data([1, 2, 3])
    # print(RES)
