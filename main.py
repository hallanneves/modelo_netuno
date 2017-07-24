import sys
import getopt
import Architectures
import architecture
import Datasets
import dataset
import DatasetManagers
import dataset_manager

def get_implementation(parent_class, child_class_name):
    for child_class in parent_class.__subclasses__():
        if child_class.__name__ == child_class_name:
            return child_class()
    return None

def process_args(argv):

    try:
        long_opts = ["help", "architecture=", "dataset=", "dataset_manager="]
        opts, _ = getopt.getopt(argv, "ha:d:m:", long_opts)
        if opts == []:
            print('main.py -a <architecture> -d <dataset> -m <dataset_manager>')
            sys.exit(2)
    except getopt.GetoptError:
        print('main.py -a <architecture> -d <dataset> -m <dataset_manager>')
        sys.exit(2)
    print(opts)
    opt_values = {}
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('main.py -a <architecture> -d <dataset> -m <dataset_manager>')
            sys.exit()
        elif opt in ("-a", "--architecture"):
            if get_implementation(architecture.Architecture, arg) != None:
                opt_values['architecture_name'] = arg
            else:
                print(str(arg)+" is not a valid architecture name.")
                sys.exit(2)
        elif opt in ("-d", "--dataset"):
            if get_implementation(dataset.Dataset, arg) != None:
                opt_values['dataset_name'] = arg
            else:
                print(str(arg)+" is not a valid dataset name.")
                sys.exit(2)
        elif opt in ("-m", "--dataset_manager"):
            if get_implementation(dataset_manager.DatasetManager, arg) != None:
                opt_values['dataset_manager_name'] = arg
            else:
                print(str(arg)+" is not a valid dataset manager name.")
                sys.exit(2)
    return opt_values

if __name__ == "__main__":
    OPT_VALUES = process_args(sys.argv[1:])
    ARCH_NM = OPT_VALUES['architecture_name']
    DATASET_NM = OPT_VALUES['dataset_name']
    DATASET_MAN_NM = OPT_VALUES['dataset_manager_name']
    ARCHITECTURE = get_implementation(architecture.Architecture, ARCH_NM)
    DATASET = get_implementation(dataset.Dataset, DATASET_NM)
    DATASET_MANAGER = get_implementation(dataset_manager.DatasetManager,
                                         DATASET_MAN_NM)
    RES = ARCHITECTURE.prediction('dUMMY', True)
    print(RES)
    RES = DATASET.next_batch(batch_size=24)
    print(RES)
    RES = DATASET_MANAGER.load_data([1, 2, 3])
    print(RES)
