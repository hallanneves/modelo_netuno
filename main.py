import sys
import getopt
import Architectures
import architecture
import Datasets
import dataset

def process_arguments(argv):
    architectures_list = architecture.Architecture.__subclasses__()
    datasets_list = dataset.Dataset.__subclasses__()
    architecture_name_list = [architecture.__name__ for architecture in architectures_list]
    dataset_name_list = [dataset.__name__ for dataset in datasets_list]
    try:
        long_options = ["help", "architecture=", "dataset=", "dataset_manager="]
        opts, _ = getopt.getopt(argv, "ha:d:m:", long_options)
        if opts == []:
            print('main.py -a <architecture> -d <dataset> -m <dataset_manager>')
            sys.exit(2)
    except getopt.GetoptError:
        print('main.py -a <architecture> -d <dataset> -m <dataset_manager>')
        sys.exit(2)

    dataset_name = None
    architecture_name = None
    dataset_manager_name = None

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('main.py -a <architecture> -d <dataset> -m <dataset_manager>')
            sys.exit()
        elif opt in ("-a", "--architecture"):
            if arg in architecture_name_list:
                architecture_name = arg
            else:
                print(str(arg)+" is not a valid architecture name.")
                sys.exit(2)
        elif opt in ("-d", "--dataset"):
            if arg in dataset_name_list:
                    dataset_name = arg
            else:
                print(str(arg)+" is not a valid dataset name.")
                sys.exit(2)
        elif opt in ("-m", "--dataset_manager"):
            dataset_manager_name = arg

    return architecture_name, dataset_name, dataset_manager_name

if __name__ == "__main__":
    ARCHITECTURE_NAME, DATASET_NAME, DATASET_MANAGER_NAME = process_arguments(sys.argv[1:])
    architectures_list = architecture.Architecture.__subclasses__()
    datasets_list = dataset.Dataset.__subclasses__()
    for architecture in architectures_list:
        if architecture.__name__ == ARCHITECTURE_NAME:
            ARCHITECTURE = architecture()

    for dataset in datasets_list:
        if dataset.__name__ == DATASET_NAME:
            DATASET = dataset()

    RES = ARCHITECTURE.prediction('dUMMY', True)
    print(RES)
    RES = DATASET.next_batch(24)
    print(RES)
