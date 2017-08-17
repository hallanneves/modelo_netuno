import sys
import dataset
class DatasetA(dataset.Dataset):
    def __init__(self):
        parameters_list = ["batch_size", "batch_size_val", "variable_names"]
        self.open_config(parameters_list)
    def next_batch_train(self):
        return "usa o dataset A com batch size " + str(self.config_dict["batch_size"]), "ground_truth"
    