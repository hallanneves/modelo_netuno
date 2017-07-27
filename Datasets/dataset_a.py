import sys
import dataset
class DatasetA(dataset.Dataset):
    def next_batch(self, batch_size):
        return "usa o dataset A com batch size " + str(batch_size)
