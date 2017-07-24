import sys
sys.path.insert(0,'..')
import dataset
class DatasetA(dataset.Dataset):
    def next_batch(self, batch_size):
        return "usa o dataset A com batch size " + str(batch_size)
