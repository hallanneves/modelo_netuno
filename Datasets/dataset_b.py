import sys
sys.path.insert(0,'..')
import dataset
class DatasetB(dataset.Dataset):
    def next_batch(self, batch_size):
        return "usa o dataset B com batch size " + str(batch_size)
