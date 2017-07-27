import sys
sys.path.insert(0,'..')
import dataset
class DatasetB(dataset.Dataset):
    def next_batch_train(self, batch_size):
        return "usa o dataset B com batch size " + str(batch_size)
    def next_batch_validation(self, batch_size):
            return "usa o dataset B com batch size " + str(batch_size)