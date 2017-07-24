import sys
sys.path.insert(0,'..')
import dataset_manager

class DatasetManagerB(dataset_manager.DatasetManager):
    def load_data(self, keys):
        return "Foi usado o dataset manager B para carregar as chaves " + str(keys)

    def store_data(self, data):
        def load_data(self, keys):
            return "Foi usado o dataset manager B para salvar as chaves " + str(keys)
    