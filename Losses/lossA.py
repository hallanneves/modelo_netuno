import sys
sys.path.insert(0,'..')
import architecture
class LossesA(architecture.Architecture):
    def prediction(self, sample, training=False):
        return "faz a predicao usando A"
