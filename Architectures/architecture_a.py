import sys
import architecture
class ArchitectureA(architecture.Architecture):
    def prediction(self, sample, training=False):
        return "faz a predicao usando A"
