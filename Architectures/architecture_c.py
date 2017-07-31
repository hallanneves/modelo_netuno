
import architecture
class ArchitectureC(architecture.Architecture):
    def __init__(self):
        parameters_list = ['example', 'input_size', 'output_size',
                                'depth_size', 'summary_writing_period']
        self.open_config(parameters_list)

    def prediction(self, sample, training=False):
        return "faz a predicao usando C "
