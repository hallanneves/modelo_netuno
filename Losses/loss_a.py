import sys
import loss

class LossA(loss.Loss):
    def __init__(self):
        parameters_list = ["WEIGHTS_FILE", "use_discriminator", "restore_discriminator"]
        self.open_config(parameters_list)
    def evaluate(self, architecture_output, target_output):
        return "faz a predicao usando A"
