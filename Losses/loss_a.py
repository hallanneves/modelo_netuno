import sys
import loss

class LossesA(loss.Loss):
    def evaluate(self, architecture_output, target_output):
        return "faz a predicao usando A"
