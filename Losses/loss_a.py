import sys
import loss

class LossA(loss.Loss):
    def evaluate(self, architecture_output, target_output):
        return "faz a predicao usando A"
