import loss
import tensorflow as tf
class MSE(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.open_config(parameters_list)
    def evaluate(self, architecture_output, target_output):
        return tf.reduce_mean(tf.squared_difference(architecture_output, target_output), reduction_indices=[1,2,3])