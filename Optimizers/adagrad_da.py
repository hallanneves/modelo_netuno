import optimizer
import tensorflow as tf


class AdagradDAOptimizer(optimizer.Optimizer):
    def __init__(self):
        parameters_list = ["learning_rate", "global_step",
                           "initial_accumulator_value",
                           "l1_regularization_strength",
                           "l2_regularization_strength"]
        self.open_config(parameters_list)
        self.optimizer = tf.train.AdagradDAOptimizer(self.config_dict["learning_rate"],
                                                     self.config_dict["global_step"],
                                                     self.config_dict["initial_accumulator_value"],
                                                     self.config_dict["l1_regularization_strength"],
                                                     self.config_dict["l2_regularization_strength"])
    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=1, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None, grad_loss=None):
        return self.optimizer.minimize(loss, global_step, var_list,
                                       gate_gradients, aggregation_method,
                                       colocate_gradients_with_ops, name,
                                       grad_loss)
