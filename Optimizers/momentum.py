import optimizer
import tensorflow as tf


class MomentumOptimizer(optimizer.Optimizer):
    def __init__(self):
        parameters_list = ["learning_rate", "momentum"]
        self.open_config(parameters_list)
        self.optimizer = tf.train.MomentumOptimizer(self.config_dict["learning_rate"],
                                                     self.config_dict["momentum"])
    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=1, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None, grad_loss=None):
        return self.optimizer.minimize(loss, global_step, var_list,
                                       gate_gradients, aggregation_method,
                                       colocate_gradients_with_ops, name,
                                       grad_loss)
