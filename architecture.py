#import numpy as np
#import tensorflow as tf
import json
import sys
import abc

class Architecture(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def prediction(self, sample, training=False):
        """This is a abstract method for architectures prediction.

        Each architecture must implement this method. Depending on
        each diferent implementation the output shape varies. So
        the loss must be chosen acoording with the achitecture
        implementation.
        In a similar way the architecture implementation depends on
        the dataset shape.

        Args:
            sample: networks input tensor
            training: boolean value indication if this prediction is
            being used on training or not

        Returns:
            achitecture output: networks output tensor

        """
        pass

    def verify_config(self, parameters_list):
        for parameter in parameters_list:
            if parameter not in self.config_dict:
                raise Exception('Config: ' + parameter + ' is necessary for ' +
                                self.__class__.__name__ + ' execution.')

    def open_config(self, parameters_list):
        config_filename = sys.modules[self.__module__].__file__[:-3]+'.json'
        with open(config_filename) as config_file:
            self.config_dict = json.load(config_file)
        self.verify_config(parameters_list)
