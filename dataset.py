#import numpy as np
#import tensorflow as tf
import abc
import sys
import json

class Dataset(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_batch_train(self):
        """

        """
        pass

    # @abc.abstractmethod
    # def next_batch_validation(self, batch_size, num_epochs=None):
    #     """

    #     """
    #     pass


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

    # @abc.abstractmethod
    # def get_N_images_train(self):
    #     pass

    # @abc.abstractmethod
    # def get_N_images_validation(self):
    #     pass
        