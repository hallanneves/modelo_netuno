#import numpy as np
#import tensorflow as tf
import abc

class Dataset(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_batch_train(self, batch_size, num_epochs=None):
        """

        """
        pass

    @abc.abstractmethod
    def next_batch_validation(self, batch_size, num_epochs=None):
        """

        """
        pass
    # @abc.abstractmethod
    # def get_N_images_train(self):
    #     pass

    # @abc.abstractmethod
    # def get_N_images_validation(self):
    #     pass
        