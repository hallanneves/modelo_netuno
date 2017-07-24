#import numpy as np
#import tensorflow as tf
import abc

class Dataset(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_batch(self, batch_size):
        """

        """
        pass
