#import numpy as np
#import tensorflow as tf
import abc

class Architecture(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def prediction(self, sample, training=False):
        """This is a abstract method for architectures prediction.

        Each architecture must implement this method. Depending on
        each diferent implementation the output shape varies. So
        the loss must be chosen acoording withe achitecture
        implementation.
        In a similar way the architecture implementation depends on
        the dataset shape.

        Args:
            sample: networks input tensor
            training: boolean value indication if this prediction is
            beeing used on training or not

        Returns:
            achitecture output: networks output tensor

        """
        pass
