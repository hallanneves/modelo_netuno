#import numpy as np
#import tensorflow as tf
import abc

class DBManager(metaclass=abc.ABCMeta):
    '''
    Abstract class designed for loading and storing
    objects from memory. 
    '''
    


    @abc.abstractmethod
    def load_data(self, keys):
        """Load Data

        responsible for loading memory objects into numpy
        arrays

        Args:
            self: the instance
            keys: intern keys for object retrieval

        Returns:
            numpy array of the chosen data
        """
        pass


    @abc.abstractmethod
    def store_data(self, data):
        """Store Data

        responsible for converting and storing numpy objects
        into hard drive files

        Args:
            self: the instance
            data: data to be stored in the DBManager format

        Returns:
            void
        """
        pass
    