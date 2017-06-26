import numpy as np
import tensorflow as tf
from abc import ABCMeta

class Dataset(metaclass=ABCMeta):
    
    
    @abstractmethod
    def next_batch(self):
        """
        """
        pass
    

