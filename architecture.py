import numpy as np
import tensorflow as tf
from abc import ABCMeta

class Architecture(metaclass=ABCMeta):
    
    
    @abstractmethod
    def prediction(self, input, training=False):
        pass
    

