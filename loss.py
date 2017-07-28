import abc

class Loss(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evaluate(self, architecture_output, target_output):
        """This is a abstract method for defining loss functions.

        Each loss must implement this method. Depending on the
        architecture the output shape varies. Depending on
        the output shape a determined loss can or not be used.

        Args:
            architecture_output: architecture output tensor
            target_output: desired output must have the same
            shape as architecture_output
        Returns:
            loss output:

        """
        pass
