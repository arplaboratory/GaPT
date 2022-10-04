import abc
import torch as t
from collections import namedtuple
ContinuousDiscreteModel = namedtuple("ContinuousDiscreteModel", ["P0", "F", "L", "H", "Q"])


class BaseKernel(metaclass=abc.ABCMeta):
    def __init__(self, t0: float = 0., **_kwargs):
        """
        Parameters:
        -----------
        t0: float, optional
        """
        self.t0 = t0
        self.dtype = t.float64

    @abc.abstractmethod
    def get_sde(self) -> ContinuousDiscreteModel:
        """
        Creates the linear time invariant continuous discrete system associated to the stationary kernels at hand

        Returns
        -------
        sde: ContinuousDiscreteModel
            The associated LTI model
        """

