import torch as t
from base.gapt.kernels.base_kernel import ContinuousDiscreteModel
from base.gapt.kernels.matern.base_matern import Matern


class Matern12(Matern):
    """
    The Matern 1/2 kernels. Functions drawn from a GP with this kernels are not
    differentiable anywhere. The kernels equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    # TODO: remember to update when mixingkernel classes are implemented
    def __init__(self, variance=1.0, lengthscales=1.0, t0: float = 0., **kwargs):
        super().__init__(t0=t0)
        self.variance = t.tensor(variance, dtype=self.dtype)
        self.lengthscales = t.tensor(lengthscales, dtype=self.dtype)

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = self._get_matern_sde(self.variance, self.lengthscales, 1)

        P_infty = (t.tensor((self.variance,), dtype=self.dtype)).unsqueeze(1)
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
