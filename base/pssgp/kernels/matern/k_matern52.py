import torch as t
import logging
from base.pssgp.kernels.base_kernel import ContinuousDiscreteModel
from base.pssgp.kernels.matern.base_matern import Matern
from tools.math_utils import solve_lyap_vec, balance_ss


class Matern52(Matern):
    """
    The Matern 5/2 kernels. Functions drawn from a GP with this kernels are twice
    differentiable. The kernels equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    def __init__(self, variance=1.0, lengthscales=1.0, t0: float = 0., **kwargs):
        super().__init__(t0=t0)
        self.variance = t.tensor(variance, dtype=self.dtype)
        self.lengthscales = t.tensor(lengthscales, dtype=self.dtype)
        self._balancing_iter = kwargs.pop('balancing_iter', 0)


    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q, = self._get_matern_sde(self.variance, self.lengthscales, d=3)
        Q = t.reshape(Q, (1, 1, 1))
        if self._balancing_iter > 0:
            Fb, Lb, Hb, Qb = balance_ss(F, L, H, Q, n_iter=self._balancing_iter)
            P_infty = solve_lyap_vec(Fb, Lb, Qb)
            return ContinuousDiscreteModel(P_infty, Fb, Lb, Hb, Qb)
        elif self._balancing_iter == 0:
            P_infty = solve_lyap_vec(F, L, Q)
            return ContinuousDiscreteModel(P_infty, F, L, H, Q)
        else:
            err_msg = 'The value of argument balancing_iter {} must be integer >= 0'.format(self._balancing_iter)
            logging.error(err_msg)
            raise ValueError(err_msg)

    def get_spec(self, T):
        return self._get_lssm_spec(3, T)
