import numpy as np
import torch
import torch as t
import math
from typing import Tuple
from gpflow import config
from tools.math_utils import solve_lyap_vec, balance_ss
from base.pssgp.kernels.base_kernel import BaseKernel, ContinuousDiscreteModel

dtype_torch = t.float64
dtype_numpy = np.dtype(float)


class RBF(BaseKernel):
    """
    The radial basis function (RBF) or squared exponential kernels. The kernels equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernels are infinitely differentiable!
    """

    # TODO: remember to update when mixingkernel classes are implemented
    def __init__(self, variance=1.0, lengthscales=1.0, t0: float = 0., **kwargs):
        self.variance = t.tensor(variance, dtype=dtype_torch)
        self.lengthscales = t.tensor(lengthscales, dtype=dtype_torch)
        self._order = kwargs.pop('order', 6)
        self._balancing_iter = kwargs.pop('balancing_iter', 5)
        super().__init__(t0=t0)

    def get_unscaled_rbf_sde(self) -> Tuple[np.ndarray, ...]:
        """Get un-scaled RBF SDE.
        Pre-computed before loading to tensorflow.

        Parameters
        ----------
        order : int, default=6
            Order of Taylor expansion

        Returns
        -------
        F, L, H, Q : np.ndarray
            SDE coefficients.

        See Also
        --------
        se_to_ss.m
        """
        order = self._order
        dtype = config.default_float()
        B = math.sqrt(2 * math.pi)
        A = np.zeros((2 * order + 1,), dtype=dtype)

        i = 0
        for k in range(order, -1, -1):
            A[i] = 0.5 ** k / math.factorial(k)
            i = i + 2

        q = B / np.polyval(A, 0)

        LA = np.real(A / (1j ** np.arange(A.size - 1, -1, -1, dtype=dtype)))

        AR = np.roots(LA)

        GB = 1
        GA = np.poly(AR[np.real(AR) < 0])

        GA = GA / GA[-1]

        GB = GB / GA[0]
        GA = GA / GA[0]

        F = np.zeros((GA.size - 1, GA.size - 1), dtype=dtype)
        F[-1, :] = -GA[:0:-1]
        F[:-1, 1:] = np.eye(GA.size - 2, dtype=dtype)

        L = np.zeros((GA.size - 1, 1), dtype=dtype)
        L[-1, 0] = 1

        H = np.zeros((1, GA.size - 1), dtype=dtype)
        H[0, 0] = GB

        return F, L, H, q

    def get_sde(self) -> ContinuousDiscreteModel:
        F_, L_, H_, q_ = self.get_unscaled_rbf_sde()
        F = t.tensor(F_, dtype=dtype_torch)
        L = t.tensor(L_, dtype=dtype_torch)
        H = t.tensor(H_, dtype=dtype_torch)
        q = t.tensor(q_, dtype=dtype_torch)
        dim = F.shape[0]
        ell_vec = self.lengthscales ** t.arange(dim, 0, -1, dtype=dtype_torch)
        update_indices = t.tensor([[dim - 1, k] for k in range(dim)]).to(torch.long)
        F[update_indices[:, 0], update_indices[:, 1]] = F[-1, :] / ell_vec
        H = H / (self.lengthscales ** dim)
        Q = self.variance * self.lengthscales * t.reshape(q, (1, 1))
        Fb, Lb, Hb, Qb = balance_ss(F, L, H, Q, n_iter=self._balancing_iter)
        # Fb, Lb, Hb, Qb = F, L, H, Q
        Pinf = solve_lyap_vec(Fb, Lb, Qb)

        return ContinuousDiscreteModel(Pinf, Fb, Lb, Hb, Q)

    def get_spec(self, T):
        return self._get_lssm_spec(self._order, T)
