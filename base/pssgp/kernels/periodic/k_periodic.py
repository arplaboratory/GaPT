import numpy as np
import torch as t
import math
from typing import Tuple, Union, List
from scipy.special import factorial, comb
from base.pssgp.kernels.base_kernel import BaseKernel, ContinuousDiscreteModel


class Periodic(BaseKernel):
    """
    The periodic family of kernels. The canonical form (based on the
    SquaredExponential kernels) can be found in Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    The following implementation is inspired by the procedure explained by Arno Solin and
    Simo sarkka: 'Explicit Link Between Periodic Covariance Functions
    and State Space Models'
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.647.357&rep=rep1&type=pdf
    """

    def __init__(self, lengthscale_base,
                 lengthscale_per,
                 period_per: Union[float, List[float]] = 1.0,
                 variance: float = 1.0,
                 **kwargs):

        super().__init__(t0=0.)
        self._order = kwargs.pop('order', 6)
        self.lengthscale_base = np.float64(lengthscale_base)
        self.lengthscale_per = np.float64(lengthscale_per)
        self.variance = np.float64(variance)
        self.period_per = np.float64(period_per)

    def _get_offline_coeffs(self) -> Tuple[t.tensor, t.tensor, t.tensor]:
        """
        Get coefficients which are independent of parameters (ell, sigma, and period). That are, fixed.

        Returns
        -------
        b: np.ndarray
        K: np.ndarray
        div_facto_K: np.ndarray

        """
        N = self._order
        r = np.arange(0, N + 1)
        J, K = np.meshgrid(r, r)
        div_facto_K = 1 / factorial(K)
        # Get b(K, J)
        b = 2 * comb(K, np.floor((K - J) / 2) * (J <= K)) / \
            (1 + (J == 0)) * (J <= K) * (np.mod(K - J, 2) == 0)

        # convert to Tensors
        b = t.tensor(b, dtype=self.dtype)
        K = t.tensor(K, dtype=self.dtype)
        div_facto_K = t.tensor(div_facto_K, dtype=self.dtype)
        return b, K, div_facto_K

    def get_sde(self) -> ContinuousDiscreteModel:
        w0 = 2.0 * math.pi / self.period_per
        lengthscales = self.lengthscale_base + self.lengthscale_per
        N = self._order

        # Prepare offline fixed coefficients
        b, K, div_facto_K = self._get_offline_coeffs()

        # F
        op_F = t.tensor([[-self.lengthscale_base, -w0], [w0, -self.lengthscale_base]], dtype=self.dtype)
        op_diag = t.arange(0,  N + 1).to(self.dtype) * t.eye(N + 1, dtype=self.dtype)
        F = t.kron(op_diag, op_F)

        # L
        L = t.eye(2*(N+1), dtype=self.dtype)

        # Pinf
        q2_aT = b * lengthscales ** (-2 * K) * div_facto_K * \
                t.exp(-t.tensor(lengthscales, dtype=self.dtype) ** (-2)) * 2 ** (-K) * self.variance
        q2T = t.sum(q2_aT, dim=0) * t.eye(N + 1, dtype=self.dtype)
        Pinf = t.kron(q2T, t.eye(N, dtype=self.dtype))

        # Q: as F, is computed considering that we use the Matern to generate the quasi periodic version
        # Brownian motion ( from Matern32)
        d = 2
        qq = t.sum(q2_aT, dim=0)[0] * t.eye(N, dtype=self.dtype)
        lamda = math.sqrt(2 * d - 1) / t.tensor(lengthscales, dtype=self.dtype)
        mat_brown_n = t.tensor((2 * lamda) ** (2 * d - 1) * self.variance *
                               math.factorial(d - 1) ** 2 / math.factorial(2 * d - 2), dtype=self.dtype)
        Q = t.kron(mat_brown_n*t.eye(N+1), qq)

        # H
        H = t.kron(t.ones((1, N + 1), dtype=self.dtype), t.tensor([[1, 0]], dtype=self.dtype))
        return ContinuousDiscreteModel(Pinf, F, L, H, Q)

    def get_spec(self, T):
        return self._get_lssm_spec(2 * (self._order + 1), T)
