from abc import ABCMeta
from base.pssgp.kernels.base_kernel import BaseKernel
import numpy as np
import torch as t
from scipy.special import binom
import math
from typing import Tuple


class Matern(BaseKernel):
    """
    This class inherit from the base class "BaseKernel"
    implementing the  Matern kernels-based methods
    """
    def __init__(self, t0):
        super().__init__(t0=t0)

    def _get_transition_matrix(self, lamda: t.Tensor, d: int) -> t.Tensor:
        """
        Description
        ----------
        Method to calculate the F Matrix in the companion form of the LTI-SDE
        """
        lamda = t.tensor(lamda.numpy())
        F = t.diag(t.ones((d - 1,), dtype=self.dtype), 1)
        binomial_coeffs = binom(d, np.arange(0, d, dtype=int)).astype(np.float64)
        binomial_coeffs = t.tensor(binomial_coeffs, dtype=self.dtype)
        lambda_powers = lamda ** np.arange(d, 0, -1, dtype=np.float64)
        update_indices = t.tensor([[d - 1, k] for k in range(d)])
        F[update_indices[:, 0], update_indices[:, 1]] -= lambda_powers * binomial_coeffs
        return F

    def _get_brownian_cov(self, variance: t.Tensor, lamda: t.Tensor, d) -> t.Tensor:
        """
        Description
        ----------
        Method to calculate the q noise
        """
        q = (2 * lamda) ** (2 * d - 1) * variance * math.factorial(d - 1) ** 2 / math.factorial(2 * d - 2)
        return q * t.eye(1, dtype=self.dtype)

    def _get_matern_sde(self, variance: t.Tensor, lengthscales: t.Tensor, d: int) -> Tuple[t.Tensor, ...]:
        """
        TODO: write description

        Parameters
        ----------
        variance
        lengthscales
        d: int
            the exponent of the Matern kernels plus one half
            for instance Matern32 -> 2, this will be used as the dimension of the latent SSM

        Returns
        -------
        F, L, H, Q: tuple of t.Tensor
            Parameters for the LTI sde
        """
        lamda = math.sqrt(2 * d - 1) / lengthscales
        F = self._get_transition_matrix(lamda, d)
        L = t.eye(d, dtype=self.dtype)[d-1, :].unsqueeze(1)
        H = t.eye(d, dtype=self.dtype)[:, 0].unsqueeze(1).t()
        Q = self._get_brownian_cov(variance, lamda, d)
        return F, L, H, Q
