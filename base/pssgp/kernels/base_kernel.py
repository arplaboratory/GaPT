import abc
from collections import namedtuple
import tensorflow as tf
import gpflow
import torch as t
from gpflow.kernels import Kernel
from typing import List, Optional
from base.pssgp.kalman import LGSSM
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

    def _get_lssm_spec(self, dim, T):
        """
        TODO: WRITE DESCRIPTION
        """
        P0_spec = tf.TensorSpec((dim, dim), dtype=self.dtype)
        Fs_spec = tf.TensorSpec((T, dim, dim), dtype=self.dtype)
        Qs_spec = tf.TensorSpec((T, dim, dim), dtype=self.dtype)
        H_spec = tf.TensorSpec((1, dim), dtype=self.dtype)
        R_spec = tf.TensorSpec((1, 1), dtype=self.dtype)

        return LGSSM(P0_spec, Fs_spec, Qs_spec, H_spec, R_spec)

    def _get_ssm(self, sde, ts, R, t0=0.):
        """
        TODO: WRITE DESCRIPTION
        """
        n = tf.shape(sde.F)[0]
        t0 = tf.reshape(tf.convert_to_tensor(t0, dtype=self.dtype), (1, 1))

        ts = tf.concat([t0, ts], axis=0)
        dts = tf.reshape(ts[1:] - ts[:-1], (-1, 1, 1))
        Fs = tf.linalg.expm(dts * tf.expand_dims(sde.F, 0))
        zeros = tf.zeros_like(sde.F)
        Phi = tf.concat(
            [tf.concat([sde.F, sde.L @ tf.matmul(sde.Q, sde.L, transpose_b=True)], axis=1),
             tf.concat([zeros, -tf.transpose(sde.F)], axis=1)],
            axis=0)
        AB = tf.linalg.expm(dts * tf.expand_dims(Phi, 0))
        AB = AB @ tf.concat([zeros, tf.eye(n, dtype=self.dtype)], axis=0)
        Qs = tf.matmul(AB[:, :n, :], Fs, transpose_b=True)
        return LGSSM(sde.P0, Fs, Qs, sde.H, R)

    def get_ssm(self, ts, R, t0=0.):
        """
        Creates the linear Gaussian state space model associated to the stationary kernels at hand

        Parameters
        ----------
        ts: tf.Tensor
            The times at which we have observations
        R: tf.Tensor
            The observation covariance
        t0: float
            Starting point of the model

        Returns
        -------
        lgssm: ContinuousDiscreteModel
            The associated state space model
        """
        sde = self.get_sde()
        ssm = self._get_ssm(sde, ts, R, t0)
        return ssm

    def __add__(self, other):
        return SDESum([self, other])  # noqa: don't complain Pycharm, I know what's good for you.

    def __mul__(self, other):
        return SDEProduct([self, other])  # noqa: don't complain Pycharm, I know what's good for you.

    @abc.abstractmethod
    def get_spec(self, **_kwargs):
        return NotImplementedError


def _sde_combination_init(self, kernels: List[Kernel], name: Optional[str] = None, **kargs):
    if not all(isinstance(k, BaseKernel) for k in kernels):
        raise TypeError("can only combine SDE Kernel instances")  # pragma: no cover
    gpflow.kernels.Sum.__init__(self, kernels, name)
    BaseKernel.__init__(self, **kargs)


def block_diag(arrs):
    xdims = [tf.shape(a)[0] for a in arrs]
    ydims = [tf.shape(a)[1] for a in arrs]
    out_dtype = arrs[0].dtype
    out = tf.zeros((0, sum(ydims)), dtype=out_dtype)
    ydim = sum(ydims)
    r, c = 0, 0
    for i, (rr, cc) in enumerate(zip(xdims, ydims)):
        paddings = [[0, 0],
                    [c, ydim - c - cc]]

        out = tf.concat([out, tf.pad(arrs[i], paddings)], 0)
        r = r + rr
        c = c + cc
    return out
