from typing import Tuple
import numba as nb
import numpy as np
import torch as t
import copy
dtype_torch = t.float64

@nb.jit(nopython=True)
def nb_balance_ss(F: np.ndarray, iters: int) -> np.ndarray:
    dim = F.shape[0]
    dtype = F.dtype
    d = np.ones((dim,), dtype=dtype)
    for k in range(iters):
        for i in range(dim):
            tmp = np.copy(F[:, i])
            tmp[i] = 0.
            c = np.linalg.norm(tmp, 2)
            tmp2 = np.copy(F[i, :])
            tmp2[i] = 0.

            r = np.linalg.norm(tmp2, 2)
            f = np.sqrt(r / c)
            d[i] *= f
            F[:, i] *= f
            F[i, :] /= f
    return d


def balance_ss(F: t.Tensor, L: t.Tensor, H: t.Tensor, q: t.Tensor, n_iter: int = 5) -> Tuple[t.Tensor, ...]:
    """Balance state-space model to have better numerical stability

    Parameters
    ----------
    F : t.Tensor
        Matrix
    L : t.Tensor
        Matrix
    H : t.Tensor
        Measurement matrix
    q : t.Tensor
        Spectral density
    P: t.Tensor, optional
        ...
    n_iter : int
        Iteration of balancing

    Returns
    -------
    F : t.Tensor
        ...
    L : t.Tensor
        ...
    H : t.Tensor
        ...
    q : t.Tensor
        ...

    References
    ----------
    https://arxiv.org/pdf/1401.5766.pdf
    """
    d = t.tensor(nb_balance_ss(F=copy.deepcopy(F.numpy()), iters=n_iter))
    d = t.reshape(d, (F.shape[0],))  # This is to make sure that the shape of d is known at compilation time.
    F = F * d[None, :] / d[:, None]
    L = L / d[:, None]
    H = H * d[None, :]

    tmp3 = t.max(t.abs(L))
    L = L / tmp3
    q = (tmp3 ** 2) * q

    tmp4 = t.max(t.abs(H))
    H = H / tmp4
    q = (tmp4 ** 2) * q
    return F, L, H, q


def solve_lyap_vec(F: t.Tensor, L: t.Tensor, Q: t.Tensor) -> t.Tensor:
    """Vectorized Lyapunov equation solver
    F P + P F' + L Q L' = 0
    Parameters
    ----------
    F : t.Tensor
        ...
    L : t.Tensor
        ...
    Q : t.Tensor
        ...

    Returns
    -------
    Pinf : t.Tensor
        Steady state covariance

    """
    dim = F.shape[0]

    op1 = F
    op2 = t.eye(dim, dtype=dtype_torch)

    F1 = t.kron(op2, op1)
    F2 = t.kron(op1, op2)

    F = F1 + F2
    Q = t.matmul(L, t.matmul(Q, L.t()))

    Pinf = t.reshape(t.linalg.solve(F, t.reshape(Q, (-1, 1))), (dim, dim))
    Pinf = -0.5 * (Pinf + Pinf.t())
    return Pinf
