"""
the model used in the kalman Filter:
x[k+1] = Ax[k] + Gu[k] + w[k]
z[k] = Hx[k] + v[n]
where
x is the state vector, y the measurement, w the process noise and v the measurement noise.
The kalman filter assumes that w and v are zero-mean, independent random variables with known
variances:
- E[ww'] = Q
- E[vv'] = R
Note: symbol " ' " in this description means transposed.
A -> State matrix
G -> Input Matrix
H -> Observation Matrix
"""
import numpy as np
import torch as t


def expm_t(mat):
    mat_tensor = t.tensor(mat)
    res = t.linalg.matrix_exp(mat_tensor)
    return res.numpy()


def expm_f(A):
    A2 = np.matmul(A, A)
    res = (np.eye(A.shape[0])) + A + 0.5 * A2 + np.matmul(A2, A) / 6
    if np.trace(res) > 50:
        return np.eye(A.shape[0])
    return res


class KalmanSISO:

    def __init__(self, F, L, H, Q, R, **kwargs):
        self.F = F
        self.Q = Q
        self.R = R
        self.L = L
        self.H = H
        self.order = kwargs.pop('order')

    def calc_qs(self, dt, Qgain_factor):
        """
        The method calc_Qs changes the values of the matrix Q according to dt. The dimension
        must agree with the one of F: order value is mandatory.
        Disclaimer:this is an arbitrary method to evaluate/update related
        to a specific task -> GSSM and kalman Filter.
        """
        Qs = np.zeros((self.order * 2, self.order * 2))
        Qs[:self.order, :self.order] = self.F
        Qs[self.order:self.order * 2, self.order:self.order * 2] = -self.F.T
        Qs[:self.order, self.order:self.order * 2] = np.matmul(self.L, self.L.T) * self.Q
        Qs = expm_f(Qs * np.linalg.norm(dt))
        Qs = np.matmul(Qs[:self.order, self.order:self.order * 2], expm_f(self.F * np.linalg.norm(dt)).T)
        return Qs * Qgain_factor

    def predict(self, x_k, P, delta_t, Qgain_factor=1.0):
        """
        F = exp(A*dt)`
        x_pred = F*x
        P_pred = F*P*F' + Q
        """
        F = expm_f(self.F * delta_t)
        x_pred = np.matmul(F, x_k)
        P_pred = np.matmul(np.matmul(F, P), np.transpose(F)) + self.calc_qs(delta_t, Qgain_factor=Qgain_factor)
        return x_pred, P_pred

    def update(self, z, x_pred, P_pred, Rgain_factor=1.0):
        n, S = self._innovation(x_pred, z, P_pred, Rgain_factor)
        return self._innovation_update(x_pred, P_pred, n, S)

    def _innovation(self, x_pred, z, P_pred, Rgain_factor):
        """
        nu = z - H*x_pred       (Innovation) -> It represents the error between the estimate and the measurement
        S = R + H*P_pred*H'     (Innovation Covariance) -> Needed to evaluate the new K
        """
        nu = z - np.matmul(self.H, x_pred)
        S = np.asmatrix((self.R * Rgain_factor) + np.matmul(np.matmul(self.H, P_pred), np.transpose(self.H)))
        return nu, S

    def _innovation_update(self, x_pred, P_pred, nu, S):
        """
        K = P_pred*H' * inv(S)  (kalman Gain) -> equivalent to K = (P_k * H) / (H*P*H' + R )
        x_new = x_pred + K*Nu   (New State)
        P_new = P_pred - K*S*K' (New Covariance)
        """
        K = np.matmul(np.matmul(P_pred, np.transpose(self.H)), np.linalg.inv(S))
        x_new = x_pred + K * nu
        P_new = P_pred - np.matmul(np.matmul(K, S), np.transpose(K))
        return x_new, P_new


class KalmanMISO:
    def __init__(self, F, L, H, Q, R, order: int, repeat: int, **kwargs):
        self.F = F
        self.Q = Q
        self.R = R
        self.L = L
        self.repeat = repeat
        self.order = order
        new_H = np.zeros((1, H.shape[1] * self.repeat))
        # new_R = np.zeros((R.shape[0]*self.repeat,R.shape[1]*self.repeat))
        avg_weight = [0.6, 0.05, 0.05, 0.05, 0.05]
        for i in range(self.repeat):
            new_H[0, H.shape[1] * i:H.shape[1] * (i + 1)] = H * avg_weight[i]
            # new_R[R.shape[0]*i:R.shape[0]*(i+1),R.shape[1]*i:R.shape[1]*(i+1)] = R
        self.H = new_H

    def calc_qs(self, dt):
        Qs = np.zeros((self.order * 2, self.order * 2))
        Qs[:self.order, :self.order] = self.F
        Qs[self.order:self.order * 2, self.order:self.order * 2] = -self.F.T
        Qs[:self.order, self.order:self.order * 2] = np.matmul(self.L, self.L.T) * self.Q
        Q_total = np.zeros((self.order * self.repeat, self.order * self.repeat))
        for i in range(self.repeat):
            if i == 0:
                c = 0.01
            else:
                c = 0.01
            Qs2 = expm_f(Qs * np.linalg.norm(dt[i]) * c)
            Q_total[self.order * i:self.order * (i + 1), self.order * i:self.order * (i + 1)] = \
                np.matmul(Qs2[:self.order, self.order:self.order * 2], expm_f(self.F * np.linalg.norm(dt[i])).T)
        return Q_total * 0.7

    def predict(self, x_k, P, delta_t):
        """
        F = exp(A*dt)`
        x_pred = F*x
        P_pred = F*P*F' + Q
        """
        F_total = np.zeros((self.F.shape[0] * self.repeat, self.F.shape[1] * self.repeat))
        for i in range(self.repeat):
            F = expm_f(self.F * np.linalg.norm(delta_t[i]))
            F_total[F.shape[0] * i:F.shape[0] * (i + 1), F.shape[1] * i:F.shape[1] * (i + 1)] = F
        x_pred = np.matmul(F_total, x_k)
        P_pred = np.matmul(np.matmul(F_total, P), np.transpose(F_total)) + self.calc_qs(delta_t)
        return x_pred, P_pred

    def _innovation(self, x_pred, z, P_pred, H_custom=None):

        """
        nu = z - H*x_pred       (Innovation) -> It represents the error between the estimate and the measurement
        S = R + H*P_pred*H'     (Innovation Covariance) -> Needed to evaluate the new K
        """

        if H_custom is None:
            nu = z - np.matmul(self.H, x_pred)
            S = np.asmatrix(self.R + np.matmul(np.matmul(self.H, P_pred), np.transpose(self.H)))
        else:
            nu = z - np.matmul(H_custom, x_pred)
            S = np.asmatrix(self.R + np.matmul(np.matmul(H_custom, P_pred), np.transpose(H_custom)))
        return nu, S

    def _innovation_update(self, x_pred, P_pred, nu, S, H_custom=None):
        """
        K = P_pred*H' * inv(S)  (kalman Gain) -> equivalent to K = (P_k * H) / (H*P*H' + R )
        x_new = x_pred + K*Nu   (New State)
        P_new = P_pred - K*S*K' (New Covariance)
        """

        if H_custom is None:
            K = np.matmul(np.matmul(P_pred, np.transpose(self.H)), np.linalg.inv(S))
            x_new = x_pred + K * nu
            P_new = P_pred - np.matmul(np.matmul(K, S), np.transpose(K))

        else:
            K = np.matmul(np.matmul(P_pred, np.transpose(H_custom)), np.linalg.inv(S))
            x_new = x_pred + K * nu
            P_new = P_pred - np.matmul(np.matmul(K, S), np.transpose(K))
        return x_new, P_new

    def update(self, z, x_pred, P_pred, H_custom=None):
        n, S = self._innovation(x_pred, z, P_pred, H_custom)
        return self._innovation_update(x_pred, P_pred, n, S, H_custom)
