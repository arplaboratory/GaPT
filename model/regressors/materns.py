import gpytorch
from base.gapt.kalman import KalmanSISO, KalmanMISO
from base.gapt.kernels import Matern12, Matern32, Matern52
from base.regressor import GPRegressor
import logging
r"""
Below the classes for the Matérn family. According to the equation of the kernel and the implementation in Gpytorch
thr e smoothness parameters \nu define the three kernels:
- Matérn_12 : \nu = 1/2
- Matérn_32 : \nu = 3/2
- Matérn_52 : \nu = 5/2
Note: even if it is possible to set smaller values, the results would be are less smooth.
"""


class Matern12Model(GPRegressor):
    def __init__(self, id_model: str, input_dim: int = 1):
        # Define the RBF model name, this is useful to handle multiple instances
        # of the same model.
        reg_name = ''.join(['Matern12_', id_model])

        # Definition of the kernel and the likelihood used for the Gpytorch
        # Regressor
        nu_matern = 1 / 2
        matern12_kernel = gpytorch.kernels.MaternKernel(nu=nu_matern)
        matern12_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(kernel=matern12_kernel, likelihood=matern12_likelihood, reg_name=reg_name, input_dim=input_dim)

        # Set the approximation order and the balancing iters for the RBF-SDE kernel
        self._order = 1             # Fixed parameter for matern kernel. It is just used to the Kalman filter function
        self._balancing_iter = 3    # Number of balancing steps for the resulting SDE to make it more stable

    def _create_sde_model(self):
        noise = self._gpt_likelihood.noise.detach().numpy().flatten()[0]
        lengthscale = self._gpt_model.covar_module.lengthscale.detach().numpy().flatten()[0]

        self._pssgp_cov = Matern12(variance=1, lengthscales=lengthscale,
                                   balancing_iter=self._balancing_iter)
        discrete_model = self._pssgp_cov.get_sde()

        P_inf = discrete_model[0].numpy()
        F = discrete_model[1].numpy()
        L = discrete_model[2].numpy()
        H = discrete_model[3].numpy()
        Q = discrete_model[4].numpy()

        if (self.input_dim == 1) and (isinstance(self.input_dim, int)):
            self._kf = KalmanSISO(F, L, H, Q, noise, order=self._order)
        elif (self.input_dim > 1) and (isinstance(self.input_dim, int)):
            self._kf = KalmanMISO(F, L, H, Q, noise, order=self._order, repeat=self.input_dim)
        else:
            msg = "The dimension input input_dim must be an integer > 0. Received value:{}".format(self.input_dim)
            logging.error(msg)
            raise ValueError(msg)

        self._is_ready = True
        self._reset_filter()


class Matern32Model(GPRegressor):
    def __init__(self, id_model: str, input_dim: int = 1):
        # Define the RBF model name, this is useful to handle multiple instances
        # of the same model.
        reg_name = ''.join(['Matern32_', id_model])

        # Definition of the kernel and the likelihood used for the Gpytorch
        # Regressor
        nu_matern = 3 / 2
        matern32_kernel = gpytorch.kernels.MaternKernel(nu=nu_matern)
        matern32_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(kernel=matern32_kernel, likelihood=matern32_likelihood, reg_name=reg_name, input_dim=input_dim)

        # Set the approximation order and the balancing iters for the RBF-SDE kernel
        self._order = 2             # Fixed parameter for matern kernel. It is just used to the Kalman filter function
        self._balancing_iter = 3    # Number of balancing steps for the resulting SDE to make it more stable

    def _create_sde_model(self):
        noise = self._gpt_likelihood.noise.detach().numpy().flatten()[0]
        lengthscale = self._gpt_model.covar_module.lengthscale.detach().numpy().flatten()[0]

        self._pssgp_cov = Matern32(variance=1, lengthscales=lengthscale,
                                   balancing_iter=self._balancing_iter)
        discrete_model = self._pssgp_cov.get_sde()

        P_inf = discrete_model[0].numpy()
        F = discrete_model[1].numpy()
        L = discrete_model[2].numpy()
        H = discrete_model[3].numpy()
        Q = discrete_model[4].numpy()

        if (self.input_dim == 1) and (isinstance(self.input_dim, int)):
            self._kf = KalmanSISO(F, L, H, Q, noise, order=self._order)
        elif (self.input_dim > 1) and (isinstance(self.input_dim, int)):
            self._kf = KalmanMISO(F, L, H, Q, noise, order=self._order, repeat=self.input_dim)
        else:
            msg = "The dimension input input_dim must be an integer > 0. Received value:{}".format(self.input_dim)
            logging.error(msg)
            raise ValueError(msg)
        self._is_ready = True
        self._reset_filter()


class Matern52Model(GPRegressor):
    def __init__(self, id_model: str, input_dim: int = 1):
        # Define the RBF model name, this is useful to handle multiple instances
        # of the same model.
        reg_name = ''.join(['Matern52_', id_model])

        # Definition of the kernel and the likelihood used for the Gpytorch
        # Regressor
        nu_matern = 5 / 2
        matern52_kernel = gpytorch.kernels.MaternKernel(nu=nu_matern)
        matern52_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(kernel=matern52_kernel, likelihood=matern52_likelihood, reg_name=reg_name, input_dim=input_dim)

        # Set the approximation order and the balancing iters for the RBF-SDE kernel
        self._order = 3             # Fixed parameter for matern kernel. It is just used to the Kalman filter function
        self._balancing_iter = 0    # Number of balancing steps for the resulting SDE to make it more stable

    def _create_sde_model(self):
        noise = self._gpt_likelihood.noise.detach().numpy().flatten()[0]
        lengthscale = self._gpt_model.covar_module.lengthscale.detach().numpy().flatten()[0]

        self._pssgp_cov = Matern52(variance=1.0, lengthscales=lengthscale,
                                   balancing_iter=self._balancing_iter)
        discrete_model = self._pssgp_cov.get_sde()

        P_inf = discrete_model[0].numpy()
        F = discrete_model[1].numpy()
        L = discrete_model[2].numpy()
        H = discrete_model[3].numpy()
        Q = discrete_model[4].numpy()

        if (self.input_dim == 1) and (isinstance(self.input_dim, int)):
            self._kf = KalmanSISO(F, L, H, Q, noise, order=self._order)
        elif (self.input_dim > 1) and (isinstance(self.input_dim, int)):
            self._kf = KalmanMISO(F, L, H, Q, noise, order=self._order, repeat=self.input_dim)
        else:
            msg = "The dimension input input_dim must be an integer > 0. Received value:{}".format(self.input_dim)
            logging.error(msg)
            raise ValueError(msg)
        self._is_ready = True
        self._reset_filter()
