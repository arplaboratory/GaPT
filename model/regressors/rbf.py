import gpytorch
from base.gapt.kalman import KalmanSISO, KalmanMISO
from base.gapt.kernels import RBF
from base.regressor import GPRegressor
import logging


class RBFModel(GPRegressor):
    def __init__(self, id_model: str, input_dim: int = 1):

        # Define the RBF model name, this is useful to handle multiple instances
        # of the same model.
        reg_name = ''.join(['Rbf_', id_model])

        # Definition of the kernel and the likelihood used for the Gpytorch
        # Regressor
        rbf_kernel = gpytorch.kernels.RBFKernel()
        rbf_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(kernel=rbf_kernel, likelihood=rbf_likelihood, reg_name=reg_name, input_dim=input_dim)

        # Set the approximation order and the balancing iters for the RBF-SDE kernel
        self._order = 6  # Order of the RBF approximation for (P)SSGP
        self._balancing_iter = 5  # Number of balancing steps for the resulting SDE to make it more stable

    def _create_sde_model(self):
        noise = self._gpt_likelihood.noise.detach().numpy().flatten()[0]
        lengthscale = self._gpt_model.covar_module.lengthscale.detach().numpy().flatten()[0]

        self._pssgp_cov = RBF(variance=1, lengthscales=lengthscale,
                              order=self._order, balancing_iter=self._balancing_iter)

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
