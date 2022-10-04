import logging
import gpytorch
from base.pssgp.kalman import KalmanSISO, KalmanMISO
from base.pssgp.kernels import Periodic
from base.regressor import GPRegressor


class PerRbfModel(GPRegressor):
    def __init__(self, id_model: str, input_dim: int = 1):
        # Define the RBF model name, this is useful to handle multiple instances
        # of the same model.
        reg_name = ''.join(['PerRbF_', id_model])

        # Define the kernel and the likelihood for the GpyTorch training
        per_rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * \
                         gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # per_rbf_kernel = gpytorch.kernels.PeriodicKernel()
        rbf_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))
        super().__init__(kernel=per_rbf_kernel, likelihood=rbf_likelihood, reg_name=reg_name, input_dim=input_dim)

        # Set the approximation order and the balancing iters for the RBF-SDE kernel
        self._order = 2                 # Order of the RBF approximation for (P)SSGP
        self._balancing_iter = 2        # Number of balancing steps for the resulting SDE to make it more stable

    def _create_sde_model(self):
        # Extract the hyperparamenters.
        # Noise
        noise = self._gpt_likelihood.noise.detach().numpy().flatten()[0]

        outputscale_0 = self._gpt_model.covar_module.kernels._modules['0'].outputscale.detach().numpy().flatten()[0]
        lengthscale_0_unscaled = \
        self._gpt_model.covar_module.kernels._modules['0'].base_kernel.lengthscale.detach().numpy().flatten()[0]
        period_0_unscaled = \
        self._gpt_model.covar_module.kernels._modules['0'].base_kernel.period_length.detach().numpy().flatten()[0]
        per_period = lengthscale_0_unscaled * outputscale_0
        per_lengthscale = period_0_unscaled * outputscale_0

        outputscale_1 = self._gpt_model.covar_module.kernels._modules['1'].outputscale.detach().numpy().flatten()[0]
        lengthscale_1_unscaled = \
        self._gpt_model.covar_module.kernels._modules['1'].base_kernel.lengthscale.detach().numpy().flatten()[0]
        rbf_lengthscale = lengthscale_1_unscaled * outputscale_1
        # TODO: AGGIUNGERE LOGGING PER LOGGARE GLI HYPERPARAMETRIs

        self._pssgp_cov = Periodic(lengthscale_base=rbf_lengthscale,
                                   lengthscale_per=per_lengthscale,
                                   period_per=per_period,
                                   order=self._order)

        discrete_model = self._pssgp_cov.get_sde()
        self._order = 2 * (self._order + 1)
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
