import gpytorch
import torch

class GPYTModel(gpytorch.models.ExactGP):
    """
    Base ExactGPModel:
        - train_x: N x D training tensor data features
        - train_y N x M  training tensor data labels
        - likelihood: gpytorch.likelihoods
        - kernels: gpytorch.kernels
    """
    def __init__(self, train_x: torch.tensor, train_y: torch.tensor,
                 likelihood: gpytorch.likelihoods, kernel: gpytorch.kernels):
        super(GPYTModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




