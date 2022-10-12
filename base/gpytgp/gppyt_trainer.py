"""
PYTORCH TRAINERS IMPLEMENTATIONS
"""
import torch
import logging
from tqdm import tqdm


class GPYTrainer:
    """
    Trainer class for finding the hyperparameters
     of Gaussian processes implemented in GPytorch
    """

    def __init__(self, loss, gp_model, likelihood, train_loader, optimizer, config, verbose=False):
        self.loss = loss
        self.model = gp_model
        self.likelihood = likelihood
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.config = config
        self._verbose = verbose

    def _find_optimal_params(self):
        # Find optimal model hyper-parameters
        self.model.train()
        self.likelihood.train()

    def train(self):
        """
        Full training logic
        """
        self._find_optimal_params()

        epochs = self.config['epochs']
        for epoch in tqdm(range(0, epochs), desc="GPT training", leave=False):
            for data, label, _ in self.train_loader:
                # TODO ACTUAL PYTORCH IS NOT COMPATIBLE WIT THE VERSION OF CUDA. PYTORCH MUST BE UPDATED
                # if torch.cuda.is_available():
                #     data, label = data.to("cuda"), label.to("cuda")
                # else:
                #     data, label = data.to("cpu"), label.to("cpu")

                # TRAINING STEPS

                # Zero the gradient
                self.optimizer.zero_grad()

                # Output from model
                targets = self.model(data)

                # Calc loss and backprop gradients
                loss = -self.loss(targets, label)

                # Backprop the gradients
                loss.backward()

                # Display info
                if self._verbose:
                    logging.info("Iter {}/{} - Loss: {:.3f}  noise: {:.3f}".format(epoch + 1, epochs, loss.item(),
                                                                                   self.model.likelihood.noise.item()))

                    print("Iter {}/{} - Loss: {:.3f}  noise: {:.3f}".format(epoch + 1, epochs, loss.item(),
                                                                            self.model.likelihood.noise.item()))

                self.optimizer.step()

