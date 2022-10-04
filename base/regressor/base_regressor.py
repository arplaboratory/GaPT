import abc
import gpytorch
import torch
import numpy as np
import logging
import os
from tqdm import tqdm
from base.gpytgp.gppyt_model import GPYTModel
from base.data.data_loader import GPDataLoader
from base.gpytgp.gppyt_trainer import GPYTrainer
from torch import optim
from tools.misc import siso_binary_search, distance_3d
import time


class GPRegressor(metaclass=abc.ABCMeta):

    # TODO: ADD DESCRIPTION OF THE CLASS
    def __init__(self, kernel: gpytorch.kernels.Kernel,
                 likelihood: gpytorch.likelihoods.Likelihood, reg_name: str, input_dim: int = 1):

        self.__name__ = reg_name  # Name of the Regressor

        # Definition of the kernel and the likelihood used for the Gpytorch
        # Regressor
        self._gpt_kernel = kernel
        self._gpt_likelihood = likelihood

        # The implementation of the model is skipped for now: we include the possibility
        # load the pre-trained parameters.
        self._gpt_model = None  # Gpytorch kernel model implementation
        self._pssgp_cov = None  # LTI-SDE kernel model implementation
        self._kf = None  # Kalman's filter for regression and smoothing
        self._order = None  # Order of the RBF approximation for (P)SSGP
        self._balancing_iter = None  # Number of balancing steps for the resulting SDE to make it more stable

        # delta x between two samples
        self._covar_P = None
        self._is_ready = False

        # dimension of input
        self.input_dim = input_dim

    # TODO: MAKE IT PUBLIC NOT PRIVATE
    def _reset_filter(self):
        if self._is_ready:
            p_inf = self._pssgp_cov.get_sde()[0].numpy()
            self._covar_P = p_inf
        else:
            logging.warning('{}: unable to set P0 of the (Kalman Filter),'
                            'the model has not been initialized.'.format(self.__name__))

    def _reset_filter_miso(self):
        if self._is_ready:
            p_inf = self._pssgp_cov.get_sde()[0].numpy()
            self._covar_P = np.zeros((p_inf.shape[0] * self.input_dim, p_inf.shape[0] * self.input_dim))
            for i in range(self.input_dim):
                self._covar_P[p_inf.shape[0] * i:p_inf.shape[0] * (i + 1)
                , p_inf.shape[0] * i:p_inf.shape[0] * (i + 1)] = p_inf
        else:
            logging.warning('{}: unable to set P0 of the (Kalman Filter),'
                            'the model has not been initialized.'.format(self.__name__))


    def train_hyperparams(self, train_dataloader: GPDataLoader, gp_training_conf, verbose=False):
        logging.debug('Training of the {} initialized'.format(self.__name__))
        x_train, y_train, _ = next(iter(train_dataloader))
        self._gpt_model = GPYTModel(likelihood=self._gpt_likelihood, kernel=self._gpt_kernel,
                                    train_x=x_train, train_y=y_train)

        # GET THE TRAINING CONF
        training_conf = gp_training_conf

        # LOSS CREATION
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gpt_likelihood, self._gpt_model)

        # OPTIMIZER
        optimizer_clss = getattr(optim, training_conf['optimizer']["type"])
        optimizer = optimizer_clss(params=self._gpt_model.parameters(), **training_conf['optimizer']["args"])

        # TRAINER CREATION
        trainer = GPYTrainer(loss=mll, gp_model=self._gpt_model, likelihood=self._gpt_likelihood,
                             train_loader=train_dataloader, optimizer=optimizer, config=training_conf, verbose=verbose)
        # MODEL TRAINING
        trainer.train()

        # Put the model and the likelihood in the evaluation mode after training
        self._gpt_model.eval()
        self._gpt_likelihood.eval()

        # Call the function to create the LTI-SDE Model
        self._create_sde_model()

        # FINALIZING
        self._is_ready = True
        self._reset_filter()

    def predict_gpyt(self, x_input: torch.Tensor) -> (np.array, np.array, np.array):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._gpt_likelihood(self._gpt_model(x_input))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
        return mean, lower, upper

    def predict_kf_mean_cov(self, x_training: np.array, y_training: np.array, z_mean, z_std, x_test: np.array,
                            Qgain=1.0, Rgain=1.0):
        if self._is_ready:

            # Reset the filte
            self._reset_filter()

            # Init values
            len_data = x_test.shape[0]
            xt = np.zeros((self._order, 1))  # set the init value = 0
            y_predict = np.zeros(len_data)
            y_lower = np.zeros(len_data)
            y_upper = np.zeros(len_data)

            # get the starting index on the training dataset
            xt[0, :] = 0.0

            count = -1
            curr_x = 0
            for ix in tqdm(range(0, x_test.shape[0]), desc="KF predicting", colour='magenta', leave=False):
                in_data = x_test[ix]
                if x_training[0] > in_data:
                    xt[0, :] = y_training[0]
                if x_training[x_training.shape[0] - 1] < in_data:
                    count = x_training.shape[0] - 1
                else:
                    count = siso_binary_search(x_training, in_data)
                count -= 10
                if count < 0:
                    count = 1
                curr_x = x_training[count - 1]
                while x_training[count] < in_data and count < x_training.shape[0] - 1:
                    dx = x_training[count] - curr_x
                    x_new, P_new = self._kf.predict(xt, self._covar_P, dx, Qgain_factor=Qgain)
                    xt, self._covar_P = self._kf.update(y_training[count], x_new, P_new, Rgain_factor=Rgain)
                    curr_x = x_training[count]
                    count += 1
                dx = in_data - curr_x
                x_new, P_new = self._kf.predict(xt, self._covar_P, dx, Qgain_factor=Qgain)
                y_predict[ix] = z_mean + max(min(y_training), min(max(y_training), x_new[0])) * z_std
                y_lower[ix] = (z_mean + x_new[0] * z_std) - (P_new[0, 0] * z_std) / 2
                y_upper[ix] = (z_mean + x_new[0] * z_std) + (P_new[0, 0] * z_std) / 2

            return y_predict, y_lower, y_upper
        else:
            msg = '{}: The LTI-MODEL for the class must be created or loaded before using it!'.format(self.__name__)
            logging.error(msg)
            raise Exception(msg)

    def predict_kf_miso(self, x_training: np.array, y_training: np.array, z_mean, z_std, x_test: np.array,
                        reversed: bool, Qgain=1.0, Rgain=1.0):
        if self._is_ready:
            self._reset_filter_miso()
            len_data = x_test.shape[0]
            xt = np.zeros((self._order * self.input_dim, 1))
            xt[0, :] = 0.0
            y_predict = np.zeros(len_data)
            y_lower = np.zeros(len_data)
            y_upper = np.zeros(len_data)
            for ix in tqdm(range(1, x_test.shape[0]), desc="KF predicting", colour='magenta', leave=False):
                in_data = x_test[ix]
                count = 0
                cost = 999999999999
                for ix_tr in range(0, x_training.shape[0]):
                    distance = distance_3d(x_training[ix_tr], in_data)
                    if distance < cost:
                        cost = distance
                        count = ix_tr
                count -= 20
                if count < 0:
                    count = 1

                curr_x = x_training[count - 1]
                while distance_3d(x_training[count], in_data) > cost and count < x_training.shape[0] - 1:
                    dx = x_training[count] - curr_x
                    x_new, P_new = self._kf.predict(xt, self._covar_P, dx)
                    xt, self._covar_P = self._kf.update(y_training[count], x_new, P_new)
                    curr_x = x_training[count]
                    count += 1

                dx = in_data - curr_x
                # TODO: WE ARE USING THE -1 TRICK SINCE WE DISCOVERED THAT THE OUTPUT IS REVERSED
                #  WITH RESPECT TO THE GT
                x_new, P_new = self._kf.predict(xt, self._covar_P, dx)
                if not reversed:
                    y_predict[ix] = z_mean + max(min(y_training), min(max(y_training), x_new[0])) * z_std
                    y_lower[ix] = (z_mean + x_new[0] * z_std) - (P_new[0, 0] * z_std) / 2
                    y_upper[ix] = (z_mean + x_new[0] * z_std) + (P_new[0, 0] * z_std) / 2
                else :
                    y_predict[ix] = z_mean - max(min(y_training), min(max(y_training), x_new[0])) * z_std
                    y_lower[ix] = (z_mean - x_new[0] * z_std) - (P_new[0, 0] * z_std) / 2
                    y_upper[ix] = (z_mean - x_new[0] * z_std) + (P_new[0, 0] * z_std) / 2


            return y_predict, y_lower, y_upper
        else:
            msg = '{}: The LTI-MODEL for the class must be created or loaded before using it!'.format(self.__name__)
            logging.error(msg)
            raise Exception(msg)

    def get_gpt_models(self):
        if self._is_ready:
            return self._gpt_kernel, self._gpt_likelihood, self._gpt_model
        else:
            msg = '{}: Model must be loaded or trained before getting it.'.format(self.__name__)
            logging.error(msg)
            raise Exception

    def dump_model(self, saving_path: str):
        if self._is_ready:
            try:
                model_name = ''.join([self.__name__, '.pth'])  # Create the name for the model
                save_path = os.path.join(saving_path, model_name)  # Create the saving path
                torch.save(self._gpt_model.state_dict(), save_path)  # Save the model
            except Exception as e:
                logging.warning('An error occur while saving the model: {}, '
                                'the error is: \n\t> {}'.format(self.__name__, e))
        else:
            logging.warning('Unable to save the model {}: it is not instantiated.'.format(self.__name__))

    def load_model(self, model_path: str):
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)  # Load the state_dictionary
            self._gpt_model.load_state_dict(state_dict)  # Load the state_dictionary into the model
            self._gpt_model.eval()
            self._gpt_likelihood.eval()
            self._create_sde_model()  # Create the sde model
            self._is_ready = True  # Set the regressor ready to work
            self._reset_filter()  # Reset the previous stored values in the P matrix

        else:
            msg = 'The model file  {} not found '.format(model_path)
            logging.error(msg)
            raise FileNotFoundError(msg)

    def predict_kf_single(self, x_input: np.float):
        pass

    @abc.abstractmethod
    def _create_sde_model(self):
        raise NotImplementedError
