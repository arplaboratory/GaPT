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


##############################################################################################################
import torch
from abc import abstractmethod
from numpy import inf


class BaseTrainer:
    """
    Base class for The trainers used for Gpytorch Training procedure
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        # self.writer = TensorboardWriter(configs.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logger information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'configs': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['configs']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in configs file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['configs']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in configs file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
