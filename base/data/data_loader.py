import logging
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from base.data.data_sampler import CustomSampler
from base.data.base_dataset import BaseDataset


class GPDataLoader(DataLoader):
    """
    This dataloader is intended for training the GpyTorch models in the regressor class.
    Given a dataset class (Base dataset) and the number of the training points the GPDataloader uses two samplers to
    obtain the indexes for the training and the validation. The logic is a little different from classic
    implementations of dataloaders in Torch. In particular, the validation split percentage is not present but
    replaced with the N. of K training points. There are two possible cases:
    - K < n. of total samples:
        - sample uniformly k samples from the entire dataset to create the training sampler,
        - remove the k samples from the indexes to create the validation sampler,
        - use the samplers to obtain the training and validation dataloader.
    - K == n. of total samples:
        - get all the indexes of the dataset,
        - use them to implement the training and validation dataloader.

    Parameters
    ----------
    dataset : dataset, instance of the Base dataset class.
    k_training_points: number of k points used to uniformly sample the indexes of the dataset.
    num_workers: num. of workers used for the training (same as for torch. non tested).
    collate_fn: collate function to apply, by default we use default_collate

    Returns
    -------
    The class implement a dataloader that can be passed as argument to the trainer.
    Calling the function "get_valid_dataloader" an instance of the validation dataloader (with the validation sampler)
    will be returned.
    """

    def __init__(self, dataset: BaseDataset, k_training_points: int, num_workers=1, collate_fn=default_collate):
        batch_size = dataset.__len__()  # In GP regressor the datasampler will do the job for us
        self.k_points = k_training_points
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.shuffle = False
        self.training_sampler, self.testing_sampler = self._get_samplers()

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.training_sampler, **self.init_kwargs)

    def _get_samplers(self):

        if self.k_points < self.n_samples:

            """ 
                Sample k values from the entire dataset. And use them as training values.
            """
            # Get the indexes of the values in the dataset.
            idx_full = np.arange(self.n_samples)

            # Uniform distribution for the samples picked for the training.
            idx_random_dist = np.random.uniform(size=self.k_points, low=idx_full[0], high=idx_full[-1])

            # Sort the indexes for the training.
            train_ix = np.sort(np.rint(idx_random_dist).astype(int))

            # Generate the sampler with the training values.
            training_sampler = CustomSampler(list(train_ix))

            # Get the indexes of the dataset except the ones used for the training.
            vaild_ix = np.delete(idx_full, train_ix)

            # Generate the sampler with the validation values.
            validation_sampler = CustomSampler(list(vaild_ix))

            return training_sampler, validation_sampler

        else:

            if self.k_points > self.n_samples:
                msg = 'Number of points selected for training ({}) are more than the ' \
                      'samples in the dataset ({}), setting them equals to the entire dataset'\
                    .format(self.k_points, self.n_samples)
                logging.debug(msg)
                self.k_points = self.n_samples

            # The training and the validation dataloader will contain all the values of the dataset.
            idx_full = np.arange(self.n_samples)
            training_sampler = CustomSampler(list(idx_full))
            validation_sampler = CustomSampler(list(idx_full))
            return training_sampler, validation_sampler

    def get_valid_dataloader(self):

        # Return the validation dataloader.
        return DataLoader(sampler=self.testing_sampler, **self.init_kwargs)
