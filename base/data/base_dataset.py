from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import numpy as np
from tools.dsp_utils import SigFilter, Tools
import torch


class BaseDataset(Dataset, ABC):
    """
     Base class for all the Dataset. The format is compliant with the dataloader schemes commonly used in
     Pytorch.
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
            *args, *kwargs: optional arguments that must be implemented on the real class.
     Note: In this work we are referring to time-series data, so we included the timestamp as returned value.
     """

    def __init__(self, dataset_label: str, *args, **kwargs):
        self.X = None
        self.Y = None
        self.timestamp = None
        self._init_dataset()
        self.__name__ = dataset_label
        super().__init__()

    def __len__(self):
        return len(self.X)

    # The _init_dataset method is used to adjust the data according to the application.
    @abstractmethod
    def _init_dataset(self, **kwargs):
        """
            This abstract method should set the three float tensors
            - X (samples)
            - Y (labels)
            - timestamp (time values)
        """
        raise NotImplementedError

    def __getitem__(self, index):
        sample = self.X[index]
        label = self.Y[index]
        ts = self.timestamp[index]
        return sample, label, ts

    def select_subset(self, start_index, end_index):
        """
            This method is used to select a subset of the orginal dataset.
            Useful if you want to train on a particular subset.
        """
        # Restore the original format of the dataset
        if isinstance(start_index, type(None)) or isinstance(end_index, type(None)):
            self._init_dataset()
        else:
            self._init_dataset()
            if end_index > len(self) or end_index == -1:
                end_index = len(self)
            if start_index < 0:
                start_index = 0
            self.X = self.X[start_index: end_index]
            self.Y = self.Y[start_index: end_index]
            self.timestamp = self.timestamp[start_index: end_index]










