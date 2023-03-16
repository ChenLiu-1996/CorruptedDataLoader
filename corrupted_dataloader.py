import numpy as np
import torch


class CorruptLabelDataLoader(torch.utils.data.DataLoader):
    '''
    This is a wrapper around a Pytorch DataLoader.
    To use, simply wrap it around an instantiated DataLoader instance,
    and use the resulting returned instance as if you are using
    a normal DataLoader instance.

    Example:
    ----------------
        train_loader = ...  # define `train_loader` as you normally would
        train_loader = CorruptLabelDataLoader(train_loader)

        for (x, y) in train_loader:
            ...
    ----------------

    Purpose of this wrapper:
        Randomly permute the labels such that there is an
        intentional mismatch between the images and labels.
    '''

    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 random_seed: int = 1) -> None:

        self.dataloader = dataloader

        np.random.seed(random_seed)
        if 'targets' in self.dataloader.dataset.__dir__():
            # The key `targets` is used in MNIST, CIFAR10, CIFAR100.
            self.dataloader.dataset.targets = np.random.permutation(
                self.dataloader.dataset.targets)
        elif 'labels' in self.dataloader.dataset.__dir__():
            # The key `labels` is used in STL10.
            self.dataloader.dataset.labels = np.random.permutation(
                self.dataloader.dataset.labels)

    def __getattr__(self, name):
        # This makes sure all methods and attributes of `dataloader`
        # is inherited by `self`, unless otherwise overwritten.
        return self.dataloader.__getattribute__(name)
