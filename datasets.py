import torch

from sklearn import datasets
from torch.utils.data import TensorDataset


def get_dataset(name, split):
    """
    Tip: only use this function for getting datasets for modularity
    """

    # Tip: use if/elif/else to switch between datasets, pytorch is great
    if name == "digits":
        # get digits dataset
        X, y = datasets.load_digits(n_class=2, return_X_y=True)

        # define train/val splits
        n = int(len(X)* 0.8)
        if split == 'train':
            X, y = X[:n], y[:n]
        elif split == 'val':
            X, y = X[n:], y[n:]

        # acquire pytorch dataset
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y))

    else:
        # Tip: include this to avoid silent bugs
        raise ValueError(f'{name} not found')
    
    return dataset