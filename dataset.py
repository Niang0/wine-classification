# dataset.py

import torch
from torch.utils.data import Dataset


class WineDataset(Dataset):

    def __init__(self, X, y):

        self.inputs = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        x = self.inputs[index]
        y = self.labels[index]

        return x, y