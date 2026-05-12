# model.py

import torch.nn as nn


class WineClassifier(nn.Module):

    def __init__(self):

        super(WineClassifier, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(13, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)

        )

    def forward(self, x):

        return self.network(x)