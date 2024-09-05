import copy

import torch
import tqdm

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size = (30, 90), output_size = (90, 1)):
        super(NeuralNetwork, self).__init__()
        self.hidden_size = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu = nn.ReLU()
        self.output_size = nn.Linear(output_size[0], output_size[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden_size(x))
        x = self.sigmoid(self.output_size(x))
        return x