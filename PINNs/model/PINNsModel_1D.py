import torch
import torch.nn as nn
from typing import List
from torchsummary import summary


class LinearNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int],
                 device=torch.device('cpu')):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._device = device

        self._activation_func = nn.Tanh()

        self._input_linear = nn.Linear(self._input_dim, hidden_layers[0])

        self._hidden_linears = nn.ModuleList()
        self._hidden_linears.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1])
                                     for i in range(len(hidden_layers)-1)])

        self._output_linear = nn.Linear(hidden_layers[-1], self._output_dim)

    def forward(self, x):
        x = self._input_linear(x)

        for i, linear in enumerate(self._hidden_linears):
            x = self._activation_func(linear(x))

        x = self._output_linear(x)

        return x

    def summary(self):
        summary(self, input_size=(self._input_dim,), device="cpu")


if __name__ == "__main__":

    net = LinearNetwork(2, 2, [5, 5])
    net.summary()
