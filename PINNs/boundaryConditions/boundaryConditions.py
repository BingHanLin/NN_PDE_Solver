from abc import ABC, abstractmethod
from unittest import result
import torch.nn as nn
import torch
import numpy as np


class BoundaryCondition(ABC):
    def __init__(self, inputs: np.ndarray, name: str):
        self._inputs = torch.from_numpy(inputs).float()
        self._name = name

    def inputs_number(self) -> int:
        return self._inputs.size(dim=0)

    @abstractmethod
    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:
        pass


class DirichletBC(BoundaryCondition):
    def __init__(self, inputs: np.ndarray, values: np.ndarray, name: str = "dirichlet"):
        super().__init__(inputs, name)
        self._values = torch.from_numpy(values).float()

    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:
        filtered_inputs = torch.index_select(
            self._inputs, 0, torch.from_numpy(filter_indice).int())

        filtered_values = torch.index_select(
            self._values, 0, torch.from_numpy(filter_indice).int())

        filtered_inputs.requires_grad = True

        device_filtered_inputs = filtered_inputs.to(device)

        results = network(device_filtered_inputs)
        results = results.to(torch.device('cpu'))

        mse = nn.MSELoss(reduction='mean')
        loss = mse(results, filtered_values)
        return loss
