from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import numpy as np
from typing import Dict


class InitialCondition(ABC):
    def __init__(self, inputs: np.ndarray, input_index_map: Dict[str, int], output_index_map: Dict[str, int], name: str):
        self._inputs = torch.from_numpy(inputs).float()
        self._input_index_map = input_index_map
        self._output_index_map = output_index_map
        self._name = name

    def inputs_number(self) -> int:
        return self._inputs.size(dim=0)

    @abstractmethod
    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:
        pass


class DirichletIC(InitialCondition):
    def __init__(self, inputs: np.ndarray, values: np.ndarray, input_index_map: Dict[str, int], output_index_map: Dict[str, int], name: str = "dirichlet"):
        super().__init__(inputs, input_index_map, output_index_map, name)
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
