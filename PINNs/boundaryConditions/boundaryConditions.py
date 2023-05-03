from abc import ABC, abstractmethod
from unittest import result
import torch.nn as nn
import torch
import numpy as np
from typing import Dict, List


class BoundaryCondition(ABC):
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


class DirichletBC(BoundaryCondition):
    def __init__(self, inputs: np.ndarray, values: np.ndarray, input_index_map: Dict[str, int], output_index_map: Dict[str, int], values_to_compare: List[str], name: str = "dirichlet"):
        super().__init__(inputs, input_index_map, output_index_map, name)
        self._values = torch.from_numpy(values).float()
        self._values_to_compare = values_to_compare

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

        total_loss = 0.0

        for one_value in self._values_to_compare:
            if one_value in self._output_index_map:
                one_index = self._output_index_map[one_value]

                if results[0].shape[0] <= one_index or filtered_values[0].shape[0] <= one_index:
                    raise Exception(
                        "one_index not fit to the resuls or filtered_values.")
                else:
                    total_loss += mse(
                        results[:, one_index:one_index+1], filtered_values[:, one_index:one_index+1])
            else:
                raise Exception(
                    "values_to_compare not found in output_index_map.")

        return total_loss
