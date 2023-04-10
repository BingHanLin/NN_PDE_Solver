from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import numpy as np
from typing import Dict


class DomainEquation(ABC):
    # : Dict[str, int]
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


class Burgers1D(DomainEquation):
    def __init__(self,inputs: np.ndarray, input_index_map: Dict[str, int], output_index_map: Dict[str, int], name: str = "burger 1D"):
        super().__init__(inputs, input_index_map,output_index_map,name)

    # https://zhuanlan.zhihu.com/p/83172023
    # https://stackoverflow.com/questions/69148622/difference-between-autograd-grad-and-autograd-backward
    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:

        filtered_inputs = torch.index_select(
            self._inputs, 0, torch.from_numpy(filter_indice).int())

        filtered_inputs.requires_grad = True

        device_filtered_inputs = filtered_inputs.to(device)

        results = network(device_filtered_inputs)

        gradient1 = torch.autograd.grad(inputs=filtered_inputs, outputs=results, create_graph=True, retain_graph=True,
                                        grad_outputs=torch.ones_like(results))[self._output_index_map['u']]

        du_dx = gradient1[:, self._input_index_map['x']]
        du_dt = gradient1[:, self._input_index_map['t']]

        gradient2 = torch.autograd.grad(inputs=filtered_inputs, outputs=du_dx, create_graph=True, retain_graph=True,
                                        grad_outputs=torch.ones_like(du_dx))[self._output_index_map['u']]
        d2u_dx2 = gradient2[:, self._input_index_map['x']]

        nu = torch.tensor(0.01/np.pi)
        f = du_dt + \
            torch.squeeze(results.to(torch.device('cpu'))) * \
            du_dx - nu * d2u_dx2

        mse = nn.MSELoss(reduction='mean')

        loss = mse(f, torch.zeros(f.shape[0]))

        return loss
