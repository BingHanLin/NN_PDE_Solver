from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import numpy as np


class DomainEquation(ABC):
    def __init__(self, inputs: np.ndarray, name: str):
        self._inputs = torch.from_numpy(inputs).float()
        self._name = name

    def inputs_number(self) -> int:
        return self._inputs.size(dim=0)

    @abstractmethod
    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:
        pass


class Burgers1D(DomainEquation):
    def __init__(self, inputs: np.ndarray, name: str = "burger 1D"):
        super().__init__(inputs, name)

    # https://zhuanlan.zhihu.com/p/83172023
    # https://stackoverflow.com/questions/69148622/difference-between-autograd-grad-and-autograd-backward
    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:

        filtered_inputs = torch.index_select(
            self._inputs, 0, torch.from_numpy(filter_indice).int())

        filtered_inputs.requires_grad = True

        device_filtered_inputs = filtered_inputs.to(device)

        results = network(device_filtered_inputs)

        gradient1 = torch.autograd.grad(inputs=filtered_inputs, outputs=results, create_graph=True, retain_graph=True,
                                        grad_outputs=torch.ones_like(results))[0]

        du_dx = gradient1[:, 0]
        du_dt = gradient1[:, 1]

        gradient2 = torch.autograd.grad(inputs=filtered_inputs, outputs=du_dx, create_graph=True, retain_graph=True,
                                        grad_outputs=torch.ones_like(du_dx))[0]
        d2u_dx2 = gradient2[:, 0]

        nu = torch.tensor(0.01/np.pi)
        f = du_dt + \
            torch.squeeze(results.to(torch.device('cpu'))) * \
            du_dx - nu * d2u_dx2

        mse = nn.MSELoss(reduction='mean')

        loss = mse(f, torch.zeros(f.shape[0]))

        return loss
