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
    def __init__(self, inputs: np.ndarray, input_index_map: Dict[str, int], output_index_map: Dict[str, int], name: str = "burger 1D"):
        super().__init__(inputs, input_index_map, output_index_map, name)

    # https://zhuanlan.zhihu.com/p/83172023
    # https://stackoverflow.com/questions/69148622/difference-between-autograd-grad-and-autograd-backward

    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:

        filtered_inputs = torch.index_select(
            self._inputs, 0, torch.from_numpy(filter_indice).int())

        filtered_inputs.requires_grad = True

        device_filtered_inputs = filtered_inputs.to(device)

        results = network(device_filtered_inputs)

        x_index = self._input_index_map['x']
        t_index = self._input_index_map['t']

        u_index = self._output_index_map['u']

        grad_u = torch.autograd.grad(inputs=filtered_inputs, outputs=results[:, u_index:u_index+1], create_graph=True, retain_graph=True,
                                     grad_outputs=torch.ones_like(results[:, u_index:u_index+1]))[0]

        du_dx = grad_u[:, x_index]
        du_dt = grad_u[:, t_index]

        grad_du_dx = torch.autograd.grad(inputs=filtered_inputs, outputs=du_dx, create_graph=True, retain_graph=True,
                                         grad_outputs=torch.ones_like(du_dx))[0]
        d2u_dx2 = grad_du_dx[:, x_index]

        nu = torch.tensor(0.01/np.pi)

        f = du_dt + \
            torch.squeeze(results.to(torch.device('cpu'))) * \
            du_dx - nu * d2u_dx2

        mse = nn.MSELoss(reduction='mean')

        loss = mse(f, torch.zeros(f.shape[0]))

        return loss


class NavierStokes(DomainEquation):
    def __init__(self, inputs: np.ndarray, input_index_map: Dict[str, int], output_index_map: Dict[str, int], name: str = "navier stokes 2D"):
        super().__init__(inputs, input_index_map, output_index_map, name)

        self._reynoldNumber = 40.0

    def computeLoss(self, network: nn.Module, filter_indice: np.ndarray, device: torch.device) -> torch.Tensor:

        filtered_inputs = torch.index_select(
            self._inputs, 0, torch.from_numpy(filter_indice).int())

        filtered_inputs.requires_grad = True

        device_filtered_inputs = filtered_inputs.to(device)

        results = network(device_filtered_inputs)

        x_index = self._input_index_map['x']
        y_index = self._input_index_map['y']

        u_index = self._output_index_map['u']
        v_index = self._output_index_map['v']
        p_index = self._output_index_map['p']

        grad1_u = torch.autograd.grad(inputs=filtered_inputs, outputs=results[:, u_index:u_index+1], create_graph=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(results[:, u_index:u_index+1]))[0]

        grad1_v = torch.autograd.grad(inputs=filtered_inputs, outputs=results[:, v_index:v_index+1], create_graph=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(results[:, v_index:v_index+1]))[0]

        grad1_p = torch.autograd.grad(inputs=filtered_inputs, outputs=results[:, p_index:p_index+1], create_graph=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(results[:, p_index:p_index+1]))[0]

        du_dx = grad1_u[:, x_index]
        du_dy = grad1_u[:, y_index]

        dv_dx = grad1_v[:, x_index]
        dv_dy = grad1_v[:, y_index]

        dp_dx = grad1_p[:, x_index]
        dp_dy = grad1_p[:, y_index]

        grad_du_dx = torch.autograd.grad(inputs=filtered_inputs, outputs=du_dx, create_graph=True, retain_graph=True,
                                         grad_outputs=torch.ones_like(du_dx))[0]

        grad_du_dy = torch.autograd.grad(inputs=filtered_inputs, outputs=du_dy, create_graph=True, retain_graph=True,
                                         grad_outputs=torch.ones_like(du_dy))[0]

        grad_dv_dx = torch.autograd.grad(inputs=filtered_inputs, outputs=dv_dx, create_graph=True, retain_graph=True,
                                         grad_outputs=torch.ones_like(dv_dx))[0]

        grad_dv_dy = torch.autograd.grad(inputs=filtered_inputs, outputs=dv_dy, create_graph=True, retain_graph=True,
                                         grad_outputs=torch.ones_like(dv_dy))[0]

        du2_dx2 = grad_du_dx[:, x_index]
        du2_dy2 = grad_du_dy[:, y_index]

        dv2_dx2 = grad_dv_dx[:, x_index]
        dv2_dy2 = grad_dv_dy[:, y_index]

        results_u = results[:, u_index].to(torch.device('cpu'))
        results_v = results[:, v_index].to(torch.device('cpu'))
        results_p = results[:, p_index].to(torch.device('cpu'))

        loss_1 = results_u*du_dx + results_v * du_dy + \
            dp_dx - 1/self._reynoldNumber * (du2_dx2 + du2_dy2)

        loss_2 = results_u*dv_dx + results_v * dv_dy + \
            dp_dy - 1/self._reynoldNumber * (dv2_dx2 + dv2_dy2)

        loss_3 = du_dx + dv_dy

        f = torch.cat((loss_1, loss_2, loss_3))

        mse = nn.MSELoss(reduction='mean')

        loss = mse(f, torch.zeros(f.shape[0]))

        return loss
