from time import sleep
from tkinter.tix import Tree
from typing_extensions import Self
import torch
import torch.nn as nn
import numpy as np
from typing import List
from PINNs.domainEquations import DomainEquation
from PINNs.initialConditions import InitialCondition
from PINNs.boundaryConditions import BoundaryCondition


class PINNsPDESolver():
    def __init__(self):
        self._initial_conditions: List[InitialCondition] = []
        self._boundary_conditions: List[BoundaryCondition] = []
        self._domain_equation: DomainEquation

    def add_initial_condition(self, condition: InitialCondition):
        self._initial_conditions.append(condition)

    def add_boundary_condition(self, condition: BoundaryCondition):
        self._boundary_conditions.append(condition)

    def set_domain_equation(self, equation: DomainEquation):
        self._domain_equation = equation

    def solve(self, num_epochs: int, batch_number: int, model: nn.Module, optimizer: torch.optim.Optimizer,
              device: torch.device = torch.device('cpu')) -> None:

        model.to(device)
        model.train(True)

        verbose_epoch_number = 100

        for epoch in range(num_epochs):
            epoch_losses = torch.zeros(())

            batch_indices_list = self._batch_indices_list(batch_number)

            for one_batch_indices_list in batch_indices_list:
                total_losses = torch.zeros((), requires_grad=True)
                batch_index = 0

                if(one_batch_indices_list[batch_index].size > 0):
                    total_losses = total_losses + \
                        self._domain_equation.computeLoss(
                            model, one_batch_indices_list[batch_index], device)
                    batch_index += 1

                    if(epoch % verbose_epoch_number == 0):
                        print(f"domain loss, epoch {epoch}: {total_losses}")

                for condition in self._initial_conditions:
                    ic_losses = torch.zeros((), requires_grad=True)
                    if(one_batch_indices_list[batch_index].size > 0):
                        ic_losses = ic_losses + condition.computeLoss(
                            model, one_batch_indices_list[batch_index], device)
                        total_losses = total_losses + ic_losses
                        batch_index += 1

                        if(epoch % verbose_epoch_number == 0):
                            print(f"ic loss, epoch {epoch}: {ic_losses}")

                for condition in self._boundary_conditions:
                    bc_losses = torch.zeros((), requires_grad=True)
                    if(one_batch_indices_list[batch_index].size > 0):
                        bc_losses = bc_losses + condition.computeLoss(
                            model, one_batch_indices_list[batch_index], device)
                        total_losses = total_losses + bc_losses
                        batch_index += 1
                        if(epoch % verbose_epoch_number == 0):
                            print(f"bc loss, epoch {epoch}: {bc_losses}")

                optimizer.zero_grad()
                total_losses.backward()
                optimizer.step()

                epoch_losses = total_losses

                if(epoch % verbose_epoch_number == 0):
                    print(
                        f"----------------Total loss, epoch {epoch}: {epoch_losses}")

    def _batch_indices_list(self, batch_number: int) -> List[List[np.ndarray]]:

        inputs_number_list: List[int] = []

        inputs_number_list.append(self._domain_equation.inputs_number())

        for condition in self._initial_conditions:
            inputs_number_list.append(condition.inputs_number())

        for condition in self._boundary_conditions:
            inputs_number_list.append(condition.inputs_number())

        total_inputs_number = sum(inputs_number_list)

        batch_indices = np.linspace(
            0,  total_inputs_number-1, total_inputs_number)

        np.random.shuffle(batch_indices)

        batch_indices_list: List[List[np.ndarray]] = []
        for one_batch_indices in np.split(batch_indices, batch_number):
            one_list = []
            for one_inputs_number in inputs_number_list:
                one_list.append(
                    one_batch_indices[one_batch_indices < one_inputs_number])
                one_batch_indices = one_batch_indices[one_batch_indices >=
                                                      one_inputs_number]-one_inputs_number

            batch_indices_list.append(one_list)

        return batch_indices_list


if __name__ == "__main__":
    solver = PINNsPDESolver()
    pass
