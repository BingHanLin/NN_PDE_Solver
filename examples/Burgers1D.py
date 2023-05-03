import argparse
from datetime import datetime
import numpy as np
import torch
import torch.optim

from PINNs.plot import plotUtil
from PINNs.initialConditions import DirichletIC
from PINNs.boundaryConditions import DirichletBC
from PINNs.domainEquations import Burgers1D
from PINNs.solver import PINNsPDESolver
from PINNs.model import LinearNetwork


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Argument Parser.')
    parser.add_argument('-m', '--model')
    args = parser.parse_args()

    ic_x = np.linspace(-1.0, 1.0, num=40)
    ic_t = np.zeros_like(ic_x)
    ic_input = np.stack((ic_x, ic_t), -1)
    ic_value = np.reshape(-np.sin(np.pi*ic_x), (-1, 1))

    bc_x = np.append(np.ones(40),
                     -np.ones(40))
    bc_t = np.append(np.linspace(0.0, 1.0, num=40),
                     np.linspace(0.0, 1.0, num=40))
    bc_input = np.stack((bc_x, bc_t), -1)
    bc_value = np.reshape(np.zeros(80), (-1, 1))

    domain_x = np.random.uniform(-1.0, 1.0, size=(500))
    domain_t = np.random.uniform(0.0, 1.0, size=(500))
    domain_input = np.stack((domain_x, domain_t), -1)

    input_index_map = {'x': 0, 't': 1}
    output_index_map = {'u': 0}

    if args.model:
        model = torch.load(args.model)
    else:
        model = LinearNetwork(2, 1, [30, 30, 30, 30, 30])
        model.summary()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

        solver = PINNsPDESolver()
        solver.set_domain_equation(
            Burgers1D(domain_input, input_index_map, output_index_map))
        solver.add_boundary_condition(DirichletBC(
            bc_input, bc_value, input_index_map, output_index_map, ['u']))
        solver.add_initial_condition(DirichletIC(
            ic_input, ic_value, input_index_map, output_index_map))

        print(f"Solving with device: {device}.")
        solver.solve(500, 20, model, optimizer, device)

        # datetime object containing current date and time
        now_time = datetime.now()
        now_time_string = now_time.strftime("%d%m%Y_%H-%M-%S")
        torch.save(model, now_time_string)

    # plot result
    pre_input_meshgrid = np.meshgrid(np.linspace(-1.0, 1.0, num=300),
                                     np.linspace(0.0, 1.0, num=150))
    pre_x = pre_input_meshgrid[0].reshape(-1, 1)
    pre_t = pre_input_meshgrid[1].reshape(-1, 1)
    pre_input = np.stack((pre_x, pre_t), -1)

    result = model(torch.from_numpy(pre_input).float().to(device)).to("cpu")

    plotUtil.plot_imshow(pre_t.flatten(), pre_x.flatten(), result.detach(
    ).numpy().reshape((pre_input_meshgrid[0].shape[0], pre_input_meshgrid[0].shape[1])).T)
