import argparse
from datetime import datetime
import numpy as np
import torch
import torch.optim

from PINNs.plot import plotUtil
from PINNs.initialConditions import DirichletIC
from PINNs.boundaryConditions import DirichletBC
from PINNs.domainEquations import NavierStokes
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

    # boundary dataset
    bc_x = np.concatenate((np.zeros(100),
                          np.linspace(0.0, 1.0, num=100),
                          np.ones(100),
                          np.linspace(0.0, 1.0, num=100)))
    bc_y = np.concatenate((np.linspace(0.0, 1.0, num=100),
                           np.ones(100),
                           np.linspace(0.0, 1.0, num=100),
                           np.zeros(100)))
    bc_input = np.stack((bc_x, bc_y), -1)

    bc_u = np.concatenate((np.zeros(100),
                           np.ones(100),
                           np.zeros(100),
                           np.zeros(100)))
    bc_v = np.concatenate((np.zeros(100),
                           np.zeros(100),
                           np.zeros(100),
                           np.zeros(100)))
    bc_value = np.stack((bc_u, bc_v), -1)

    # domain dataset
    domain_x = np.random.uniform(0.0, 1.0, size=(1500))
    domain_y = np.random.uniform(0.0, 1.0, size=(1500))
    domain_input = np.stack((domain_x, domain_y), -1)

    input_index_map = {'x': 0, 'y': 1}
    output_index_map = {'u': 0, 'v': 1, 'p': 2}

    if args.model:
        model = torch.load(args.model)
    else:
        model = LinearNetwork(2, 3, [80, 80, 80, 80, 80, 80])

        model.summary()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        solver = PINNsPDESolver()
        solver.set_domain_equation(
            NavierStokes(domain_input, input_index_map, output_index_map))

        solver.add_boundary_condition(DirichletBC(
            bc_input, bc_value, input_index_map, output_index_map, ['u', 'v']))

        print(f"Solving with device: {device}.")
        solver.solve(1000, 5, model, optimizer, device)

        # datetime object containing current date and time
        now_time = datetime.now()
        now_time_string = now_time.strftime("%d%m%Y_%H-%M-%S")
        torch.save(model, now_time_string)

    # plot result
    pre_input_meshgrid = np.meshgrid(np.linspace(0.0, 1.0, num=200),
                                     np.linspace(0.0, 1.0, num=200))
    pre_x = pre_input_meshgrid[0].reshape(-1)
    pre_y = pre_input_meshgrid[1].reshape(-1)
    pre_input = np.stack((pre_x, pre_y), -1)

    result = model(torch.from_numpy(pre_input).float().to(device)).to("cpu")

    u_index = output_index_map['u']
    v_index = output_index_map['v']
    p_index = output_index_map['p']

    plotUtil.plot_contour(pre_x.flatten(), pre_y.flatten(), result[:, u_index:u_index+1].detach(
    ).numpy().flatten())

    plotUtil.plot_contour(pre_x.flatten(), pre_y.flatten(), result[:, v_index:v_index+1].detach(
    ).numpy().flatten())
