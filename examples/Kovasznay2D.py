import torch.optim
import torch
import numpy as np
from datetime import datetime
import argparse

from PINNs.model import LinearNetwork
from PINNs.solver import PINNsPDESolver
from PINNs.domainEquations import Burgers1D
from PINNs.boundaryConditions import DirichletBC
from PINNs.initialConditions import DirichletIC
from PINNs.plot import plotUtil


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

    nu = 1.0/40.0
    zeta = 1.0/(2.0*nu) - np.sqrt(1.0/(2.0*nu**2)+4.0*np.pi**2)

    # ic_x = np.linspace(-0.5, 1.0, num=40)
    # ic_y = np.linspace(-0.5, 1.5, num=40)

    # ic_input = np.stack((ic_x, ic_y), -1)
    # ic_u = np.reshape(1.0-np.exp(ic_x*zeta)*np.cos(2.0*np.pi*ic_y), (-1, 1))
    # ic_v = np.reshape(zeta/(2.0*np.pi)*np.exp(zeta*ic_x)
    #                   * np.sin(2.0*np.pi*ic_y), (-1, 1))
    # ic_p = np.reshape(0.5*(1-np.exp(2.0*zeta*ic_x)), (-1, 1))

    bc_x = np.concatenate((-0.5*np.ones(40), np.linspace(-0.5, 1.0, num=40),
                           1.0*np.ones(40), np.linspace(1.0, -0.5, num=40)))
    bc_y = np.concatenate((np.linspace(0.0, 1.0, num=40), 1.5*np.ones(40),
                           np.linspace(0.0, 1.0, num=40), -0.5*np.ones(40)))
    bc_input = np.stack((bc_x, bc_y), -1)

    bc_u = np.reshape(1.0-np.exp(bc_x*zeta)*np.cos(2.0*np.pi*bc_y), (-1, 1))
    bc_y = np.reshape(zeta/(2.0*np.pi)*np.exp(zeta*bc_x)
                      * np.sin(2.0*np.pi*bc_y), (-1, 1))
    bc_p = np.reshape(0.5*(1-np.exp(2.0*zeta*bc_x)), (-1, 1))

    domain_x = np.random.uniform(-0.5, 1.0, size=(500))
    domain_y = np.random.uniform(-0.5, 1.5, size=(500))
    domain_input = np.stack((domain_x, domain_y), -1)

    # # plotUtil.plot_scatter(domain_x, domain_t)

    # pre_input_meshgrid = np.meshgrid(np.linspace(-1.0, 1.0, num=300),
    #                                  np.linspace(0.0, 1.0, num=150))
    # pre_x = pre_input_meshgrid[0].reshape(-1, 1)
    # pre_t = pre_input_meshgrid[1].reshape(-1, 1)
    # pre_input = np.stack((pre_x, pre_t), -1)

    # if args.model:
    #     model = torch.load(args.model)
    # else:
    #     model = LinearNetwork(2, 3, [30, 30, 30, 30, 30])
    #     model.summary()

    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #     solver = PINNsPDESolver()
    #     solver.set_domain_equation(Burgers1D(domain_input))
    #     solver.add_boundary_condition(DirichletBC(bc_input, bc_value))
    #     solver.add_initial_condition(DirichletIC(ic_input, ic_value))

    #     print(f"Solving with device: {device}.")
    #     solver.solve(1, 20, model, optimizer, device)

    #     # datetime object containing current date and time
    #     now_time = datetime.now()
    #     now_time_string = now_time.strftime("%d%m%Y_%H-%M-%S")
    #     torch.save(model, now_time_string)

    # result = model(torch.from_numpy(pre_input).float().to(device)).to("cpu")

    # plotUtil.plot_imshow(pre_t.flatten(), pre_x.flatten(), result.detach(
    # ).numpy().reshape((pre_input_meshgrid[0].shape[0], pre_input_meshgrid[0].shape[1])).T)
