import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


def plot_scatter(x: np.ndarray, y: np.ndarray):
    plt.scatter(x, y, marker="o")
    plt.colorbar()
    plt.show()


def plot_imshow(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    plt.imshow(z, cmap='coolwarm', extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar()
    plt.show()


def plot_contour(x_list: np.ndarray, y_list: np.ndarray, z_list: np.ndarray):

    xi, yi = np.linspace(x_list.min(), x_list.max(), 300), np.linspace(
        y_list.min(), y_list.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    zi = scipy.interpolate.griddata(
        (x_list, y_list), z_list, (xi, yi), method='linear')

    plt.imshow(zi, vmin=z_list.min(), vmax=z_list.max(), origin='lower',
               extent=[x_list.min(), x_list.max(), y_list.min(), y_list.max()])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pass
