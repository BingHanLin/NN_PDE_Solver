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


def plot_contour(x: np.ndarray, y: np.ndarray, z: np.ndarray):

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)

    plt.contourf(xi, yi, zi, 20)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pass
