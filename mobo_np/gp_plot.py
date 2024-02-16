import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import animation, cm


## Plots the results ##
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], save_path=None):
    """
    Plots the results of a Gaussian Process regression.

    Args:
        mu: Array representing the mean values of the Gaussian Process predictions.
        cov: Covariance matrix of the Gaussian Process predictions.
        X: Input locations for plotting.
        X_train: Training input locations (if available).
        Y_train: Training target values (if available).
        samples: List of sample predictions to be plotted.
        save_path: Path to save the plot as an image file (e.g., 'plot.png').

    Returns:
        None
    """
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.figure()
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)