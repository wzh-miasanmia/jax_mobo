import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.scipy.linalg import cholesky, solve_triangular

class GaussianProcessRegressor:
    def __init__(self, l=1.0, sigma_f=1.0, sigma_y=1e-8):
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        self.X_train = None
        self.Y_train = None
        self.L = None
        self.alpha = None

    def kernel(self, X1, X2):
        sqdist = jnp.sum(X1**2, axis=1).reshape((-1, 1)) + jnp.sum(X2**2, axis=1) - 2 * jnp.dot(X1, X2.T)
        return self.sigma_f**2 * jnp.exp(-0.5 / self.l**2 * sqdist)

    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y
        K = self.kernel(X, X) + self.sigma_y**2 * jnp.eye(len(X))
        self.L = cholesky(K, lower=True)
        
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y, lower=True), lower=False)

    def log_marginal_likelihood(self, theta):
        return jnp.sum(jnp.log(jnp.diag(self.L))) + 0.5 * jnp.dot(self.Y_train, self.alpha) + 0.5 * len(self.X_train) * jnp.log(2 * jnp.pi)

    def predict(self, X, return_std=False, return_cov=False):
        K_s = self.kernel(self.X_train, X)
        K_ss = self.kernel(X, X) + 1e-8 * jnp.eye(len(X))
        mu_s = K_s.T.dot(self.alpha)
        v = solve_triangular(self.L, K_s, lower=True)
        cov_s = K_ss - v.T.dot(v)
        if return_std and return_cov:
            return mu_s, jnp.sqrt(jnp.diag(cov_s)), cov_s
        elif return_std:
            return mu_s, jnp.sqrt(jnp.diag(cov_s))
        elif return_cov:
            return mu_s, cov_s
        else:
            return mu_s
