import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

class GaussianProcessRegressor:
    def __init__(self, l=1.0, sigma_f=1.0, sigma_y=1e-8):
        # data samples
        self.X_train = None
        self.Y_train = None

        # hyperparameters
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        
        # cov relevant
        self.K = None
        self.L = None
        self.alpha = None


    def kernel(self, X1, X2, l=1.0, sigma_f=1.0):
        sqdist = jnp.sum(X1**2, axis=1).reshape((-1, 1)) + jnp.sum(X2**2, axis=1) - 2 * jnp.dot(X1, X2.T)
        return sigma_f**2 * jnp.exp(-0.5 / l**2 * sqdist)

    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_y**2 * jnp.eye(len(self.X_train))
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, self.Y_train, lower=True), lower=False)


    def optim_jax(self, num_steps, lr, method='SGD'):
        theta = jnp.array([self.l, self.sigma_f])

        def mll(theta):
            K = self.kernel(self.X_train, self.X_train, l=theta[0], sigma_f=theta[1]) + \
                self.sigma_y**2 * jnp.eye(len(self.X_train))
            L = cholesky(K)
            S1 = solve_triangular(L, self.Y_train, lower=True)
            S2 = solve_triangular(L.T, S1, lower=False)
            return jnp.sum(jnp.log(jnp.diag(L))) + \
                    0.5 * jnp.dot(self.Y_train, S2) + \
                    0.5 * len(self.X_train) * jnp.log(2*jnp.pi)
        grad_fun = jit(grad(mll))

        momentums = jnp.zeros_like(theta)
        scales = jnp.zeros_like(theta)
        if method == 'SGD':
            for _ in range(num_steps):
                grads = grad_fun(theta)

                momentums = 0.9 * momentums + 0.1 * grads

                scales = 0.9 * scales + 0.1 * grads ** 2

                theta -= lr * momentums / jnp.sqrt(scales + 1e-5)

            self.l, self.sigma_f = theta
            self.fit(self.X_train, self.Y_train)

        elif method == 'BFGS':
            return 'BFGS is not finish yet'
        else:
            return 'Please choose a method correctly'
        
    def mll(self, theta):
        K = self.kernel(self.X_train, self.X_train, l=theta[0], sigma_f=theta[1]) + self.sigma_y**2 * jnp.eye(len(self.X_train))
        L = cholesky(K)
        S1 = solve_triangular(L, self.Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        return jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * jnp.dot(self.Y_train, S2) + 0.5 * len(self.X_train) * jnp.log(2 * jnp.pi)

    def optim_np(self, bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B'):
        theta = jnp.array([self.l, self.sigma_f])

        optimized_mll = jit(self.mll)

        def mll_to_minimize(theta):
            return optimized_mll(theta)

        res = minimize(mll_to_minimize, [self.l, self.sigma_f], bounds=bounds, method=method)
        self.l, self.sigma_f = res.x  # Update self.l and self.sigma_f with the optimized values
        self.fit(self.X_train, self.Y_train)
        return res


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
        
