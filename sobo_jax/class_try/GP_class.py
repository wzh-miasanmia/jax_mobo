import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

import numpy as np
from numpy.linalg import cholesky as np_cholesky
from scipy.linalg import solve_triangular as np_solve_triangular

from jaxopt import ScipyMinimize

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
        self.K = self.kernel(self.X_train, self.X_train, self.l, self.sigma_f) + self.sigma_y**2 * jnp.eye(len(self.X_train))
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, self.Y_train, lower=True), lower=False)


    def optim_np(self, method='L-BFGS-B'): # only availiable for numpy
        X_train = np.asarray(self.X_train)
        Y_train = np.asarray(self.Y_train)
        noise = self.sigma_y
        def np_kernel(X1, X2, l=1.0, sigma_f=1.0):
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
            return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
            
        def mll(X_train, Y_train, noise):
            Y_train = Y_train.ravel()
            def mll_calculate(theta):
                K = np_kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
                    noise**2 * np.eye(len(X_train))
                L = np_cholesky(K)
                
                S1 = np_solve_triangular(L, Y_train, lower=True)
                S2 = np_solve_triangular(L.T, S1, lower=False)
                
                return np.sum(np.log(np.diagonal(L))) + \
                    0.5 * Y_train.dot(S2) + \
                    0.5 * len(X_train) * np.log(2*np.pi)
            return mll_calculate
        res = minimize(mll(X_train, Y_train, noise), [1,1], method=method)
        self.l, self.sigma_f = res.x  # Update self.l and self.sigma_f with the optimized values
        self.fit(self.X_train, self.Y_train)
        return res

    def optim_jaxopt(self, method='L-BFGS-B'): # try jaxopt library 
        def mll(theta):
            K = self.kernel(self.X_train, self.X_train, l=theta[0], sigma_f=theta[1]) + \
                self.sigma_y**2 * jnp.eye(len(self.X_train))
            L = cholesky(K)
            S1 = solve_triangular(L, self.Y_train, lower=True)
            S2 = solve_triangular(L.T, S1, lower=False)
            return jnp.sum(jnp.log(jnp.diag(L))) + \
                    0.5 * jnp.dot(self.Y_train, S2) + \
                    0.5 * len(self.X_train) * jnp.log(2*jnp.pi)

        scipy_minimize = ScipyMinimize(fun=mll, method=method, tol=1e-6)
        x0 = jnp.array([self.l, self.sigma_f])
        result = scipy_minimize.run(x0)
        self.l, self.sigma_f = result.params
        self.fit(self.X_train, self.Y_train)

        return result


    def predict(self, X, return_std=False, return_cov=False):
        K_s = self.kernel(self.X_train, X, self.l, self.sigma_f)
        K_ss = self.kernel(X, X, self.l, self.sigma_f) + 1e-8 * jnp.eye(len(X))
        mu_s = K_s.T.dot(self.alpha)
        v = solve_triangular(self.L, K_s, lower=True)
        cov_s = K_ss - v.T.dot(v)
        # if return_std and return_cov:
        #     return mu_s, jnp.sqrt(jnp.diag(cov_s)), cov_s
        # elif return_std:
        #     return mu_s, jnp.sqrt(jnp.diag(cov_s))
        # elif return_cov:
        #     return mu_s, cov_s
        # else:
        #     return mu_s

        if return_std and return_cov:
            return np.asarray(mu_s), np.asarray(jnp.sqrt(jnp.diag(cov_s))), np.asarray(cov_s)
        elif return_std:
            return np.asarray(mu_s), np.asarray(jnp.sqrt(jnp.diag(cov_s)))
        elif return_cov:
            return np.asarray(mu_s), np.asarray(cov_s)
        else:
            return np.asarray(mu_s)
        