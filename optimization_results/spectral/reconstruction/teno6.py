from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class TENO6(SpatialReconstruction):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(TENO6, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for 6-th order convergence
        self.dr_ = [0.050, 0.450, 0.300, 0.200]
        # # Coefficients for optimized spectral properties
        # self.dr_ = [0.054, 0.462, 0.300, 0.184] 

        self.cr_ = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6], 
            [3/12, 13/12, -5/12, 1/12]
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-7

        self._stencil_size = 6
        self.array_slices([range(-3, 3, 1), range(2, -4, -1)])
        self.stencil_slices([range(0, 6, 1), range(5, -1, -1)])

    def reconstruct_xi(
            self, 
            buffer: jnp.ndarray, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> jnp.ndarray:
        s1_ = self.s_[j][axis]

        beta_0 = 13.0 / 12.0 * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) \
            * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) \
            + 1.0 / 4.0 * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]]) \
            * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]])
        beta_1 = 13.0 / 12.0 * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
            * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
            + 1.0 / 4.0 * (buffer[s1_[1]] - buffer[s1_[3]]) * (buffer[s1_[1]] - buffer[s1_[3]])
        beta_2 = 13.0 / 12.0 * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
            * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
            + 1.0 / 4.0 * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]]) \
            * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]])
        beta_3 = 1.0 / 240.0 * (
            buffer[s1_[2]]   * (2107  * buffer[s1_[2]] - 9402  * buffer[s1_[3]] + 7042 * buffer[s1_[4]] - 1854 * buffer[s1_[5]]) \
            + buffer[s1_[3]] * (11003 * buffer[s1_[3]] - 17246 * buffer[s1_[4]] + 4642 * buffer[s1_[5]]) \
            + buffer[s1_[4]] * (7043  * buffer[s1_[4]] - 3882  * buffer[s1_[5]]) \
            + 547 * buffer[s1_[5]] * buffer[s1_[5]]
        )

        beta_6 = 1.0 / 120960 * (
            271779 * buffer[s1_[0]] * buffer[s1_[0]] + \
            buffer[s1_[0]] * (-2380800 * buffer[s1_[1]] + 4086352  * buffer[s1_[2]]  - 3462252  * buffer[s1_[3]] + 1458762 * buffer[s1_[4]]  - 245620  * buffer[s1_[5]]) + \
            buffer[s1_[1]] * (5653317  * buffer[s1_[1]] - 20427884 * buffer[s1_[2]]  + 17905032 * buffer[s1_[3]] - 7727988 * buffer[s1_[4]]  + 1325006 * buffer[s1_[5]]) + \
            buffer[s1_[2]] * (19510972 * buffer[s1_[2]] - 35817664 * buffer[s1_[3]]  + 15929912 * buffer[s1_[4]] - 2792660 * buffer[s1_[5]]) + \
            buffer[s1_[3]] * (17195652 * buffer[s1_[3]] - 15880404 * buffer[s1_[4]]  + 2863984  * buffer[s1_[5]]) + \
            buffer[s1_[4]] * (3824847  * buffer[s1_[4]] - 1429976  * buffer[s1_[5]]) + \
            139633 * buffer[s1_[5]] * buffer[s1_[5]]
            )

        tau_6 = jnp.abs(beta_6 - 1/6 * (beta_0 + 4 * beta_1 + beta_2))

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_6 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_6 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_6 / (beta_2 + self.eps))**self.q
        gamma_3 = (self.C + tau_6 / (beta_3 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2 + gamma_3)

        # SHARP CUTOFF FUNCTION
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)
        delta_3 = jnp.where(gamma_3 * one_gamma_sum < self.CT, 0, 1)

        w0 = delta_0 * self.dr_[0]
        w1 = delta_1 * self.dr_[1]
        w2 = delta_2 * self.dr_[2]
        w3 = delta_3 * self.dr_[3]

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + w3 + self.eps)
        # one_dk = 1.0 / (w0 + w1 + w2 + w3)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 
        omega_3 = w3 * one_dk 

        p_0 = self.cr_[0][0] * buffer[s1_[0]] + self.cr_[0][1] * buffer[s1_[1]] + self.cr_[0][2] * buffer[s1_[2]]
        p_1 = self.cr_[1][0] * buffer[s1_[1]] + self.cr_[1][1] * buffer[s1_[2]] + self.cr_[1][2] * buffer[s1_[3]]
        p_2 = self.cr_[2][0] * buffer[s1_[2]] + self.cr_[2][1] * buffer[s1_[3]] + self.cr_[2][2] * buffer[s1_[4]]
        p_3 = self.cr_[3][0] * buffer[s1_[2]] + self.cr_[3][1] * buffer[s1_[3]] + self.cr_[3][2] * buffer[s1_[4]] + self.cr_[3][3] * buffer[s1_[5]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3
        return cell_state_xi_j