from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.shock_sensor.ducros_squared import DucrosSquared
from jaxfluids.domain.domain_information import DomainInformation

class TENO5DV(SpatialReconstruction):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            cell_sizes: tuple = None,
            domain_information: DomainInformation = None,
            **kwargs) -> None:
        super(TENO5DV, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        self.dim = domain_information.dim
        if self.dim < 2:
            raise ValueError("TENO5DV is only implemented for 2D and 3D problems.")
        self.nx = domain_information.global_number_of_cells[0]
        self.ny = domain_information.global_number_of_cells[1]
        self.nz = domain_information.global_number_of_cells[2]
        # Coefficients for 5-th order convergence
        # self.dr_ = [1/10, 6/10, 3/10]
        # Coefficients for optimized spectral properties
        # self.dr_ = [0.05, 0.55, 0.40]
        eta_eno = 0.4558
        ds_eno = [eta_eno / 4.0, (2.0 + eta_eno) / 4.0, (1.0 - eta_eno) / 2.0]
        eta_lnr = 0.4246
        ds_lnr = [eta_lnr / 4.0, (2.0 + eta_lnr) / 4.0, (1.0 - eta_lnr) / 2.0]
        eta_v = 0.0539
        ds_v = [eta_v / 4.0, (2.0 + eta_v) / 4.0, (1.0 - eta_v) / 2.0]
        self.dr_ = [ds_eno, ds_lnr, ds_v]
        self.ducros_cutoff = 0.0135

        self.cr_ = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6]
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-5

        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
        self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])

        self.shock_sensor = DucrosSquared(domain_information)
        self.fs = None
        self.fs_slice = [
            [
                jnp.s_[..., j:self.nx+j+1, 1:self.ny+1, None:None if "z" in inactive_axes else 1:self.nz+1],
                jnp.s_[..., 1:self.nx+1, j:self.ny+j+1, None:None if "z" in inactive_axes else 1:self.nz+1],
                jnp.s_[..., 1:self.nx+1, 1:self.ny+1, j:self.nz+j+1],
            
            ] for j in range(2)
        ]

    def compute_ducros_sensor(self, vels: jnp.ndarray, axis: int) -> None:
        self.fs = self.shock_sensor.compute_sensor_function(vels, axis, self.ducros_cutoff)

    def reconstruct_xi(self, 
            buffer: jnp.ndarray, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> jnp.ndarray:
        s1_ = self.s_[j][axis]
        fs = self.fs[self.fs_slice[j][axis]]

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

        tau_5 = jnp.abs(beta_0 - beta_2)

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_5 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_5 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_5 / (beta_2 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2)

        # dilational structures
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0.0, 1.0)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0.0, 1.0)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0.0, 1.0)

        is_eno = jnp.where(delta_0 + delta_1 + delta_2 < 2.9, 1.0, 0.0)

        w_dil0 = delta_0 * self.dr_[0][0] * is_eno + self.dr_[1][0] * (1.0 - is_eno)
        w_dil1 = delta_1 * self.dr_[0][1] * is_eno + self.dr_[1][1] * (1.0 - is_eno)
        w_dil2 = delta_2 * self.dr_[0][2] * is_eno + self.dr_[1][2] * (1.0 - is_eno)

        # vortical structures
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < 1.0e-10, 0.0, 1.0)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < 1.0e-10, 0.0, 1.0)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < 1.0e-10, 0.0, 1.0)

        w_v0 = delta_0 * self.dr_[2][0]
        w_v1 = delta_1 * self.dr_[2][1]
        w_v2 = delta_2 * self.dr_[2][2]

        # switch based on ducros sensor
        w0 = fs * w_dil0 + (1.0 - fs) * w_v0
        w1 = fs * w_dil1 + (1.0 - fs) * w_v1
        w2 = fs * w_dil2 + (1.0 - fs) * w_v2


        
        # SHARP CUTOFF FUNCTION
        # delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        # delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        # delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)

        # w0 = delta_0 * self.dr_[0]
        # w1 = delta_1 * self.dr_[1]
        # w2 = delta_2 * self.dr_[2]

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 

        p_0 = self.cr_[0][0] * buffer[s1_[0]] + self.cr_[0][1] * buffer[s1_[1]] + self.cr_[0][2] * buffer[s1_[2]]
        p_1 = self.cr_[1][0] * buffer[s1_[1]] + self.cr_[1][1] * buffer[s1_[2]] + self.cr_[1][2] * buffer[s1_[3]]
        p_2 = self.cr_[2][0] * buffer[s1_[2]] + self.cr_[2][1] * buffer[s1_[3]] + self.cr_[2][2] * buffer[s1_[4]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2
        return cell_state_xi_j