from typing import List
import json
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.shock_sensor.ducros_squared import DucrosSquared
from jaxfluids.domain.domain_information import DomainInformation

# with open("numerical_setup.json", "r") as read_file:
#     data = json.load(read_file)
#     teno6dv_eta_eno = data["scheme_parameters"]["teno6_dv"]["eta_eno"]
#     teno6dv_eta_v = data["scheme_parameters"]["teno6_dv"]["eta_v"]
#     teno6dv_ducros_cutoff = data["scheme_parameters"]["teno6_dv"]["ducros_cutoff"]

class TENO6DV(SpatialReconstruction):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            domain_information: DomainInformation = None,
            **kwargs
            ) -> None:
        super(TENO6DV, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        self.dim = domain_information.dim
        if self.dim < 2:
            raise ValueError("TENO6-DV is only implemented for 2D and 3D problems.")
        self.nx = domain_information.global_number_of_cells[0]
        self.ny = domain_information.global_number_of_cells[1]
        self.nz = domain_information.global_number_of_cells[2]
        self.shock_sensor = DucrosSquared(domain_information)
        self.fs = None
        self.fs_slice = [
            [
                jnp.s_[..., j:self.nx+j+1, 1:self.ny+1, None:None if "z" in inactive_axes else 1:self.nz+1],
                jnp.s_[..., 1:self.nx+1, j:self.ny+j+1, None:None if "z" in inactive_axes else 1:self.nz+1],
                jnp.s_[..., 1:self.nx+1, 1:self.ny+1, j:self.nz+j+1],
            
            ] for j in range(2)
        ]
        eta_eno = 0.8919 # optimized
        ds_eno = [eta_eno / 10.0, (3.0 + 3.0 * eta_eno) / 10.0, 0.3, (2.0 - 2.0 * eta_eno) / 5.0]
        eta_v = 0.4829 # optimized
        ds_v = [eta_v / 10.0, (3.0 + 3.0 * eta_v) / 10.0, 0.3, (2.0 - 2.0 * eta_v) / 5.0]
        self.dr_ = [ds_eno, ds_v]
        self.ducros_cutoff = 0.1145 # optimized
        
        # Coefficients for 6-th order convergence
        # self.dr_ = [0.050, 0.450, 0.300, 0.200]
        # # Coefficients for optimized spectral properties
        # self.dr_ = [0.08, 0.54, 0.300, 0.08] 

        self.cr_ = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6], 
            [3/12, 13/12, -5/12, 1/12]
        ]

        self.C = 1.0
        self.q = 6
        self.CT_eno = 1e-7
        self.CT_vor = 1e-10

        self._stencil_size = 6
        self.array_slices([range(-3, 3, 1), range(2, -4, -1)])
        self.stencil_slices([range(0, 6, 1), range(5, -1, -1)])

    def compute_ducros_sensor(self, vels: jnp.ndarray, axis: int) -> None:
        self.fs = self.shock_sensor.compute_sensor_function(vels, axis, self.ducros_cutoff)

    def reconstruct_xi(
            self, 
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
        beta_3 = jnp.abs(1.0 / 240.0 * (
            buffer[s1_[2]]   * (2107  * buffer[s1_[2]] - 9402  * buffer[s1_[3]] + 7042 * buffer[s1_[4]] - 1854 * buffer[s1_[5]]) \
            + buffer[s1_[3]] * (11003 * buffer[s1_[3]] - 17246 * buffer[s1_[4]] + 4642 * buffer[s1_[5]]) \
            + buffer[s1_[4]] * (7043  * buffer[s1_[4]] - 3882  * buffer[s1_[5]]) \
            + 547 * buffer[s1_[5]] * buffer[s1_[5]]
        ))

        beta_6 = jnp.abs(1.0 / 120960 * (
            271779 * buffer[s1_[0]] * buffer[s1_[0]] + \
            buffer[s1_[0]] * (-2380800 * buffer[s1_[1]] + 4086352  * buffer[s1_[2]]  - 3462252  * buffer[s1_[3]] + 1458762 * buffer[s1_[4]]  - 245620  * buffer[s1_[5]]) + \
            buffer[s1_[1]] * (5653317  * buffer[s1_[1]] - 20427884 * buffer[s1_[2]]  + 17905032 * buffer[s1_[3]] - 7727988 * buffer[s1_[4]]  + 1325006 * buffer[s1_[5]]) + \
            buffer[s1_[2]] * (19510972 * buffer[s1_[2]] - 35817664 * buffer[s1_[3]]  + 15929912 * buffer[s1_[4]] - 2792660 * buffer[s1_[5]]) + \
            buffer[s1_[3]] * (17195652 * buffer[s1_[3]] - 15880404 * buffer[s1_[4]]  + 2863984  * buffer[s1_[5]]) + \
            buffer[s1_[4]] * (3824847  * buffer[s1_[4]] - 1429976  * buffer[s1_[5]]) + \
            139633 * buffer[s1_[5]] * buffer[s1_[5]]
            ))

        tau_6 = jnp.abs(beta_6 - 1/6 * (beta_0 + 4 * beta_1 + beta_2))

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_6 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_6 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_6 / (beta_2 + self.eps))**self.q
        gamma_3 = (self.C + tau_6 / (beta_3 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2 + gamma_3)

        # SHARP CUTOFF FUNCTION
        delta_eno0 = jnp.where(gamma_0 * one_gamma_sum < self.CT_eno, 0, 1)
        delta_eno1 = jnp.where(gamma_1 * one_gamma_sum < self.CT_eno, 0, 1)
        delta_eno2 = jnp.where(gamma_2 * one_gamma_sum < self.CT_eno, 0, 1)
        delta_eno3 = jnp.where(gamma_3 * one_gamma_sum < self.CT_eno, 0, 1)

        delta_vor0 = jnp.where(gamma_0 * one_gamma_sum < self.CT_vor, 0, 1)
        delta_vor1 = jnp.where(gamma_1 * one_gamma_sum < self.CT_vor, 0, 1)
        delta_vor2 = jnp.where(gamma_2 * one_gamma_sum < self.CT_vor, 0, 1)
        delta_vor3 = jnp.where(gamma_3 * one_gamma_sum < self.CT_vor, 0, 1)

        w0 = delta_eno0 * self.dr_[0][0] * fs + (1.0 - fs) * delta_vor0 * self.dr_[1][0]
        w1 = delta_eno1 * self.dr_[0][1] * fs + (1.0 - fs) * delta_vor1 * self.dr_[1][1]
        w2 = delta_eno2 * self.dr_[0][2] * fs + (1.0 - fs) * delta_vor2 * self.dr_[1][2]
        w3 = delta_eno3 * self.dr_[0][3] * fs + (1.0 - fs) * delta_vor3 * self.dr_[1][3]

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