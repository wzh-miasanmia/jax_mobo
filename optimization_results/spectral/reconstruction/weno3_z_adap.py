from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno3

class WENO3ZADAP(SpatialReconstruction):
    ''' Don and Borges - 2013 - Accuracy of the WENO conservative FD schemes '''

    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            is_mesh_stretching: List = None,
            cell_sizes: List = None,
            offset: int = 0,
            **kwargs) -> None:
        super(WENO3ZADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self.dr_uniform = [1/3, 2/3]

        self.cr_uniform = [
            [-0.5, 1.5], [0.5, 0.5]
        ]

        self.betar_uniform = [
            [-1.0, 1.0],
            [-1.0, 1.0]
        ] 

        self._stencil_size = 4
        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])

        self.cr_, self.betar_, self.dr_ = compute_coefficients_stretched_mesh_weno3(
            cr_uniform=self.cr_uniform,
            betar_uniform=self.betar_uniform,
            dr_uniform=self.dr_uniform,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes,
            slices_mesh=self.s_mesh,
            slices_cell_sizes=self.s_nh_xi)

    def reconstruct_xi(
            self, 
            buffer: jnp.ndarray, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs
        ) -> jnp.ndarray:
        s1_ = self.s_[j][axis]

        if self.cr_[j][axis][0][0].ndim == 4:
            cr_ = [[], []]
            betar_ = [[], []]
            dr_ = []
            device_id = jax.lax.axis_index(axis_name="i")
            for m in range(2):
                for n in range(2):
                    cr_[n].append(self.cr_[j][axis][n][m][device_id])
            for m in range(2):
                for n in range(2):
                   betar_[n].append(self.betar_[j][axis][n][m][device_id])
            for m in range(2):
                dr_.append(self.dr_[j][axis][m][device_id])
        else:
            cr_ = self.cr_[j][axis]
            betar_ = self.betar_[j][axis]
            dr_ = self.dr_[j][axis]

        beta_0 = (betar_[0][0] * buffer[s1_[0]] + betar_[0][1] * buffer[s1_[1]]) \
            * (betar_[0][0] * buffer[s1_[0]] + betar_[0][1] * buffer[s1_[1]])

        beta_1 = (betar_[1][0] * buffer[s1_[1]] + betar_[1][1] * buffer[s1_[2]]) \
            * (betar_[1][0] * buffer[s1_[1]] + betar_[1][1] * buffer[s1_[2]])

        tau_3 = jnp.abs(beta_0 - beta_1)

        alpha_z_0 = dr_[0] * (1.0 + tau_3 / (beta_0 + self.eps) )
        alpha_z_1 = dr_[1] * (1.0 + tau_3 / (beta_1 + self.eps) )

        one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1)

        omega_z_0 = alpha_z_0 * one_alpha_z
        omega_z_1 = alpha_z_1 * one_alpha_z

        p_0 = cr_[0][0] * buffer[s1_[0]] + cr_[0][1] * buffer[s1_[1]] 
        p_1 = cr_[1][0] * buffer[s1_[1]] + cr_[1][1] * buffer[s1_[2]]

        cell_state_xi_j = omega_z_0 * p_0 + omega_z_1 * p_1

        return cell_state_xi_j