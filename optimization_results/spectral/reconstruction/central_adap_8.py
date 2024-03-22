from typing import List

import jax, jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class CentralEighthOrderAdapReconstruction(SpatialReconstruction):
    """CentralEighthOrderAdapReconstruction 

    8th order stencil for reconstruction at the cell face
                             x
    |     |      |     |     |     |     |     |     |
    | i-3 | i-2  | i-1 |  i  | i+1 | i+2 | i+3 | i+4 | 
    |     |      |     |     |     |     |     |     |
    """

    required_halos = 4
    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            is_mesh_stretching: List = None,
            cell_sizes: List = None
        ) -> None:
        super(CentralEighthOrderAdapReconstruction, self).__init__(nh=nh,
                                                                   inactive_axes=inactive_axes,
                                                                   offset=offset)

        # TODO
        assert is_mesh_stretching is not None, "is_mesh_stretching is not optional for adap stencil."
        assert cell_sizes is not None, "cell_sizes is not optional for adap stencil."

        self.order = 8
        self.array_slices([(-4, -3, -2, -1, 0, 1, 2, 3)])

        self.coeffs = []
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                cell_sizes_i = cell_sizes_i[self.s_nh_xi[i]]

                # CELL SIZES
                delta_x0 = cell_sizes_i[self.s_mesh[i][0]] # x_{i-3}
                delta_x1 = cell_sizes_i[self.s_mesh[i][1]] # x_{i-2}
                delta_x2 = cell_sizes_i[self.s_mesh[i][2]] # x_{i-1}
                delta_x3 = cell_sizes_i[self.s_mesh[i][3]] # x_{i}
                delta_x4 = cell_sizes_i[self.s_mesh[i][4]] # x_{i+1}
                delta_x5 = cell_sizes_i[self.s_mesh[i][5]] # x_{i+2}
                delta_x6 = cell_sizes_i[self.s_mesh[i][6]] # x_{i+3}
                delta_x7 = cell_sizes_i[self.s_mesh[i][7]] # x_{i+4}

                # DISTANCE TO i+1/2
                d0 = - delta_x3 - delta_x2 - delta_x1 - 0.5 * delta_x0
                d1 = - delta_x3 - delta_x2 - 0.5 * delta_x1
                d2 = - delta_x3 - 0.5 * delta_x2
                d3 = - 0.5 * delta_x3
                d4 = 0.5 * delta_x4
                d5 = delta_x4 + 0.5 * delta_x5
                d6 = delta_x4 + delta_x5 + 0.5 * delta_x6
                d7 = delta_x4 + delta_x5 + delta_x6 + 0.5 * delta_x7

                # COEFFICIENTS
                c0 = - (d1 * d2 * d3 * d4 * d5 * d6 * d7) \
                    / ((d0 - d1) * (d0 - d2) * (d0 - d3) * (d0 - d4) * (d0 - d5) * (d0 - d6) * (d0 - d7))
                c1 = (d0 * d2 * d3 * d4 * d5 * d6 * d7) \
                    / ((d0 - d1) * (d1 - d2) * (d1 - d3) * (d1 - d4) * (d1 - d5) * (d1 - d6) * (d1 - d7))
                c2 = - (d0 * d1 * d3 * d4 * d5 * d6 * d7) \
                    / ((d0 - d2) * (d1 - d2) * (d2 - d3) * (d2 - d4) * (d2 - d5) * (d2 - d6) * (d2 - d7))
                c3 = (d0 * d1 * d2 * d4 * d5 * d6 * d7) \
                    / ((d0 - d3) * (d1 - d3) * (d2 - d3) * (d3 - d4) * (d3 - d5) * (d3 - d6) * (d3 - d7))
                c4 = - (d0 * d1 * d2 * d3 * d5 * d6 * d7) \
                    / ((d0 - d4) * (d1 - d4) * (d2 - d4) * (d3 - d4) * (d4 - d5) * (d4 - d6) * (d4 - d7))
                c5 = (d0 * d1 * d2 * d3 * d4 * d6 * d7) \
                    / ((d0 - d5) * (d1 - d5) * (d2 - d5) * (d3 - d5) * (d4 - d5) * (d5 - d6) * (d5 - d7))
                c6 = - (d0 * d1 * d2 * d3 * d4 * d5 * d7) \
                    / ((d0 - d6) * (d1 - d6) * (d2 - d6) * (d3 - d6) * (d4 - d6) * (d5 - d6) * (d6 - d7))
                c7 = (d0 * d1 * d2 * d3 * d4 * d5 * d6) \
                    / ((d0 - d7) * (d1 - d7) * (d2 - d7) * (d3 - d7) * (d4 - d7) * (d5 - d7) * (d6 - d7))

            else:
                d = 2048.0
                c0 = -5.0 / d
                c1 = 49.0 / d
                c2 = -245.0 / d 
                c3 = 1225.0 / d
                c4 = 1225.0 / d
                c5 = -245.0 / d
                c6 = 49.0 / d
                c7 = -5.0 / d

            self.coeffs.append([jnp.array(c0), jnp.array(c1), jnp.array(c2),
                                jnp.array(c3), jnp.array(c4), jnp.array(c5),
                                jnp.array(c6), jnp.array(c7),])

    def reconstruct_xi(
            self,
            buffer: jnp.ndarray,
            axis: int,
            **kwargs
        ) -> jnp.ndarray:

        s1_  = self.s_[axis]
        coeffs = self.coeffs[axis]

        if coeffs[0].ndim == 4:
            c_xi = []
            device_id = jax.lax.axis_index(axis_name="i")
            for m in range(self.order):
                c_xi.append(coeffs[m][device_id])
        else:
            c_xi = coeffs

        cell_state_xi = c_xi[0] * buffer[s1_[0]] \
            + c_xi[1] * buffer[s1_[1]] \
            + c_xi[2] * buffer[s1_[2]] \
            + c_xi[3] * buffer[s1_[3]] \
            + c_xi[4] * buffer[s1_[4]] \
            + c_xi[5] * buffer[s1_[5]] \
            + c_xi[6] * buffer[s1_[6]] \
            + c_xi[7] * buffer[s1_[7]]

        return cell_state_xi