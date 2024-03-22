import numpy as np


def periodic(array: np.array, halo_size=2):
    r"""
    Create periodic boundary
    """
    value = array.copy()
    if len(array.shape) == 1:
        value = np.insert(value, 0, array[-halo_size:])
        value = np.append(value, array[:halo_size])
    elif len(array.shape) == 2:
        # horizontal
        for i in range(halo_size):
            value = np.insert(value, 0, array[:, -(i + 1)], axis=1)
        value = np.append(value, array[:, :halo_size], axis=1)
        value = np.insert(value, 0, value[-halo_size:, :], axis=0)
        value = np.append(value, value[halo_size:2 * halo_size, :], axis=0)
    else:
        raise Exception(f"Boundary condition for {len(value)}-dimensional is not implemented yet. ")
    return value


def symmetry_x_fixed_y(array: np.array, halo_size=2):
    r"""
    Create symmetry boundary.
    """
    value = array.copy()
    if len(array.shape) == 1:
        for i in range(halo_size):
            value = np.insert(value, 0, array[i])
            value = np.append(value, array[-(i+1)])
    elif len(array.shape) == 2:
        # horizontal
        for i in range(halo_size):
            value = np.insert(value, 0, array[:, i], axis=1)
            value = np.append(value, array[:, -(i+1)].reshape(-1, 1), axis=1)
        # vertical
        for i in range(halo_size):
            value = np.insert(value, 0, value[0, :], axis=0)
            value = np.append(value, value[-1, :].reshape(1, -1), axis=0)
    else:
        raise Exception(f"Boundary condition for {len(value)}-dimensional is not implemented yet. ")
    return value


def symmetry(array: np.array, halo_size=2):
    r"""
    Create symmetry boundary.
    """
    value = array.copy()
    if len(array.shape) == 1:
        for i in range(halo_size):
            value = np.insert(value, 0, array[i])
            value = np.append(value, array[-(i+1)])
    elif len(array.shape) == 2:
        # horizontal
        for i in range(halo_size):
            value = np.insert(value, 0, array[:, i], axis=1)
            value = np.append(value, array[:, -(i+1)].reshape(-1, 1), axis=1)
        # vertical
        for i in range(halo_size):
            value = np.insert(value, 0, value[i*2, :], axis=0)
            value = np.append(value, value[-(i*2+1), :].reshape(1, -1), axis=0)
    else:
        raise Exception(f"Boundary condition for {len(value)}-dimensional is not implemented yet. ")
    return value


def do_weno5_si(array):
    v1 = array[0]
    v2 = array[1]
    v3 = array[2]
    v4 = array[3]
    v5 = array[4]

    coef_smoothness_1_ = 13.0 / 12.0
    coef_smoothness_2_ = 0.25

    coef_smoothness_11_ = 1.0
    coef_smoothness_12_ = -2.0
    coef_smoothness_13_ = 1.0
    coef_smoothness_14_ = 1.0
    coef_smoothness_15_ = -4.0
    coef_smoothness_16_ = 3.0

    coef_smoothness_21_ = 1.0
    coef_smoothness_22_ = -2.0
    coef_smoothness_23_ = 1.0
    coef_smoothness_24_ = 1.0
    coef_smoothness_25_ = -1.0

    coef_smoothness_31_ = 1.0
    coef_smoothness_32_ = -2.0
    coef_smoothness_33_ = 1.0
    coef_smoothness_34_ = 3.0
    coef_smoothness_35_ = -4.0
    coef_smoothness_36_ = 1.0

    epsilon_weno5_ = 1.0e-6

    s11 = coef_smoothness_11_ * v1 + coef_smoothness_12_ * v2 + coef_smoothness_13_ * v3
    s12 = coef_smoothness_14_ * v1 + coef_smoothness_15_ * v2 + coef_smoothness_16_ * v3
    s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

    s21 = coef_smoothness_21_ * v2 + coef_smoothness_22_ * v3 + coef_smoothness_23_ * v4
    s22 = coef_smoothness_24_ * v2 + coef_smoothness_25_ * v4
    s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

    s31 = coef_smoothness_31_ * v3 + coef_smoothness_32_ * v4 + coef_smoothness_33_ * v5
    s32 = coef_smoothness_34_ * v3 + coef_smoothness_35_ * v4 + coef_smoothness_36_ * v5
    s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32

    one_s1 = 1.0 / ((s1 + epsilon_weno5_) ** 2)
    one_s2 = 1.0 / ((s2 + epsilon_weno5_) ** 2)
    one_s3 = 1.0 / ((s3 + epsilon_weno5_) ** 2)

    a1_weno5 = one_s1 / (one_s1 + one_s2 + one_s3)
    a2_weno5 = one_s2 / (one_s1 + one_s2 + one_s3)
    a3_weno5 = one_s3 / (one_s1 + one_s2 + one_s3)

    return a1_weno5, a2_weno5, a3_weno5


def weno5_si(array, halo_size=2, st=0.01):
    value = array.copy()
    shape = value.shape

    a1_si = np.zeros((np.array(shape) - 2 * halo_size))
    a2_si = a1_si.copy()
    a3_si = a1_si.copy()
    dis = a1_si.copy()

    if len(shape) == 1:
        for i in range(halo_size, shape[0] - halo_size):
            v1 = value[i - 2]
            v2 = value[i - 1]
            v3 = value[i]
            v4 = value[i + 1]
            v5 = value[i + 2]

            a1_weno5, a2_weno5, a3_weno5 = do_weno5_si([v1, v2, v3, v4, v5])

            a1_si[i - 2] = a1_weno5
            a2_si[i - 2] = a2_weno5
            a3_si[i - 2] = a3_weno5
            dis[i - 2] = 1 if a1_weno5 < st or a2_weno5 < st or a3_weno5 < st else 0

    if len(shape) == 2:
        # x direction
        for i in range(halo_size, shape[0] - halo_size):
            for j in range(halo_size, shape[1] - halo_size):
                v1 = value[i, j - 2]
                v2 = value[i, j - 1]
                v3 = value[i, j]
                v4 = value[i, j + 1]
                v5 = value[i, j + 2]

                a1_weno5, a2_weno5, a3_weno5 = do_weno5_si([v1, v2, v3, v4, v5])

                a1_si[i - 2, j - 2] = a1_weno5
                a2_si[i - 2, j - 2] = a2_weno5
                a3_si[i - 2, j - 2] = a3_weno5
                dis[i - 2, j - 2] = 1 if a1_weno5 < st or a2_weno5 < st or a3_weno5 < st else 0
        # y direction
        for i in range(halo_size, shape[0] - halo_size):
            for j in range(halo_size, shape[1] - halo_size):
                v1 = value[i - 2, j]
                v2 = value[i - 1, j]
                v3 = value[i, j]
                v4 = value[i + 1, j]
                v5 = value[i + 2, j]

                a1_weno5, a2_weno5, a3_weno5 = do_weno5_si([v1, v2, v3, v4, v5])

                a1_si[i - 2, j - 2] = a1_weno5
                a2_si[i - 2, j - 2] = a2_weno5
                a3_si[i - 2, j - 2] = a3_weno5
                dis[i - 2, j - 2] = 1 if a1_weno5 < st or a2_weno5 < st or a3_weno5 < st or dis[
                    i - 2, j - 2] == 1 else 0

    return dis, (a1_si, a2_si, a3_si)
