import numpy as np

def is_non_dominated_np(Y, deduplicate=True):
    Y1 = np.expand_dims(Y, axis=-3) # [1, n, m]
    Y2 = np.expand_dims(Y, axis=-2) # [n, 1, m]
    dominates = (Y1 <= Y2).all(axis=-1) & (Y1 < Y2).any(axis=-1)
    nd_mask = ~(dominates.any(axis=-1))

    if deduplicate:
        # remove duplicates
        indices = np.all(Y1 == Y2, axis=-1).argmax(axis=-1)
        keep = np.zeros_like(nd_mask, dtype=float)
        keep[indices] = 1.0
        return nd_mask & keep.astype(bool)
    
    return nd_mask

# test
Y = np.array([[1, 6], [2, 5], [3, 4], [5, 2], [4, 5], [5, 5], [5, 4], [4, 3]])
mask_no_dedup = is_non_dominated_np(Y)

# determine the number of points in Y that are Pareto optimal
num_new_pareto = mask_no_dedup[-Y.shape[-2] :].sum()

# non-dominated points
Y_nd = Y[mask_no_dedup]
print(num_new_pareto, Y_nd)