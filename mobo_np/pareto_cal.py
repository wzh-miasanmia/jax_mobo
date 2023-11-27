import numpy as np

def is_non_dominated_np(Y, deduplicate=True):
    Y1 = np.expand_dims(Y, axis=-3) # [1, n, m]
    Y2 = np.expand_dims(Y, axis=-2) # [n, 1, m]
    dominates = (Y1 >= Y2).all(axis=-1) & (Y1 > Y2).any(axis=-1) # default is maximization problem
    nd_mask = ~(dominates.any(axis=-1))

    if deduplicate:
        # remove duplicates
        indices = np.all(Y1 == Y2, axis=-1).argmax(axis=-1)
        keep = np.zeros_like(nd_mask, dtype=float)
        keep[indices] = 1.0
        return nd_mask & keep.astype(bool)
    
    return nd_mask
