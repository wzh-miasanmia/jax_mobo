import numpy as np
import matplotlib.pyplot as plt
from imp_postprocessing import get_highorder_dissipation_rate, get_effective_dissipation_rate

schemes = ["weno9", "teno6", "teno8", "teno6dv"]
fig, axs = plt.subplots(1, len(schemes), figsize=(20, 5))

# Initialize variables to determine the global color mapping range
global_min = np.inf
global_max = -np.inf

# First loop to determine the global minimum and maximum values
for scheme in schemes:
    target_folder_name = f"{scheme}/64*64/results/fullimplosion"
    effective_dissipation_rate = get_effective_dissipation_rate(target_folder_name)
    file = f"{target_folder_name}/domain/data_*.h5"
    highorder_dissipation_rate = get_highorder_dissipation_rate(file)
    adi_to_ref = -(effective_dissipation_rate - highorder_dissipation_rate)
    epsilon = np.where(adi_to_ref < 0, adi_to_ref, 0)

    # Update the global minimum and maximum values
    global_min = min(global_min, np.min(epsilon))
    global_max = max(global_max, np.max(epsilon))

# Now loop again to plot the subplots and apply the global color mapping range
for ax, scheme in zip(axs, schemes):
    ## Reading files and calculating, same as before
    target_folder_name = f"{scheme}/64*64/results/fullimplosion"
    effective_dissipation_rate = get_effective_dissipation_rate(target_folder_name)
    file = f"{target_folder_name}/domain/data_*.h5"
    highorder_dissipation_rate = get_highorder_dissipation_rate(file)
    adi_to_ref = -(effective_dissipation_rate - highorder_dissipation_rate)
    epsilon = np.where(adi_to_ref < 0, adi_to_ref, 0)
    ## Plotting, using the global color range
    im = ax.imshow(epsilon, cmap='gray', extent=[0, 0.3, 0, 0.3], vmin=global_min, vmax=global_max, origin='lower')
    ax.set_title(r"$|\epsilon_{n,r}^{disper}|$ with " + scheme.upper())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks([0, 0.15, 0.3])
    ax.set_yticks([0, 0.15, 0.3])

# Adding a shared color bar
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Parameters are [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax, shrink=0.85)

plt.tight_layout()
plt.savefig("disper_results.png")
plt.show()
