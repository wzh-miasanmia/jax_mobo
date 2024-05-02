import numpy as np
import matplotlib.pyplot as plt
from tgv_postprocessing import calculate_reference, create_spectrum

schemes = ["weno9", "teno6", "teno8"]
# define lines
line_styles = ['-', '--', '-.']
markers = ['s', 'o', '^']  # square, circle, triangle up, diamond
colors = ['blue', 'red', 'green']

# define domain
y_lower=1e-6
y_upper=1e-1
x_lower = 10**0.2
x_upper = 10**1.5
plt.figure(figsize=(5, 4), dpi=200)

for scheme, line_style, marker, color in zip(schemes, line_styles, markers, colors):
    ## Reading files and calculating, same as before
    target_folder_name = f"{scheme}/results/tgv"
    file = f"{target_folder_name}/domain/data_*.h5"
    spectrum_data = create_spectrum(file)
    spectrum_ref  = calculate_reference(spectrum_data["data"])
    plt.loglog(np.arange(0, spectrum_data["data"].shape[0]),
            spectrum_data["data"][:, 1],
            linestyle=line_style,
            marker=marker,
            markersize=5, 
            markerfacecolor='none',
            markeredgewidth=1,
            markevery=3,
            color=color,
            linewidth=1,
            label=f'{scheme.upper()}')

plt.loglog(spectrum_data["data"][:, 0], spectrum_ref, c='black', linestyle='--', linewidth=0.8, label='$Ak^{-5/3}$')
plt.xlabel(r"$k$")
plt.ylabel(r"$E(k)$")
plt.ylim(y_lower, y_upper)
plt.xlim(x_lower, x_upper)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("./tke_no_dv.png", dpi=200)