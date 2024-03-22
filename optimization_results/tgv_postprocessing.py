import h5py, glob
import matplotlib.pyplot as plt
import numpy as np

H5_FILE = "teno6/results/tgv/domain/data_*.h5"
TITLE = "TENO6"
# cell number in each direction
N = 64
# average velocity
U0 = 1.0

def get_file_list(file):
    """
    Get the latest file from a list of files
    """
    files = glob.glob(file)
    assert len(files) > 0, f"File {file} not found"
    sorted_files = sorted(files, key=lambda x: float(x.split("_")[-1].replace(".h5", "")))
    # return the file name of the last time step
    return sorted_files[-1]


def get_jaxfluids_results(file, quantities):
    """
    Get the results from a jaxfluids simulation
    """
    file = get_file_list(file)
    print(f"Load {file}")
    data_dict = {}
    with h5py.File(file, "r") as h5file:
        for quantity in quantities:
            # since velocity is a vector
            if quantity == "velocity":
                data_dict[quantity] = {}
                data_dict[quantity]["velocity_x"] = h5file["primitives/velocity"][..., 0]
                data_dict[quantity]["velocity_y"] = h5file["primitives/velocity"][..., 1]
                data_dict[quantity]["velocity_z"] = h5file["primitives/velocity"][..., 2]
            else:
                data_dict[quantity] = h5file[f"primitives/{quantity}"][:]
    return data_dict

def create_spectrum(file: str):
    """
    Create the spectrum from the velocity field
    """
    # we only need the velocity field
    velocity = get_jaxfluids_results(file, ["velocity"])["velocity"]
    velocity_x = velocity['velocity_x']
    velocity_y = velocity['velocity_y']
    velocity_z = velocity['velocity_z']

    eps = 1e-50  # to void log(0)

    U = velocity_x / U0
    V = velocity_y / U0
    W = velocity_z / U0

    amplsU = abs(np.fft.fftn(U) / U.size)
    amplsV = abs(np.fft.fftn(V) / V.size)
    amplsW = abs(np.fft.fftn(W) / W.size)
    
    EK_U = amplsU ** 2
    EK_V = amplsV ** 2
    EK_W = amplsW ** 2

    EK_U = np.fft.fftshift(EK_U)
    EK_V = np.fft.fftshift(EK_V)
    EK_W = np.fft.fftshift(EK_W)

    box_sidex = np.shape(EK_U)[0]
    box_sidey = np.shape(EK_U)[1]
    box_sidez = np.shape(EK_U)[2]

    box_radius = int(np.ceil((np.sqrt((box_sidex) ** 2 + (box_sidey) ** 2 + (box_sidez) ** 2)) / 2.) + 1)

    centerx = int(box_sidex / 2)
    centery = int(box_sidey / 2)
    centerz = int(box_sidez / 2)

    EK_U_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
    EK_V_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
    EK_W_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            for k in range(box_sidez):
                wn = int(np.round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2 + (k - centerz) ** 2)))
                EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i, j, k]
                EK_V_avsphr[wn] = EK_V_avsphr[wn] + EK_V[i, j, k]
                EK_W_avsphr[wn] = EK_W_avsphr[wn] + EK_W[i, j, k]

    EK_avsphr = 0.5 * (EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)
    real_size = len(np.fft.rfft(U[:, 0, 0]))

    dataout = np.zeros((box_radius, 2))
    dataout[:, 0] = np.arange(0, len(dataout))
    dataout[:, 1] = EK_avsphr[0:len(dataout)]

    return {
        "real_size": real_size, 
        "data": dataout
    }
    

def calculate_A(spectrum):
    """
    Calculate the A coefficient from the spectrum 
    """
    effective_wn = slice(7, 33)
    wn_for_interpolation = spectrum[effective_wn, 0].reshape(-1, 1)
    ke_for_interpolation = spectrum[effective_wn, 1].reshape(-1, 1)
    p = np.power(wn_for_interpolation, -5 / 3)
    A = (np.linalg.inv(p.T.dot(p))).dot(p.T).dot(ke_for_interpolation)
    return A.squeeze()

def calculate_reference(spectrum):
    """
    Calculate the reference from the spectrum
    """
    A = calculate_A(spectrum)
    reference = A * np.power(spectrum[:, 0] + 1e-50, -5 / 3)
    return reference

def calculate_Rtke_baseline(file):
    """
    Calculate the ftke from the spectrum and reference
    """
    spectrum_data = create_spectrum(file)["data"]
    spectrum_ref  = calculate_reference(spectrum_data)
    Rtke_baseline = np.sum((np.log(spectrum_data[3:34, 1]) - np.log(spectrum_ref[3:34])) ** 2) / len(spectrum_data[3:34, 1])
    return Rtke_baseline

def plot_tke(
        file: str = H5_FILE,
        plot_reference: bool = True, 
        y_lower: float = 1e-15, 
        y_upper: float = 1, 
        title: str = None,
        save_path: str = None, 
        grid: bool =False
    ):
    """
    Plot the TKE
    """
    spectrum_data = create_spectrum(file)
    spectrum_ref  = calculate_reference(spectrum_data["data"])
    real_size = spectrum_data["real_size"]
    plt.figure(figsize=(5, 4), dpi=200)
    plt.loglog(np.arange(0, real_size),
                spectrum_data["data"][0:real_size, 1],
                c='black',
                linestyle='-',
                linewidth=1,
                label=f'$k<{real_size}$')
    plt.loglog(np.arange(real_size - 1, spectrum_data["data"].shape[0]),
                spectrum_data["data"][real_size - 1:, 1],
                c='black',
                linestyle='--',
                linewidth=1,
                label=f'$k\geq{real_size}$')
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.legend(loc='lower left')
    plt.ylim(y_lower, y_upper)
    if plot_reference:
        plt.loglog(spectrum_data["data"][:, 0], spectrum_ref, c='red', linewidth=0.8, label='$ref$')
    if title:
        plt.title(title)
    if grid:
        plt.grid(which='both')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    # plt.show()

if "__main__" in __name__:
    plot_tke(
        file=H5_FILE,
        plot_reference=True, 
        y_lower=1e-10, 
        y_upper=1, 
        title=TITLE,
        save_path="./tke.png", 
        grid=True
    )
    # print(calculate_ftke(H5_FILE))