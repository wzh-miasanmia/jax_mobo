import h5py, glob
import matplotlib.pyplot as plt
import numpy as np

def get_file_list(file):
    """
    Get the latest file from a list of files
    """
    files = glob.glob(file)
    assert len(files) > 0, f"File {file} not found"
    # Check for files ending with 'nan'
    for file in files:
        if file.endswith('nan.h5'):
            return file
    sorted_files = sorted(files, key=lambda x: float(x.split("_")[-1].replace(".h5", "")))
    # return the file name of the latest time step
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

def get_highorder_dissipation_rate(file):
    """
    Get the ho_dissipation_rate from a jaxfluids simulation
    """
    file = get_file_list(file)
    print(f"Load {file}")
    with h5py.File(file, "r") as h5file:
        highorder_dissipation_rate = h5file['miscellaneous']['highorder_dissipation_rate'][...]
    
    return np.squeeze(highorder_dissipation_rate, axis=0)

def get_effective_dissipation_rate(target_folder_name):

    with h5py.File(f"{target_folder_name}/domain/final_effective_dissipation_rate.h5", "r") as h5file:
        effective_dissipation_rate = np.array(h5file["effective_dissipation_rate"])
    return np.squeeze(effective_dissipation_rate, axis=2)


def calculate_fdisper(target_folder_name):

    effective_dissipation_rate = get_effective_dissipation_rate(target_folder_name)
    file = f"{target_folder_name}/domain/data_*.h5"
    highorder_dissipation_rate = get_highorder_dissipation_rate(file)
    adi_to_ref = - (effective_dissipation_rate - highorder_dissipation_rate)
    epsilon_sample = np.where(adi_to_ref < 0, adi_to_ref, 0).sum()

    epsilon_baseline = -1.4803669984900289e-05 # read from baseline folder
    
    fdisper = epsilon_sample / epsilon_baseline - 1
    return fdisper