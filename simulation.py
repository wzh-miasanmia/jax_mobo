import json
import os
import random
import time
import pandas as pd
from tgv_postprocessing import plot_tke, calculate_ftke
import numpy as np

def write_parameters_to_json(parameters, scheme: str, scheme_setup):
    r"""
    Write the Cq and q parameters to numerical_setup.json
    """
    with open("numerical_setup.json", "r") as read_file:
        data = json.load(read_file)
        data["scheme_parameters"][scheme.lower()][scheme_setup[scheme.lower()]["para1_name"]] = int(parameters[0])
        data["scheme_parameters"][scheme.lower()][scheme_setup[scheme.lower()]["para2_name"]] = int(parameters[1])
    with open("numerical_setup.json", "w") as write_file:
        json.dump(data, write_file, indent=4)

def run_simulation(DEBUG=False):
    os.system("python run_tgv.py") if not DEBUG else print("python3 run_tgv.py")

def rename_simulation_folder(target:str, origin: str = "results/tgv", DEBUG=False):
    os.system(f"mv {origin} {target}") if not DEBUG else print(f"mv {origin} {target}")

def fill_runtime_data(data_dict: dict, paras, ftke):
    r"""
    Fill the runtime data to data_dict for saving to csv
    """
    data_dict["para1"].append(paras[0])
    data_dict["para2"].append(paras[1])
    data_dict["ftke"].append(ftke)
    data_dict["time"].append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))   

def select_para_and_simulation(para, N_SAMPLES=5, SCHEME="teno6", scheme_setup=None, DEBUG=False):
    '''
    The input paras are first updated to a json file and then the corresponding y(objective function) is calculated and output.
    '''
    # create a dictionary to store the runtime data
    runtime_data = {"para1": [], "para2": [], "ftke": [], "time": []}

    write_parameters_to_json(para, SCHEME, scheme_setup)
    run_simulation(DEBUG=DEBUG)

    # rename the simulation folder from tgv to tgv_{para1}_{para2}
    origin_folder_name = "results/tgv"
    target_folder_name = f"results/tgv_{str(int(para[0]))}_{str(int(para[1]))}"
    rename_simulation_folder(target_folder_name, origin_folder_name, DEBUG)

    # output the object function: ftke
    ftke = calculate_ftke(f"{target_folder_name}/domain/data_*.h5") if not DEBUG else random.random()
    # fill the runtime data to data_dict for saving the runtime data to csv
    fill_runtime_data(runtime_data, para, ftke)
    df = pd.DataFrame(runtime_data)
    with open("ftke.csv", "a") as f:
        df.to_csv(f, header=f.tell()==0, index=False)
    return np.round(-ftke, 4) # turn minimum to maximum and round

if "__main__" in __name__:
    import jax.numpy as jnp
    parameters = jnp.array([1,1])
    write_parameters_to_json(parameters, "teno6", scheme_setup)
