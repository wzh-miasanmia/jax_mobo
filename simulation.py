import json
import os
import random
import time
import math
import pandas as pd
from tgv.tgv_postprocessing import plot_tke, calculate_ftke, calculate_fcons
from imp.imp_postprocessing import calculate_fdisper, get_file_list
import numpy as np
import subprocess

def write_parameters_to_json(numerical_file, parameters, scheme: str, scheme_setup):
    r"""
    Write the parameters to numerical_setup.json
    """
    with open(numerical_file, "r") as read_file:
        data = json.load(read_file)
        data["scheme_parameters"][scheme.lower()][scheme_setup[scheme.lower()]["para1_name"]] = parameters[0]
        data["scheme_parameters"][scheme.lower()][scheme_setup[scheme.lower()]["para2_name"]] = parameters[1]
        data["scheme_parameters"][scheme.lower()][scheme_setup[scheme.lower()]["para3_name"]] = parameters[2]
    with open(numerical_file, "w") as write_file:
        json.dump(data, write_file, indent=4)

def run_simulation_tgv(DEBUG=False):
    work_dir = "tgv"
    command = "python run_tgv.py"

    if DEBUG:
        print(f"DEBUG: {command}")
    else:
        subprocess.run(command, shell=True, cwd=work_dir)

def run_simulation_imp(DEBUG=False):
    work_dir = "imp"
    command = "python run_fullimplosion.py"

    if DEBUG:
        print(f"DEBUG: {command}")
    else:
        subprocess.run(command, shell=True, cwd=work_dir)

def rename_simulation_folder(target:str, origin: str, DEBUG=False):
    os.system(f"mv {origin} {target}") if not DEBUG else print(f"mv {origin} {target}")

def fill_runtime_data(data_dict: dict, paras, fdisper, ftke, fcons):
    r"""
    Fill the runtime data to data_dict for saving to csv
    """
    data_dict["eta_eno"].append(paras[0])
    data_dict["eta_v"].append(paras[1])
    data_dict["ducros_cutoff"].append(paras[2])
    data_dict["fdisper"].append(fdisper)
    data_dict["ftke"].append(ftke)
    data_dict["fcons"].append(fcons)
    data_dict["time"].append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))   

def select_para_and_simulation(para, SCHEME="teno6_dv", scheme_setup=None, DEBUG=False):
    '''
    The input paras are first updated to a json file and then the corresponding y(objective function) is calculated and output.
    simulate implosion to calculate fdisper
    simulate tgv to calculate ftke and fcons
    '''
    # create a dictionary to store the runtime data
    runtime_data = {"eta_eno": [], "eta_v": [], "ducros_cutoff": [], "fdisper": [], "ftke": [], "fcons": [], "time": []}

    ## Part 1: Implosion ##
    # simulation
    imp_numerical_file =  "imp/numerical_setup.json"
    write_parameters_to_json(imp_numerical_file, para, SCHEME, scheme_setup)
    run_simulation_imp(DEBUG=DEBUG)

    # rename the simulation folder from imp to imp_{para1}_{para2}_{para3}
    origin_folder_name = "imp/results/fullimplosion"
    target_folder_name = f"imp/results/fullimplosion_{str(para[0])}_{str(para[1])}_{str(para[2])}"
    rename_simulation_folder(target_folder_name, origin_folder_name, DEBUG)

    # output the object function: fdisper
    test_file = get_file_list(f"{target_folder_name}/domain/data_*.h5")
    if test_file.endswith('nan.h5'):
        fdisper = 9
    else:
        fdisper = calculate_fdisper(target_folder_name) if not DEBUG else random.random()
        fdisper = np.round(fdisper, 4)
        


    ## Part 2: TGV ## 
    # simulation
    tgv_numerical_file =  "tgv/numerical_setup.json"
    write_parameters_to_json(tgv_numerical_file, para, SCHEME, scheme_setup)
    run_simulation_tgv(DEBUG=DEBUG)

    # rename the simulation folder from tgv to tgv_{para1}_{para2}_{para3}
    origin_folder_name = "tgv/results/tgv"
    target_folder_name = f"tgv/results/tgv_{str(para[0])}_{str(para[1])}_{str(para[2])}"
    rename_simulation_folder(target_folder_name, origin_folder_name, DEBUG)

    # output the object function: ftke
    ftke = calculate_ftke(f"{target_folder_name}/domain/data_*.h5") if not DEBUG else random.random()
    ftke = np.round(ftke, 4)
    if math.isnan(ftke):
        ftke = 4
    
    # output the object function: fcons
    fcons = calculate_fcons(f"{target_folder_name}/domain/data_*.h5") if not DEBUG else random.random()
    fcons = np.round(fcons, 4)
    if math.isnan(fcons):
        fcons = 1
    
    ## Part 3: Save ## 
    # fill the runtime data to data_dict for saving the runtime data to csv
    fill_runtime_data(runtime_data, para, fdisper, ftke, fcons)
    df = pd.DataFrame(runtime_data)
    with open("results.csv", "a") as f:
        df.to_csv(f, header=f.tell()==0, index=False)

    # turn minimum to maximum and round
    return np.array([-fdisper, -ftke, -fcons])



if "__main__" in __name__:
    import jax.numpy as jnp
    parameters = jnp.array([0.9713,0.4751,0.473])
    scheme_setup = {
        "teno6_dv": {
            "para1_name": "eta_eno",
            "para1_range": [0, 1],
            "para2_name": "eta_v",
            "para2_range": [0, 1],
            "para3_name": "ducros_cutoff",
            "para3_range": [0, 1],
        },
    }
    a = select_para_and_simulation(parameters, "teno6_dv", scheme_setup)
