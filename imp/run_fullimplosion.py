from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure
from jaxfluids_rl.rl_simulation_manger import ReinforcementLearninngSimulationManager

# SETUP SIMULATION
input_manager = InputManager("fullimplosion.json", "numerical_setup.json")
initialization_manager  = InitializationManager(input_manager)
sim_manager  = ReinforcementLearninngSimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)
print(times)
# PLOT
nrows_ncols = (1,1)
plot_dict = {
    "density": data_dict["density"]}
# create_2D_animation(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols, plane="xy", interval=100)
create_2D_figure(plot_dict, nrows_ncols, cell_centers=cell_centers, plane="xy", plane_value=0.0, save_fig="full_implosion.png")