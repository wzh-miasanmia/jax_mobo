from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_rl.rl_simulation_manger import ReinforcementLearninngSimulationManager

# SETUP SIMULATION
input_manager = InputManager("tgv.json", "numerical_setup.json")
initialization_manager  = InitializationManager(input_manager)
sim_manager  = ReinforcementLearninngSimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

