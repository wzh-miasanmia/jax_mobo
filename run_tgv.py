from jaxfluids import InputManager, InitializationManager, SimulationManager

# SETUP SIMULATION
input_manager = InputManager("tgv.json", "numerical_setup.json")
initialization_manager  = InitializationManager(input_manager)
sim_manager  = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

