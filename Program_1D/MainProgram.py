import json
from Simulation import Simulation
from Visualizer import Visualizer
from FileHelper import FileHelper


class MainProgram:
    """
    Main class to run the Particle-in-Cell simulation of a 1D plasma and visualize the results.

    # Attributes:
        - `parameters (dict)`: Dictionary containing simulation parameters loaded from 'Parameters.json'.

    # Methods:
        - `set_parameters()`: Loads simulation parameters from 'Parameters.json'.
        - `run()`: Runs the simulation and visualizes the results using the Simulation and Visualizer classes.
    """
    
    def __init__(self):
        """
        Initializes the MainProgram instance and loads simulation parameters.
        """
        self.parameters = None
        self.set_parameters()
        self.directory_name = FileHelper.create_test_directory()

    def set_parameters(self):
        """
        Loads simulation parameters from 'Parameters.json'.
        """
        with open('Parameters.json', 'r') as file:
            self.parameters = json.load(file)

    def run(self):
        """
        Runs the simulation and visualizes the results using the Simulation and Visualizer classes.
        """
        sim = Simulation(self.parameters, self.directory_name)
        sim.run()
        visualizer = Visualizer(self.parameters['iteration_save_rate'])
        visualizer.plot_phase_space_foreach_saved_step()
        visualizer.plot_electric_field_foreach_saved_step()
        visualizer.plot_potential_foreach_saved_step()
        visualizer.plot_particule_path()
        visualizer.plot_single_particle_phase_space()
        visualizer.plot_potential_for_single_cell()
        visualizer.plot_electric_field_fot_single_cell()


if __name__ == '__main__':
    main_program = MainProgram()
    main_program.run()
