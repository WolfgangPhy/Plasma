import os
import json


class FileHelper:
    """
    Class for handling file operations.
    """

    @staticmethod
    def create_test_directory():
        """
        Creates a filename for the plot based on the parameters in the 'Parameters.json' file.
        
        # Returns:
            str: The filename for the plot.
        """
        test_directory = FileHelper.get_test_directory_name()

        if not os.path.exists(test_directory):
            os.makedirs(test_directory)
        elif input("Directory already exists. Do you want to do calculations in this directory? (y/n): ") == "y":
            return test_directory
        else:
            raise ValueError("Directory already exists")

        output_files_directory = os.path.join(test_directory, 'OutputFiles')
        plots_directory = os.path.join(test_directory, 'Plots')
        phase_space_plots_directory = os.path.join(plots_directory, 'Phase_Space')
        followed_particle_filepath = os.path.join(plots_directory, 'Followed_Particle')
        followed_cell_filepath = os.path.join(plots_directory, 'Followed_Cell')
        global_plots_directory = os.path.join(plots_directory, 'Global')
        global_electric_field_directory = os.path.join(global_plots_directory, 'Electric_Field')
        global_potential_directory = os.path.join(global_plots_directory, 'Potential')

        os.makedirs(output_files_directory)
        os.makedirs(plots_directory)
        os.makedirs(phase_space_plots_directory)
        os.makedirs(followed_particle_filepath)
        os.makedirs(followed_cell_filepath)
        os.makedirs(global_plots_directory)
        os.makedirs(global_electric_field_directory)
        os.makedirs(global_potential_directory)

        return test_directory

    @staticmethod
    def get_test_directory_name():

        with open('Parameters.json', 'r') as file:
            parameters = json.load(file)

        thousands_particles = parameters['particles_number'] // 1000
        test_directory = f"PN_{thousands_particles}k_"
        test_directory += f"DS_{parameters['domain_size']}_"
        test_directory += f"CN_{parameters['cells_number']}_"
        test_directory += f"V_{parameters['initial_velocity']}_"
        test_directory += f"PC_{parameters['particle_charge']}_"
        test_directory += f"PM_{parameters['particle_mass']}_"
        if parameters['velocity_repartition'] == 'random':
            test_directory += f"VR_R"
        elif parameters['velocity_repartition'] == 'normal':
            test_directory += f"VR_N"
        elif parameters['velocity_repartition'] == '2streams':
            test_directory += f"VR_2S"

        test_directory = test_directory.replace('.', '-')
        test_directory = os.path.join('Tests', test_directory)

        return test_directory
