import os
import json

class FileHelper:
    
    @staticmethod
    def create_plot_filename():
        with open('Parameters.json', 'r') as file:
            parameters = json.load(file)
        
        
        thousands_iterations = parameters['iterations_number'] // 1000
        thousands_particles = parameters['particles_number'] // 1000
        plot_filename = 'Plots/PS_'
        plot_filename += f"PN_{thousands_particles}k_"
        plot_filename += f"DS_{parameters['domain_size']}_"
        plot_filename += f"CN_{parameters['cells_number']}_"
        plot_filename += f"V_{parameters['initial_velocity']}_"
        if(parameters['velocity_repartition'] == 'random'):
            plot_filename += f"VR_R"
        elif(parameters['velocity_repartition'] == 'normal'):
            plot_filename += f"VR_N"
        elif(parameters['velocity_repartition'] == '2streams'):
            plot_filename += f"VR_2S"
        plot_filename += f"IN_{thousands_iterations}k_"
        plot_filename = plot_filename.replace('.', '-')
        return plot_filename