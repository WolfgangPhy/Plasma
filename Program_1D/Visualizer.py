import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from FileHelper import FileHelper
import os

class Visualizer:
    """
    Class for visualizing the results of a particle-in-cell simulation.

    # Methods:
        - `plot_phase_space()`: Plots the phase space using the last time step of the simulation
            results. (Static Method)
        - `animate_results(file_path='results.npz')`: Animates the simulation results stored
            in 'results.npz'. (Static Method)

    """
    
    def __init__(self, position_filename='positions.csv', velocity_filename='velocities.csv'):
        self.position_filename = position_filename
        self.velocity_filename = velocity_filename
        self.directory_name = FileHelper.get_test_directory_name()
        self.position_filepath = os.path.join(self.directory_name, 'OutputFiles', self.position_filename)
        self.velocity_filepath = os.path.join(self.directory_name, 'OutputFiles', self.velocity_filename)
    
    def plot_phase_space(self, iteration_number=-1):
        if iteration_number != -1:
            index = iteration_number // 500
        else:
            index = -1
        
        positions_df = pd.read_csv(self.position_filepath)
        velocities_df = pd.read_csv(self.velocity_filepath)
        
        index_positions = positions_df.iloc[index]
        index_velocities = velocities_df.iloc[index]
        last_data = pd.DataFrame({'Position': index_positions, 'Velocity': index_velocities})
        
        first_position = positions_df.iloc[0]
        first_velocity = velocities_df.iloc[0]
        first_data = pd.DataFrame({'Position': first_position, 'Velocity': first_velocity})

        half_length = len(last_data) // 2
        last_first_half_df = last_data.iloc[:half_length]
        last_second_half_df = last_data.iloc[half_length:]
        first_first_half_df = first_data.iloc[:half_length]
        first_second_half_df = first_data.iloc[half_length:]

        sns.set_theme()
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.scatterplot(x='Position', y='Velocity', data=first_first_half_df, s=1, color='green', ax=ax[0])
        sns.scatterplot(x='Position', y='Velocity', data=first_second_half_df, s=1, color='purple', ax=ax[0])
        sns.scatterplot(x='Position', y='Velocity', data=last_first_half_df, s=1, color='green', ax=ax[1])
        sns.scatterplot(x='Position', y='Velocity', data=last_second_half_df, s=1, color='purple', ax=ax[1])
        
        ax[0].set_title('Initial Phase Space')
        ax[1].set_title('Final Phase Space')
        ax[0].set_xlabel('Position')
        ax[0].set_ylabel('Velocity')
        ax[1].set_xlabel('Position')
        ax[1].set_ylabel('Velocity')
        
        plot_filename = os.path.join(self.directory_name, 'Plots', f'PhaseSpace_{iteration_number}.png')
        
        plt.savefig(plot_filename)
        
    def plot_phase_space_foreach_saved_step(self):
        with open(self.position_filepath) as f:
            line_nmber = sum(1 for _ in f)
        for i in range(1, line_nmber - 1):
            Visualizer.plot_phase_space(self, i*500)
            
            