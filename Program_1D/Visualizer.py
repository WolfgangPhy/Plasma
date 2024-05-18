import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from FileHelper import FileHelper

class Visualizer:
    """
    Class for visualizing the results of a particle-in-cell simulation.

    # Methods:
        - `plot_phase_space()`: Plots the phase space using the last time step of the simulation
            results. (Static Method)
        - `animate_results(file_path='results.npz')`: Animates the simulation results stored
            in 'results.npz'. (Static Method)

    """

    @staticmethod
    def plot_phase_space():
        """
        Plots the phase space using the last time step of the simulation results.
        
        """
        # Load data from npz file
        positions_df = pd.read_csv('./OutputFiles/positions.csv')
        velocities_df = pd.read_csv('./OutputFiles/velocities.csv')
        last_positions = positions_df.iloc[-1]
        last_velocities = velocities_df.iloc[-1]
        last_data = pd.DataFrame({'Position': last_positions, 'Velocity': last_velocities})
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
        
        plt.savefig(FileHelper.create_plot_filename())