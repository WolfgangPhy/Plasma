import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import pandas as pd

class Visualizer:
    """
    Class for visualizing the results of a particle-in-cell simulation.

    # Methods:
        - `plot_phase_space()`: Plots the phase space using the last time step of the simulation results. (Static Method)
        - `animate_results(file_path='results.npz')`: Animates the simulation results stored in 'results.npz'. (Static Method)

    """
       
    @staticmethod
    def plot_phase_space():
        """
        Plots the phase space using the last time step of the simulation results.
        
        """
        # Load data from npz file
        positions_df = pd.read_csv('./OutputFiles/positions.csv')
        velocities_df = pd.read_csv('./OutputFiles/velocities.csv')
        last_positions = positions_df.iloc[500]
        last_velocities = velocities_df.iloc[500]
        last_data = pd.DataFrame({'Position': last_positions, 'Velocity': last_velocities})
        
        half_length = len(last_data) // 2
        first_half_df = last_data.iloc[:half_length]
        second_half_df = last_data.iloc[half_length:]
        

        sns.set_theme()
        sns.scatterplot(x='Position', y='Velocity', data=first_half_df, s=1, color='green')
        sns.scatterplot(x='Position', y='Velocity', data=second_half_df, s=1, color='purple')

        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Phase space')

        # Show the plot
        plt.savefig('Plots/phase_space.png')

        
    @staticmethod
    def animate_results(file_path='results.npz'):
        """
        Animates the simulation results stored in 'results.npz'.

        # Args:
            - `file_path (str, optionnal)`: Path to the file containing simulation results. Default is 'results.npz'.

        """
        data = np.load(file_path)
        positions = data['positions']
        velocities = data['velocities']
        dt = data['dt']

        # Initialize the figure
        fig, ax = plt.subplots()
        scatter, = ax.plot([], [], 'ro', markersize=1)

        # Set axis limits

        # Initialization function for the animation
        def init():
            ax.set_xlim(0., 1.)
            ax.set_ylim(-1, 1)
            return scatter,

        # Animation function
        def update(frame):
            x = positions[frame, :]
            y = velocities[frame, :]
            scatter.set_data(x, y)
            return scatter,

        # Create the animation
        num_frames = len(dt)
        animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
        animation.save('Plots/animation.gif', writer='PillowWriter', fps=120)
