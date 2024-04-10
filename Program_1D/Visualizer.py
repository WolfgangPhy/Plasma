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
        positions_df = pd.read_csv('positions.csv')
        velocities_df = pd.read_csv('velocities.csv')
        # Convert the 500th row of the dataframes to numpy arrays
        positions = positions_df.iloc[500].to_numpy()
        velocities = velocities_df.iloc[500].to_numpy()
        # Determine the number of points
        num_points = len(positions)

        # Determine the indices for splitting the points into two halves
        midpoint_index = num_points // 2

        # Split positions and velocities into two halves
        first_half_positions = positions[:midpoint_index]
        second_half_positions = positions[midpoint_index:]
        first_half_velocities = velocities[:midpoint_index]
        second_half_velocities = velocities[midpoint_index:]

        # Plot the first half of points in blue
        plt.scatter(first_half_positions, first_half_velocities, s=1, color='blue')

        # Plot the second half of points in orange
        plt.scatter(second_half_positions, second_half_velocities, s=1, color='orange')

        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Phase space')

        # Show the plot
        plt.savefig('phase_space.png')

        
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
        animation.save('animation.gif', writer='PillowWriter', fps=120)
