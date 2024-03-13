import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

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
        data = np.load('results.npz')
        positions_array = data['positions']
        velocities_array = data['velocities']
        positions = positions_array[-1, :]
        velocities = velocities_array[-1, :]

        # Create a scatter plot
        _, ax = plt.subplots()
        sns.scatterplot(positions, velocities, ax=ax)

        # Set plot labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Phase Space')

        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)

        # Show the plot
        plt.show()
        
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

        # Show the animation
        plt.show()
