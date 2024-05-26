import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from FileHelper import FileHelper
from tqdm import tqdm
import os


class Visualizer:
    """
    Class for visualizing the results of a particle-in-cell simulation.
    
    # Attributes:
        - `directory_name (str)`: Name of the directory where the simulation results are saved.
        - `position_filepath (str)`: Path to the file containing the particle positions.
        - `velocity_filepath (str)`: Path to the file containing the particle velocities.
        - `electric_field_filepath (str)`: Path to the file containing the electric field values.
        - `potential_filepath (str)`: Path to the file containing the potential values.
        - `followed_particle_filepath (str)`: Path to the file containing the followed particle data.
        - `followed_cell_filepath (str)`: Path to the file containing the followed cell data.
        - `time (numpy.ndarray)`: Array containing the time values.
        - `iteration_save_rate (int)`: Number of iterations between each saved step.

    # Methods:
        - `plot_phase_space(iteration_number=-1)`: Plot the phase space of the particles at a specific iteration.
        - `plot_phase_space_foreach_saved_step()`: Plot the phase space of the particles for each saved step.
        - `plot_electric_field(iteration_number=-1)`: Plot the electric field at a specific iteration.
        - `plot_electric_field_foreach_saved_step()`: Plot the electric field for each saved step.
        - `plot_potential(iteration_number=-1)`: Plot the potential at a specific iteration.
        - `plot_potential_foreach_saved_step()`: Plot the potential for each saved step.
        - `plot_single_particle_position()`: Plot the position of a followed particle over time.
        - `plot_single_particle_phase_space()`: Plot the phase space of a single followed particle.
        - `plot_potential_for_single_cell()`: Plot the potential for a single cell over time.
        - `plot_electric_field_fot_single_cell()`: Plot the electric field for a single cell over time.

    """

    def __init__(self, iteration_save_rate, position_filename='positions.csv', velocity_filename='velocities.csv'):
        self.directory_name = FileHelper.get_test_directory_name()
        self.position_filepath = os.path.join(self.directory_name, 'OutputFiles', position_filename)
        self.velocity_filepath = os.path.join(self.directory_name, 'OutputFiles', velocity_filename)
        self.electric_field_filepath = os.path.join(self.directory_name, 'OutputFiles', 'electric_field.csv')
        self.potential_filepath = os.path.join(self.directory_name, 'OutputFiles', 'potential.csv')
        self.followed_particle_filepath = os.path.join(self.directory_name, 'OutputFiles', 'followed_particle.csv')
        self.followed_cell_filepath = os.path.join(self.directory_name, 'OutputFiles', 'followed_cell.csv')
        dt_filepath = os.path.join(self.directory_name, 'OutputFiles', 'dt.csv')
        dt = pd.read_csv(dt_filepath).iloc[:, 0].to_numpy()
        self.time = np.cumsum(dt)
        self.iteration_save_rate = iteration_save_rate

    def plot_phase_space(self, iteration_number=-1):
        """
        Plot the phase space of the particles at a specific iteration.

        # Args:
            - `iteration_number` (int, optional): The iteration number at which the phase space should be plotted. Defaults to -1.
                 
        # Remarks:
            If `iteration_number` is -1, the phase space for the last iteration is plotted.
        """
        if iteration_number != -1:
            index = iteration_number // self.iteration_save_rate
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
        ax[1].set_title(f'Phase Space for {iteration_number} th iteration')
        ax[0].set_xlabel('Position')
        ax[0].set_ylabel('Velocity')
        ax[1].set_xlabel('Position')
        ax[1].set_ylabel('Velocity')

        plot_filename = os.path.join(self.directory_name, 'Plots', 'Phase_Space', f'phase_space_{iteration_number}.png')

        plt.savefig(plot_filename)
        plt.close()

    def plot_phase_space_foreach_saved_step(self):
        """
        Plot the phase space of the particles for each saved step according to the `iteration_save_rate`.
        """
        with open(self.position_filepath) as f:
            line_number = sum(1 for _ in f)
        for i in tqdm(range(1, line_number - 1), desc='Plotting Phase Space', unit='steps'):
            Visualizer.plot_phase_space(self, i * self.iteration_save_rate)

    def plot_electric_field(self, iteration_number=-1):
        """
        Plot the electric field at a specific iteration.

        # Args:
            -`iteration_number` (int, optional): The iteration number at which the electric field
                should be plotted. Defaults to -1.
                
        # Remarks:
            If `iteration_number` is -1, the electric field for the last iteration is plotted.
        """
        if iteration_number != -1:
            index = iteration_number // self.iteration_save_rate
        else:
            index = -1

        electric_field_df = pd.read_csv(self.electric_field_filepath)

        electric_field = electric_field_df.iloc[index]
        x = electric_field.index

        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=x, y=electric_field, ax=ax, color='purple')
        ax.set_title('Electric Field')
        ax.set_xlabel('Cell Index')
        ax.set_ylabel('Electric Field')
        ax.set_xticks(np.arange(0, len(x), 20))
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Global', 'Electric_Field',
                                 f'electric_field_{iteration_number}.png'))
        plt.close()

    def plot_electric_field_foreach_saved_step(self):
        """
        Plot the electric field for each saved step according to the `iteration_save_rate`.
        """
        with open(self.electric_field_filepath) as f:
            line_number = sum(1 for _ in f)
        for i in tqdm(range(1, line_number - 1), desc='Plotting Electric Field', unit='steps'):
            Visualizer.plot_electric_field(self, i * self.iteration_save_rate)

    def plot_potential(self, iteration_number=-1):
        """
        Plot the potential at a specific iteration.

        # Args:
            - `iteration_number` (int, optional): The iteration number at which the potential
                should be plotted. Defaults to -1.
        """
        if iteration_number != -1:
            index = iteration_number // self.iteration_save_rate
        else:
            index = -1

        potential_df = pd.read_csv(self.potential_filepath)

        potential = potential_df.iloc[index]
        x = potential.index

        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=x, y=potential, ax=ax, color='purple')
        ax.set_title('Potential')
        ax.set_xlabel('Cell Index')
        ax.set_ylabel('Potential')
        ax.set_xticks(np.arange(0, len(x), 20))
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Global', 'Potential',
                                 f'potential_{iteration_number}.png'))
        plt.close()

    def plot_potential_foreach_saved_step(self):
        """
        Plot the potential for each saved step according to the `iteration_save_rate`.
        """
        with open(self.potential_filepath) as f:
            line_number = sum(1 for _ in f)
        for i in tqdm(range(1, line_number - 1), desc='Plotting Potential', unit='steps'):
            Visualizer.plot_potential(self, i * self.iteration_save_rate)

    def plot_single_particle_position(self):
        """
        Plot the position of a followed particle over time.
        """
        particle_df = pd.read_csv(self.followed_particle_filepath)
        particle_position = particle_df['Position']

        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=self.time, y=particle_position, ax=ax, color='purple')
        ax.set_title('Particle Path')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Particle', 'single_particle_position.png'))

    def plot_single_particle_phase_space(self):
        """
        Plot the phase space of a single followed particle.
        """
        particle_df = pd.read_csv(self.followed_particle_filepath)
        particle_position = particle_df['Position']
        particle_velocity = particle_df['Velocity']

        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.scatterplot(x=particle_position, y=particle_velocity, ax=ax, color='purple', s=1)
        ax.set_title('Single Particle Phase Space')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Particle', 'single_particle_phase_space.png'))

    def plot_potential_for_single_cell(self):
        """
        Plot the potential for a single cell over time.
        """
        cell_df = pd.read_csv(self.followed_cell_filepath)
        potential = cell_df['Potential']

        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=self.time, y=potential, ax=ax, color='purple')
        ax.set_title('Potential for Single Cell')
        ax.set_xlabel('Time')
        ax.set_ylabel('Potential')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Cell', 'single_cell_potential.png'))

    def plot_electric_field_for_single_cell(self):
        """
        Plot the electric field for a single cell over time.
        """
        cell_df = pd.read_csv(self.followed_cell_filepath)
        electric_field = cell_df['Electric Field']

        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=self.time, y=electric_field, ax=ax, color='purple')
        ax.set_title('Electric Field for Single Cell')
        ax.set_xlabel('Time')
        ax.set_ylabel('Electric Field')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Cell', 'single_cell_electric.png'))

    def plot_particles_density(self):
        """
        Plot the density of the particles over time.
        """
        position = pd.read_csv(self.position_filepath).iloc[-1]
        potential = pd.read_csv(self.potential_filepath).iloc[-1]

        sns.set_theme()
        _, ax1 = plt.subplots(figsize=(12, 6))

        sns.histplot(position, binwidth=5, ax=ax1, color='green')
        ax1.set_title('Particles Density')
        ax1.set_xlabel('Position', color='green')
        ax1.set_ylabel('Density', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.tick_params(axis='x', labelcolor='green')
        ax1.grid(False)

        ax2 = ax1.twinx().twiny()

        sns.lineplot(x=potential.index, y=potential, ax=ax2, color='purple')
        ax2.set_xticks(np.arange(0, len(potential), 20))
        ax2.set_xlabel('Cell Index ', color='purple')
        ax2.set_ylabel('Potential', color='purple')
        ax2.tick_params(axis='x', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_ticks_position('right')
        ax2.yaxis.set_label_position('right')
        ax2.grid(False)
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Global', 'particles_density.png'))