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

    # Methods:
        - `plot_phase_space()`: Plots the phase space using the last time step of the simulation
            results. (Static Method)
        - `animate_results(file_path='results.npz')`: Animates the simulation results stored
            in 'results.npz'. (Static Method)

    """
    
    def __init__(self, iteration_save_rate, position_filename='positions.csv', velocity_filename='velocities.csv'):
        self.position_filename = position_filename
        self.velocity_filename = velocity_filename
        self.directory_name = FileHelper.get_test_directory_name()
        self.position_filepath = os.path.join(self.directory_name, 'OutputFiles', self.position_filename)
        self.velocity_filepath = os.path.join(self.directory_name, 'OutputFiles', self.velocity_filename)
        self.electric_field_filepath = os.path.join(self.directory_name, 'OutputFiles', 'electric_field.csv')
        self.potential_filepath = os.path.join(self.directory_name, 'OutputFiles', 'potential.csv')
        self.followed_particle_filepath = os.path.join(self.directory_name, 'OutputFiles', 'followed_particle.csv')
        self.followed_cell_filepath = os.path.join(self.directory_name, 'OutputFiles', 'followed_cell.csv')
        self.dt_filepath = os.path.join(self.directory_name, 'OutputFiles', 'dt.csv')
        self.iteration_save_rate = iteration_save_rate
    
    def plot_phase_space(self, iteration_number=-1):
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
        
        plot_filename = os.path.join(self.directory_name, 'Plots', 'Phase_Space', f'PhaseSpace_{iteration_number}.png')
        
        plt.savefig(plot_filename)
        plt.close()
        
    def plot_phase_space_foreach_saved_step(self):
        with open(self.position_filepath) as f:
            line_nmber = sum(1 for _ in f)
        for i in tqdm(range(1, line_nmber - 1), desc='Plotting Phase Space', unit='steps'):
            Visualizer.plot_phase_space(self, i*self.iteration_save_rate)
            
    def plot_electric_field(self, iteration_number=-1):
        if iteration_number != -1:
            index = iteration_number // self.iteration_save_rate
        else:
            index = -1
            
        electric_field_df = pd.read_csv(self.electric_field_filepath)
        
        electric_field = electric_field_df.iloc[index]
        x = electric_field.index
        
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=x, y=electric_field, ax=ax,color='purple')
        ax.set_title('Electric Field')
        ax.set_xlabel('Position')
        ax.set_ylabel('Electric Field')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Global', 'Electric_Field', f'ElectricField_{iteration_number}.png'))
        plt.close()
        
    def plot_electric_field_foreach_saved_step(self):
        with open(self.electric_field_filepath) as f:
            line_nmber = sum(1 for _ in f)
        for i in tqdm(range(1, line_nmber - 1), desc='Plotting Electric Field', unit='steps'):
            Visualizer.plot_electric_field(self, i*self.iteration_save_rate)
            
    def plot_potential(self, iteration_number=-1):
        if iteration_number != -1:
            index = iteration_number // self.iteration_save_rate
        else:
            index = -1
            
        potential_df = pd.read_csv(self.potential_filepath)
        
        potential = potential_df.iloc[index]
        x = potential.index
        
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=x, y=potential, ax=ax,color='purple')
        ax.set_title('Potential')
        ax.set_xlabel('Position')
        ax.set_ylabel('Potential')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Global', 'Potential', f'Potential_{iteration_number}.png'))
        plt.close()
        
    def plot_potential_foreach_saved_step(self):
        with open(self.potential_filepath) as f:
            line_nmber = sum(1 for _ in f)
        for i in tqdm(range(1, line_nmber - 1), desc='Plotting Potential', unit='steps'):
            Visualizer.plot_potential(self, i*self.iteration_save_rate)
            
    def plot_particule_path(self):
        particule_df = pd.read_csv(self.followed_particle_filepath)
        particule_position = particule_df['Position']
        dt = pd.read_csv(self.dt_filepath).iloc[:,0].to_numpy()
        time = np.cumsum(dt)
        
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=time, y=particule_position, ax=ax,color='purple')
        ax.set_title('Particule Path')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Particle', 'ParticulePath.png'))
        
    def plot_single_particle_phase_space(self):
        particule_df = pd.read_csv(self.followed_particle_filepath)
        particule_position = particule_df['Position']
        particule_velocity = particule_df['Velocity']
        
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.scatterplot(x=particule_position, y=particule_velocity, ax=ax, color='purple', s=1)
        ax.set_title('Single Particle Phase Space')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Particle', 'SingleParticlePhaseSpace.png'))
        
    def plot_potential_for_single_cell(self):
        cell_df = pd.read_csv(self.followed_cell_filepath)
        potential = cell_df['Potential']
        dt = pd.read_csv(self.dt_filepath).iloc[:,0].to_numpy()
        time = np.cumsum(dt)
        
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=time, y=potential, ax=ax,color='purple')
        ax.set_title('Potential for Single Cell')
        ax.set_xlabel('Time')
        ax.set_ylabel('Potential')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Cell', 'PotentialForSingleCell.png'))
        
    def plot_electric_field_fot_single_cell(self):
        cell_df = pd.read_csv(self.followed_cell_filepath)
        electric_field = cell_df['Electric Field']
        dt = pd.read_csv(self.dt_filepath).iloc[:,0].to_numpy()
        time = np.cumsum(dt)
        
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.lineplot(x=time, y=electric_field, ax=ax,color='purple')
        ax.set_title('Electric Field for Single Cell')
        ax.set_xlabel('Time')
        ax.set_ylabel('Electric Field')
        plt.savefig(os.path.join(self.directory_name, 'Plots', 'Followed_Cell', 'ElectricFieldForSingleCell.png'))
                