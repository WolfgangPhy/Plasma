import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from scipy.constants import epsilon_0


class Simulation:
    """
    Class for simulating a 1D plasma using the Particle-in-Cell method.

    # Args:
        - `parameters (dict)`: Dictionary containing simulation parameters.
        - `directory_name (str)`: Name of the directory where the simulation results will be saved.

    # Attributes:
        - `dx (float)`: Spatial resolution.
        - `EPSILON_0 (float)`: Permittivity of free space.
        - `FACTOR (float)`: Factor used in the calculation of potential.
        - `particles_number (int)`: Number of particles in the simulation.
        - `particle_charge (float)`: Charge of each particle.
        - `particle_mass (float)`: Mass of each particle.
        - `domain_size (float)`: Size of the simulation domain.
        - `cells_number (int)`: Number of cells in the simulation domain.
        - `max_initial_velocity_deviation (float)`: Maximum deviation in initial velocities.
        - `iterations_number (int)`: Number of simulation iterations.
        - `potential_matrix (numpy.ndarray)`: Matrix used in the calculation of potential.
        - `positions (numpy.ndarray)`: Array containing particle positions for each iteration.
        - `velocities (numpy.ndarray)`: Array containing particle velocities for each iteration.
        - `dt (numpy.ndarray)`: Array containing time steps for each iteration.

    # Methods:
        - `set_initial_conditions()`: Set initial conditions for the simulation.
        - `init_arrays()`: Initialize arrays and matrices for the simulation.
        - `compute_charge_density(iteration)`: Compute charge density at a given iteration.
        - `compute_potential(density)`: Compute potential based on charge density.
        - `compute_electric_field(potential)`: Compute electric field based on potential.
        - `compute_particle_electric_field(electric_field, iteration)`: Compute particle electric field.
        - `compute_force(particle_electric_field)`: Compute force on particles.
        - `compute_time_step(iteration)`: Compute time step based on particle velocities.
        - `update_positions_velocities(force, iteration)`: Update particle positions and velocities using Euler method.
        - `save_results()`: Save simulation results to 'results.npz'.
        - `run()`: Run the simulation.

    """

    def __init__(self, parameters, directory_name):
        self.directory_name = directory_name
        self.parameters = parameters
        
        self.cells_number = None
        self.domain_size = None
        self.dt = None
        self.electric_field = None
        self.followed_cell_index = None
        self.followed_particle_index = None
        self.max_initial_velocity_deviation = None
        self.particle_charge = None
        self.particle_mass = None
        self.particles_number = None
        self.positions = None
        self.potential = None
        self.initial_velocity = None
        self.iteration_save_rate = None
        self.iterations_number = None
        self.velocity_repartition = None
        self.velocities = None
        self.dx = self.parameters['domain_size'] / self.parameters['cells_number']
        
        #self.EPSILON_0 = 0.57  # m^(-3) s^2 ProtonMass^(-1) ElementaryCharge^2
        self.FACTOR = self.dx * self.dx / epsilon_0
        self.electric_field_filepath = os.path.join(self.directory_name, 'OutputFiles', 'electric_field.csv')
        self.followed_cell_filepath = os.path.join(self.directory_name, 'OutputFiles', 'followed_cell.csv')
        self.followed_particle_filepath = os.path.join(self.directory_name, 'OutputFiles', 'followed_particle.csv')
        self.positions_filepath = os.path.join(self.directory_name, 'OutputFiles', 'positions.csv')
        self.potential_filepath = os.path.join(self.directory_name, 'OutputFiles', 'potential.csv')
        self.velocities_filepath = os.path.join(self.directory_name, 'OutputFiles', 'velocities.csv')
        
        self.set_initial_conditions()
        self.init_arrays()
        self.write_output_files_headers()

    def set_initial_conditions(self):
        """
        Set initial conditions for the simulation.

        # Returns:
            None
        """
        self.particles_number = self.parameters['particles_number']
        self.particle_charge = self.parameters['particle_charge']
        self.particle_mass = self.parameters['particle_mass']
        self.domain_size = self.parameters['domain_size']
        self.cells_number = self.parameters['cells_number']
        self.initial_velocity = self.parameters['initial_velocity']
        self.max_initial_velocity_deviation = self.parameters['max_initial_velocity_deviation']
        self.iterations_number = self.parameters['iterations_number']
        self.iteration_save_rate = self.parameters['iteration_save_rate']
        self.velocity_repartition = self.parameters['velocity_repartition']
        self.followed_particle_index = np.random.randint(0, self.particles_number)
        self.followed_cell_index = np.random.randint(0, self.cells_number)

    def init_arrays(self):
        """
        Initialize arrays for the simulation.

        # Returns:
            None
        """
        self.positions = np.random.uniform(0, self.domain_size, size=self.particles_number)
        self.potential = np.zeros(self.cells_number)
        self.electric_field = np.zeros(self.cells_number)

        if self.velocity_repartition == "2streams":
            negative_velocities = (-np.ones(self.particles_number // 2) * self.initial_velocity +
                                   np.random.uniform(-self.max_initial_velocity_deviation,
                                                     self.max_initial_velocity_deviation,
                                                     size=self.particles_number // 2))
            positive_velocities = (np.ones(self.particles_number // 2) * self.initial_velocity +
                                   np.random.uniform(-self.max_initial_velocity_deviation,
                                                     self.max_initial_velocity_deviation,
                                                     size=self.particles_number // 2))
            self.velocities = np.concatenate((negative_velocities, positive_velocities))
        elif self.velocity_repartition == "random":
            self.velocities = np.random.uniform(-self.initial_velocity, self.initial_velocity,
                                                size=self.particles_number)
        elif self.velocity_repartition == "normal":
            self.velocities = np.random.normal(0, self.initial_velocity, size=self.particles_number)
        else:
            raise ValueError("Invalid velocity repartition")

        pd.DataFrame([self.positions]).to_csv(self.positions_filepath, index=False)
        pd.DataFrame([self.velocities]).to_csv(self.velocities_filepath, index=False)
        pd.DataFrame([self.potential]).to_csv(self.potential_filepath, index=False)
        pd.DataFrame([self.electric_field]).to_csv(self.electric_field_filepath, index=False)

    def write_output_files_headers(self):
        """
        Write headers to output files.
        
        # Returns:
            None
        """
        with open(self.followed_particle_filepath, 'w') as f:
            f.write('Position,Velocity\n')
        with open(self.followed_cell_filepath, 'w') as f:
            f.write('Potential,Electric Field\n')
        with open(os.path.join(self.directory_name, 'OutputFiles', 'dt.csv'), 'w') as f:
            f.write('dt\n')

    def compute_charge_density(self):
        """
        Compute charge density for the current iteration.
        
        # Returns:
            `numpy.ndarray`: Charge density array.
        """
        histograms, _ = np.histogram(self.positions, bins=self.cells_number, range=(0, self.domain_size))
        return self.particles_number / self.cells_number - histograms

    def compute_potential(self, density):
        """
        Update potential based on charge density using relaxation method.

        # Args:
            `density` (numpy.ndarray): Charge density array.
            
        # Returns:
            None
        """
        new_potential = ((np.roll(self.potential, 1) + np.roll(self.potential, -1)) / 2 +
                         density * self.FACTOR * (1 / 2))
        self.potential = new_potential

    def compute_electric_field(self, potential):
        """
        Compute electric field based on potential.

        # Args:
            - `potential (numpy.ndarray)`: Potential array.
            
        # Returns:
            None
        """

        self.electric_field = - np.gradient(potential, self.dx)

    def compute_particle_electric_field(self):
        """
        Compute particle electric field using interpolation.

        # Returns:
            `numpy.ndarray`: Array containing electric field for each particle.
        """
        return np.interp(self.positions, np.arange(len(self.electric_field)), self.electric_field)

    def compute_force(self, particle_electric_field):
        """
        Compute force on particles.

        # Args:
            - `particle_electric_field (numpy.ndarray)`: Array containing electric field for each particle.

        # Returns:
            `numpy.ndarray`: Force array.
        """
        return self.particle_charge * particle_electric_field

    def compute_time_step(self):
        """
        Compute time step based on particle velocities.

        # Args:
            - `iteration (int)`: Iteration index.

        # Returns:
            None
        """
        self.dt = np.min(self.dx / np.abs(self.velocities))
        time_step_filepath = os.path.join(self.directory_name, 'OutputFiles', 'dt.csv')
        with open(time_step_filepath, 'a') as f:
            f.write(str(self.dt) + '\n')

    def update_positions_velocities(self, force):
        """
        Update particle positions and velocities using Euler method.

        # Args:
            - `force (numpy.ndarray)`: Force array.

        # Returns:
            None
        """
        new_positions = self.positions + self.dt * self.velocities
        new_positions = np.mod(new_positions, self.domain_size)
        new_velocities = self.velocities + self.dt * (1 / self.particle_mass) * force
        self.positions = new_positions
        self.velocities = new_velocities

    def save_global_results(self):
        """
        Save global results to output files.
        
        # Remarks:
            - The results are saved to the following files:
                - 'positions.csv': Particle positions.
                - 'velocities.csv': Particle velocities.
                - 'electric_field.csv': Electric field.
                - 'potential.csv': Potential.
            - These datas are saved every 'iteration_save_rate' iterations.
        # Returns:
            None
        """
        pd.DataFrame([self.positions]).to_csv(self.positions_filepath, mode='a', header=False, index=False)
        pd.DataFrame([self.velocities]).to_csv(self.velocities_filepath, mode='a', header=False, index=False)
        pd.DataFrame([self.electric_field]).to_csv(self.electric_field_filepath, mode='a', header=False, index=False)
        pd.DataFrame([self.potential]).to_csv(self.potential_filepath, mode='a', header=False, index=False)

    def save_unit_results(self):
        """
        Save unit results to output files.
        
        # Remarks:
            - The results are saved to the following files:
                - 'followed_particle.csv': Followed particle position and velocity.
                - 'followed_cell.csv': Followed cell potential and electric field.
            - These datas are saved every iteration.
            
        # Returns:
            None
        """
        followed_particle_position = self.positions[self.followed_particle_index]
        followed_particle_velocity = self.velocities[self.followed_particle_index]
        followed_particle_df = pd.DataFrame(
            {'Position': [followed_particle_position], 'Velocity': [followed_particle_velocity]})
        followed_particle_df.to_csv(self.followed_particle_filepath, mode='a', header=False, index=False)

        followed_cell_potential = self.potential[self.followed_cell_index]
        followed_cell_electric_field = self.electric_field[self.followed_cell_index]
        followed_cell_df = pd.DataFrame(
            {'Potential': [followed_cell_potential], 'Electric Field': [followed_cell_electric_field]})
        followed_cell_df.to_csv(self.followed_cell_filepath, mode='a', header=False, index=False)

    def run(self):
        """
        Run the simulation.

        # Returns:
            None
        """
        for i in tqdm(range(self.iterations_number), desc='Running Simulation', unit='iterations'):
            charge_density = self.compute_charge_density()
            self.compute_potential(charge_density)
            self.compute_electric_field(self.potential)
            particle_electric_field = self.compute_particle_electric_field()
            force = self.compute_force(particle_electric_field)
            self.compute_time_step()
            self.update_positions_velocities(force)
            if i % self.iteration_save_rate == 0 or i == self.iterations_number:
                self.save_global_results()
            self.save_unit_results()
