import numpy as np
from tqdm import tqdm
import pandas as pd

class Simulation:
    """
    Class for simulating a 1D plasma using the Particle-in-Cell method.

    # Args:
        - `parameters (dict)`: Dictionary containing simulation parameters.

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

    def __init__(self, parameters):
        self.potential_array = None
        self.parameters = parameters
        self.dx = self.parameters['domain_size'] / self.parameters['cells_number']
        self.dt = None
        self.EPSILON_0 = 0.55
        self.FACTOR = self.dx * self.dx / self.EPSILON_0
        self.set_initial_conditions()
        self.init_arrays()

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
        self.tolerance = self.parameters['tolerance']
        self.max_iteration_number_potential = self.parameters['max_iteration_number_potential']

    def init_arrays(self):
        """
        Initialize arrays and matrices for the simulation.

        # Returns:
            None
        """
        self.potential_matrix = -2 * np.eye(self.cells_number) + np.eye(self.cells_number, k=1) + \
                                np.eye(self.cells_number, k=-1)
        self.potential_matrix[0, -1] = self.potential_matrix[-1, 0] = 1

        self.positions = np.random.uniform(0, self.domain_size, size=self.particles_number)
        self.potential_array = np.zeros(self.cells_number)
        
        negative_velocities = -np.ones(self.particles_number//2)*self.initial_velocity+ \
            np.random.uniform(-self.max_initial_velocity_deviation, self.max_initial_velocity_deviation,
                              size=self.particles_number//2)
        positive_velocities = np.ones(self.particles_number//2)*self.initial_velocity+ \
            np.random.uniform(-self.max_initial_velocity_deviation, self.max_initial_velocity_deviation,
                                size=self.particles_number//2)
        self.velocities = np.concatenate((negative_velocities, positive_velocities))
        
        position_df = pd.DataFrame([self.positions])
        position_df.to_csv('positions.csv', index=False)
        velocity_df = pd.DataFrame([self.velocities])
        velocity_df.to_csv('velocities.csv', index=False)

    def compute_charge_density(self, iteration):
        """
        Compute charge density at a given iteration.

        # Args:
            - `iteration (int)`: Iteration index.

        # Returns:
            `numpy.ndarray`: Charge density array.
        """
        histograms, _ = np.histogram(self.positions, bins=self.cells_number, range=(0, self.domain_size))
        return (self.particles_number) / self.cells_number - histograms

    def compute_potential_relaxation(self, density):
        """
        Update potential based on charge density using relaxation method.

        Args:
            density (numpy.ndarray): Charge density array.
        """
        new_potential = ((np.roll(self.potential_array, 1) + np.roll(self.potential_array, -1)) / 2 +
                         density * self.FACTOR * (1 / 2))
        self.potential_array = new_potential

    def compute_potential_jacobi(self, density):

        for p in range(self.max_iteration_number_potential):

            new_potential = ((np.roll(self.potential_array, 1) + np.roll(self.potential_array, -1)) / 2 +
                             density * self.FACTOR * (1 / 2))
            new_potential = np.mod(new_potential, self.domain_size)

            delta = np.max(np.abs(new_potential - self.potential_array[-1, :]))

            if delta < self.tolerance:
                break
            return new_potential

    def compute_potential_matrix_inversion(self, density):
        """
        Compute potential based on charge density.

        # Args:
            - `density (numpy.ndarray)`: Charge density array.

        # Returns:
            `numpy.ndarray`: Potential array.
        """
        potential = np.linalg.solve(self.potential_matrix, - density * self.FACTOR)  # TODO : corriger
        self.potential_array = potential

    def compute_potential_Fourier(self, density):
        """
        Compute potential based on charge density using Fourier method.

        # Args:
            - `density (numpy.ndarray)`: Charge density array.

        # Returns:
            `numpy.ndarray`: Potential array.
        """
        density_fourier = np.fft.fft(density)
        potential_fourier = density_fourier / (
                    -np.fft.fftfreq(self.cells_number) * np.fft.fftfreq(self.cells_number)) * self.FACTOR
        potential = np.fft.ifft(potential_fourier)
        self.potential_array = np.vstack((self.potential_array, potential))

    def compute_electric_field(self):
        """
        Compute electric field based on potential.

        # Args:
            - `potential (numpy.ndarray)`: Potential array.

        # Returns:
            `numpy.ndarray`: Electric field array.
        """
        return - np.gradient(self.potential_array, self.dx)

    def compute_particle_electric_field(self, electric_field, iteration):
        """
        Compute particle electric field.

        # Args:
            - `electric_field (numpy.ndarray)`: Electric field array.
            - `iteration (int)`: Iteration index.

        # Returns:
            `numpy.ndarray`: Particle electric field array.
        """
        return np.interp(self.positions, np.arange(len(electric_field)), electric_field)

    def compute_force(self, particle_electric_field):
        """
        Compute force on particles.

        # Args:
            - `particle_electric_field (numpy.ndarray)`: Particle electric field array.

        # Returns:
            `numpy.ndarray`: Force array.
        """
        return self.particle_charge * particle_electric_field

    def compute_time_step(self, iteration):
        """
        Compute time step based on particle velocities.

        # Args:
            - `iteration (int)`: Iteration index.

        # Returns:
            None
        """
        self.dt = np.min(self.dx / np.abs(self.velocities))
        with open('dt.csv', 'a') as f:
            f.write(str(self.dt) + '\n')

    def update_positions_velocities(self, force, iteration):
        """
        Update particle positions and velocities using Euler method.

        # Args:
            - `force (numpy.ndarray)`: Force array.
            - `iteration (int)`: Iteration index.

        # Returns:
            None
        """
        new_positions = self.positions + self.dt * self.velocities
        new_positions = np.mod(new_positions, self.domain_size)
        new_velocities = self.velocities + self.dt * (1 / self.particle_mass) * force
        self.positions = new_positions
        self.velocities = new_velocities
        
        # Save positions and velocities in a new column of the csv file
        pd.DataFrame([self.positions]).to_csv('positions.csv', mode='a', header=False, index=False)
        pd.DataFrame([self.velocities]).to_csv('velocities.csv', mode='a', header=False, index=False)
        

    def run(self):
        """
        Run the simulation.

        # Returns:
            None
        """
        for iteration in tqdm(range(self.iterations_number)):
            charge_density = self.compute_charge_density(iteration) # WORKS
            # self.compute_potential_matrix_inversion(charge_density)
            self.compute_potential_relaxation(charge_density)
            # self.compute_potential_Fourier(charge_density
            # potential = self.compute_potential_jacobi(charge_density
            # self.potential_array = np.vstack((self.potential_array, potential))
            electric_field = self.compute_electric_field()
            particle_electric_field = self.compute_particle_electric_field(electric_field, iteration)
            force = self.compute_force(particle_electric_field)
            self.compute_time_step(iteration)
            self.update_positions_velocities(force, iteration)
