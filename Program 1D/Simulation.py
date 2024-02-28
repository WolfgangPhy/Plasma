import numpy as np
from tqdm import tqdm

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
        self.parameters = parameters
        self.dx = self.parameters['domain_size'] / self.parameters['cells_number']
        self.EPSILON_0 = 8.854187817e-12
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
        self.max_initial_velocity_deviation = self.parameters['max_initial_velocity_deviation']
        self.iterations_number = self.parameters['iterations_number']

    def init_arrays(self):
        """
        Initialize arrays and matrices for the simulation.

        # Returns:
            None
        """
        self.potential_matrix = -2 * np.eye(self.cells_number) + np.eye(self.cells_number, k=1) +\
                                np.eye(self.cells_number, k=-1)
        self.potential_matrix[0, -1] = self.potential_matrix[-1, 0] = 1
        
        self.positions = np.random.uniform(0, self.domain_size, size=(1, self.particles_number))
        self.velocities = np.random.choice([-1, 1], size=(1, self.particles_number)) +\
            np.random.uniform(-self.max_initial_velocity_deviation, self.max_initial_velocity_deviation,
                              size=(1, self.particles_number))
        self.dt = np.zeros(self.iterations_number)

    def compute_charge_density(self, iteration):
        """
        Compute charge density at a given iteration.

        # Args:
            - `iteration (int)`: Iteration index.

        # Returns:
            `numpy.ndarray`: Charge density array.
        """
        histograms, _ = np.histogram(self.positions[iteration, :], bins=self.cells_number, range=(0, self.domain_size))
        return self.particles_number / self.cells_number - histograms

    def compute_potential(self, density):
        """
        Compute potential based on charge density.

        # Args:
            - `density (numpy.ndarray)`: Charge density array.

        # Returns:
            `numpy.ndarray`: Potential array.
        """
        return np.linalg.solve(self.potential_matrix, - density * self.FACTOR)

    def compute_electric_field(self, potential):
        """
        Compute electric field based on potential.

        # Args:
            - `potential (numpy.ndarray)`: Potential array.

        # Returns:
            `numpy.ndarray`: Electric field array.
        """
        return - np.gradient(potential, self.dx)

    def compute_particle_electric_field(self, electric_field, iteration):
        """
        Compute particle electric field.

        # Args:
            - `electric_field (numpy.ndarray)`: Electric field array.
            - `iteration (int)`: Iteration index.

        # Returns:
            `numpy.ndarray`: Particle electric field array.
        """
        return np.interp(self.positions[iteration, :], np.arange(len(electric_field)), electric_field)

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
        self.dt[iteration] = np.min(self.dx / np.abs(self.velocities[iteration, :]))

    def update_positions_velocities(self, force, iteration):
        """
        Update particle positions and velocities using Euler method.

        # Args:
            - `force (numpy.ndarray)`: Force array.
            - `iteration (int)`: Iteration index.

        # Returns:
            None
        """
        new_positions = self.positions[-1, :] + self.dt[iteration]*self.velocities[-1,:]
        new_velocities = self.velocities[-1,:] + self.dt[iteration]*(1/self.particle_mass)*force
        
    def save_results(self):
        """
        Save simulation results to 'results.npz'.

        # Returns:
            None
        """
        np.savez('results.npz', positions=self.positions, velocities=self.velocities, dt=self.dt)

    def run(self):
        """
        Run the simulation.

        # Returns:
            None
        """
        for iteration in tqdm(range(self.iterations_number)):
            charge_density = self.compute_charge_density(iteration)
            potential = self.compute_potential(charge_density)
            electric_field = self.compute_electric_field(potential)
            particle_electric_field = self.compute_particle_electric_field(electric_field, iteration)
            force = self.compute_force(particle_electric_field)
            self.compute_time_step(iteration)
            self.update_positions_velocities(force, iteration)
        self.save_results()
