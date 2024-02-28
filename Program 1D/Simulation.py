import numpy as np
from tqdm import tqdm


class Simulation:

    def __init__(self, parameters):
        self.parameters = parameters
        self.dx = self.parameters['domain_size'] / self.parameters['cells_number']
        self.EPSILON_0 = 8.854187817e-12
        self.FACTOR = self.dx * self.dx / self.EPSILON_0
        self.set_initial_conditions()
        self.init_arrays()

    def set_initial_conditions(self):
        self.particles_number = self.parameters['particles_number']
        self.particle_charge = self.parameters['particle_charge']
        self.particle_mass = self.parameters['particle_mass']
        self.domain_size = self.parameters['domain_size']
        self.cells_number = self.parameters['cells_number']
        self.max_initial_velocity_deviation = self.parameters['max_initial_velocity_deviation']
        self.iterations_number = self.parameters['iterations_number']

    def init_arrays(self):
        self.potential_matrix = -2 * np.eye(self.cells_number) + np.eye(self.cells_number, k=1) +\
                                np.eye(self.cells_number, k=-1)
        self.potential_matrix[0, -1] = self.potential_matrix[-1, 0] = 1
        
        self.positions = np.random.uniform(0, self.domain_size, size=(1, self.particles_number))
        self.velocities = np.random.choice([-1, 1], size=(1, self.particles_number)) +\
            np.random.uniform(-self.max_initial_velocity_deviation, self.max_initial_velocity_deviation,
                              size=(1, self.particles_number))
        self.dt = np.zeros(self.iterations_number)

    def compute_charge_density(self, iteration):
        histograms, _ = np.histogram(self.positions[iteration, :], bins=self.cells_number, range=(0, self.domain_size))
        return self.particles_number / self.cells_number - histograms

    def compute_potential(self, density):
        return np.linalg.solve(self.potential_matrix, - density * self.FACTOR)

    def compute_electric_field(self, potential):
        return - np.gradient(potential, self.dx)

    def compute_particle_electric_field(self, electric_field, iteration):
        return np.interp(self.positions[iteration, :], np.arange(len(electric_field)), electric_field)

    def compute_force(self, particle_electric_field):
        return self.particle_charge * particle_electric_field

    def compute_time_step(self, iteration):
        self.dt[iteration] = np.min(self.dx / np.abs(self.velocities[iteration, :]))

    #update_positions_velocities using a Euler method
    def update_positions_velocities(self, force, iteration):
        #TODO: update positions and velocities using a Euler method

    def save_results(self):
        np.savez('results.npz', positions=self.positions, velocities=self.velocities, dt=self.dt)

    def run(self):
        for iteration in tqdm(range(self.iterations_number)):
            charge_density = self.compute_charge_density(iteration)
            potential = self.compute_potential(charge_density)
            electric_field = self.compute_electric_field(potential)
            particle_electric_field = self.compute_particle_electric_field(electric_field, iteration)
            force = self.compute_force(particle_electric_field)
            self.compute_time_step(iteration)
            self.update_positions_velocities(force, iteration)
        self.save_results()
