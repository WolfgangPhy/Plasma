import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
class Visualizer:
    
    @staticmethod
    def plot_phase_space():
        #load data from npz file
        data = np.load('results.npz')
        positions_array = data['positions']
        velocities_array = data['velocities']
        positions = positions_array[-1, :]
        velocities = velocities_array[-1, :]
        plt.figure()
        plt.scatter(positions, velocities)
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        #plt.xlim(0., 1.)
        #plt.ylim(-1., 1.)
        plt.title('Phase space')
        plt.show()
        
    @staticmethod
    def animate_results(file_path='results.npz'):
        data = np.load(file_path)
        positions = data['positions']
        velocities = data['velocities']
        dt = data['dt']

        # Initialiser la figure
        fig, ax = plt.subplots()
        scatter, = ax.plot([], [], 'ro')

        # Définir les limites des axes
        

        # Initialisation de l'animation
        def init():
            ax.set_xlim(0., 1.)
            ax.set_ylim(1., 1)
            return scatter,

        # Fonction d'animation
        def update(frame):
            x = positions[frame, :]
            y = velocities[frame, :]
            scatter.set_data(x, y)
            return scatter,

        # Créer l'animation
        num_frames = len(dt)
        animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

        # Afficher l'animation
        plt.show()