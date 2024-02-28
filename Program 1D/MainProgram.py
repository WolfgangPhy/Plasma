import json
from Simulation import Simulation
from Visualizer import Visualizer


class MainProgram:
    
    def __init__(self):
        self.parameters = None
        self.set_parameters()
    
    def set_parameters(self):
        with open('Parameters.json', 'r') as file:
            self.parameters = json.load(file)
            
    def run(self):
        sim = Simulation(self.parameters)
        sim.run()
        Visualizer.plot_phase_space()
        Visualizer.animate_results()


if __name__ == '__main__':
    main_program = MainProgram()
    main_program.run()
