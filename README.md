# Plasma Instabilities with a 1D Particle-In-Cell code

# Description

This repository contains a 1D Particle-In-Cell (PIC) code that simulates the evolution of a plasma with a fixed
background of ions and a population of electrons.

# Installation

To install the code, you can clone the following repository:  https://github.com/WolfgangPhy/Plasma using the following command:

```bash
git clone https://github.com/WolfgangPhy/Plasma
```

You can also download the code as a zip file by clicking on the green button "Code" and then "Download ZIP".

# Documentation

The documentation of the code is available in the `docs` directory. You can find it in two fomat, in pdf and html. The html format is more user friendly and you can navigate through the differents classes and methods easily. To use it open the `index.html` file in your web browser.

Of course, the documentation is also available in the code itself. You can use the `help()` function in python to get information about a class or a method

# How it works ?

The code is build such that all parameters are tunable form the 'Parameters.json' file. When you run the code 
(see *How to run the code ?* part), it will automatically create a directory for the output files and plots in the
'Tests' directory. The directory is named with the parameters used for the simulation using this scheme :

```
PN_{particle_number}_DS_{domain_size}_CN_{cell_number}_V_{initial_velocity}_PC_{particle_charge}_PM_{particle_mass}_VR_{velocity_repartition}
```

The test directory will have the following structure:

```
Directory
│
└───OutputFiles
│   │   dt.csv
│   │   electric_field.csv
│   │   followed_cell.csv
│   │   followed_particle.csv
│   │   positions.csv
│   │   velocities.csv
└───Plots
    └─── Followed_Cell
    │   │   single_cell_electric.png
    │   │   single_cell_potential.png
    └─── Followed_Particle
    │   │   single_particle_position.png
    │   │   single_particle_phase_space.png
    └─── Global
    │   └───Electric_Field
    │   │   │   electric_field_{iteration_number}.png
    │   │   │   ...
    │   └───Potential
    │   │   │   potential_{iteration_number}.png
    └─── Phase_Space
    │   │   phase_space_{iteration_number}.png
    │   │   ...
```

# Structure of the code

This code is build using the single responsibility principle. Each class has a single responsibility and the code is organized in a way that makes it easy to understand and to modify. Each class is in its own file and the code is organized in the following way:

- `MainProgram.py` : This is the main file of the code. It get the parameters from the `Parameters.json` file
and use the `Simulation` class to run the simulation and the `Visulizer` class to plot the results.
- `Simulation.py` : This class is responsible for the simulation. It contains the `run` method that runs the simulation.
- `Visualizer.py` : This class is responsible for the visualization of the results. It multiple methods 
to plot the results.
- `FileHelper.py` : This class is responsible for the file handling. It helps to create the output directory and gives
back the name of the current test directory.

# How to run the code ?

To run the code, you need to have the following dependencies installed:
- numpy
- seaborn
- tqdm
- pandas


Then you need to launch the `MainProgram.py` file using the following command:

```bash
python MainProgram.py
```

If you want to really control the lauch of the code you need to create an instance of the `MainProgram` class and call
 the `execute()` method :
 
 ```python
 from MainProgram import MainProgram
 main_program = MainProgram()
 main_program.execute()
```

# Parameters

The `Parameters.json` looks like this:

```json
{
    "cells_number": 300,
    "domain_size": 100,
    "max_initial_velocity_deviation": 0.1,
    "particle_charge": -1.0,
    "particle_mass": 1.0,
    "particles_number": 100000,
    "initial_velocity": 1e6,
    "iteration_save_rate": 500,
    "iterations_number": 10000,
    "velocity_repartition": "normal"
}
```

The parameters are the following:
- `cells_number` : The number of cells in the domain
- `domain_size` : The size of the domain
- `max_initial_velocity_deviation` : The maximum deviation of the initial velocity of the particles (only used for
2streams velocity repartition)
- `particle_charge` : The charge of the particles (in elementary charge)
- `particle_mass` : The mass of the particles (in proton mass)
- `particles_number` : The number of particles
- `initial_velocity` : The initial velocity of the particles
- `iteration_save_rate` : The rate at which the results are saved
- `iterations_number` : The number of iterations
- `velocity_repartition` : The velocity repartition of the particles. Can be either "normal", "2streams" or "random"
