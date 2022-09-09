import sys
import os
sys.path.append(os.path.abspath("."))

from simulator import Simulator
import time
import numpy as np
from controllers.cpg_controller import CPGController
from controllers.cpg_controller import CPGParameterHandlerMAPElites
from controllers.reference_controller import Controller, tripod_gait, reshape

def read_in_cpg_individuals(filenames):
    """
    Read in the individuals (controllers) from files.

    Args:
        filenames: Array of filenames to read in controllers from.
    Returns:
        ([np.array]) The individuals.
    """
    individuals = []
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith('phase_biases:'):
                    # intrinsic_amplitudes
                    intrinsic_amplitudes = lines[i+3][1:-3].split(", ")
                    intrinsic_amplitudes = np.array(intrinsic_amplitudes)
                    intrinsic_amplitudes = CPGParameterHandlerMAPElites.sigmoid_intrinsic_amplitudes(intrinsic_amplitudes.astype(float))
                    # phase_biases
                    phase_biases = np.array(lines[i+1].replace('[','').replace(']','').replace(' ','').replace("None", "0.0").split(','))
                    phase_biases = CPGParameterHandlerMAPElites.sigmoid_phase_bias(phase_biases.astype(float))
                    individuals.append(np.concatenate((intrinsic_amplitudes,phase_biases)))
    return individuals

def display_cpg_individual(individual,delay=0):
	"""Simulates the individual/controller, displays its gait and return its fitness

	Args:
		individual: the individual (controller) in CPG formats
		delay: the number of seconds to delay between each 'frame' 
			this can help slow down the simulation
	"""
	intrinsic_amplitudes = individual[:12]
	intrinsic_amplitudes = CPGParameterHandlerMAPElites.scale_intrinsic_amplitudes(individual[:12]) # convert from 12 intrinsic amps in range [0-1]
	phase_biases = individual[12:]
	phase_biases = CPGParameterHandlerMAPElites.scale_phase_biases(individual[12:]) # convert from 144 phase biases in range [0-1]
	cpg_controller = CPGController(intrinsic_amplitudes=intrinsic_amplitudes, phase_biases=phase_biases, seconds=5, velocity=0, crab_angle=0)
	simulator = Simulator(cpg_controller, visualiser=True, collision_fatal=False, failed_legs=[])
	fitness = run_simulation(simulator,delay=delay)
	return fitness

def run_simulation(simulator, delay=0, duration=5):
	"""Runs a simulated gait for the specified duration with the specified delays

	Args:
		simulator: the simulator object
		delay: the number of seconds to delay between each 'frame' 
			this can help slow down the simulation
		duration: how long to run for

	Returns:
		fitness of gait
	"""
	x=0
	while x<(240*duration)-1:
		simulator.step()
		time.sleep(delay)
		x = x+1
	fitness = simulator.base_pos()[0]
	simulator.terminate()
	return fitness


if __name__ == "__main__":
	# os.path.join(os.path.dirname(__file__), "..", "output", "REF", "tripod-no-adapatation", f"tripod_{failure_scenario}.dat")
	individuals = read_in_cpg_individuals(["all-best-genomes.txt"])
	print("Fitness:",display_cpg_individual(individuals[-1],delay=0))