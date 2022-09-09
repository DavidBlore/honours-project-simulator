import sys
import os
sys.path.append(os.path.abspath("."))

import math
from random import random
import numpy as np
import matplotlib.pyplot as plt
from hexapod.controllers.amplitude_ode import AmplitudeOde
# from controllers.amplitude_ode import AmplitudeOde
# from amplitude_ode import AmplitudeOde

class PhaseODE():
    """Calculates the phase values for the CPG controller

    This is an ordinary differential equation integrator class. 
    What is an ODE integrator? --> https://mathinsight.org/ordinary_differential_equation_linear_integrating_factor
    Refer to the research paper to understand what the different variables are and why they have been choosen.

    Attributes:
        amplitudes: (array) the amplitudes of the oscillators at timeSTEPS [[1.2,0.02,..], [], []...]
        phase_bias: (array) the inter-oscillator phase bias - from the CPPN
        weights: (float) the weights of the Inter-oscillator couplings
        seconds: (int) the number of seconds to get values for
    """
    def __init__(self,amplitudes, phase_bias, weights=20.0, seconds=5) -> None:
        """
        Constructor for the tool to get phases based on the differential equation.
        
        self.phases=
        [
            [
                [] # joint 0, leg1
                [] # joint 1, leg1
            ],
            [
                [] # joint 0, leg2
            ],
            .
            .
            .
        ]

        Args:
            amplitudes: (array) the amplitudes of the oscillators at timeSTEPS [[1.2,0.02,..], [], []...]
            phase_bias: (array) the inter-oscillator phase bias - from the CPPN
            weights: (float) the weights of the Inter-oscillator couplings
            seconds: (int) the number of seconds to get values for
        """
        refresh_rate=240 
        self.amplitudes =  amplitudes
        self.phase_bias =  phase_bias
        self.seconds = seconds
        self.weights =  weights                         # this is just a float, not an array as Mouret used 20 for all weights
        self.num_oscillators = 12 
        # set initial conditons
        self.DT = 1/refresh_rate #0.02                  # EULERSTEPSIZE
        self.num_steps = math.floor(self.seconds/self.DT)
        self.phases = []                                # [[0.0008,...], [], []...]
        self.prev_phases = []                           # [[0.0008,...], [], []...]
        
        self.__set_initial_random_phases()              # only do this once to have starting point
        self.__set_initial_prev_phases()                # only do this once to have starting point
        self.__calculate_phases()
        
    def __set_initial_random_phases(self):
        """Sets random initial phases"""
        for leg in range(6): # 0,1,2,3,4,5
            random_intital_phases = []
            for joint in range(2): # 0,1
                # set each oscillator's starting phase to a random phase in range [0, 0.001)
                random_intital_phases.append([random()/1_000.0])
            self.phases.append(random_intital_phases)

    def __set_initial_prev_phases(self):
        """Sets initial previous phases (required for the diff. equation steps)"""
        for leg in range(6): # 0,1,2,3,4,5
            prev_phases_for_leg = []
            for joint in range(2): # 0,1
                prev_phases_for_leg.append(self.phases[leg][joint][0])
            self.prev_phases.append(prev_phases_for_leg)

    def __calculate_phases(self):
        """Calculates the phases for the oscillators based on the differential equation"""
        for t in range(1,(self.num_steps)):
            # print(t) # FOR DEBUG
            for leg in range(6):
                for joint in range(2):
                    summation = 0.0
                    for other_leg in range(6):
                        for other_joint in range(2):
                            if not((leg == other_leg) and (joint == other_joint)): # don't include the coupling between the same oscillator
                                part1 = self.amplitudes[other_leg][other_joint][t] * self.weights
                                summation += part1 * np.sin(self.prev_phases[other_leg][other_joint] - self.prev_phases[leg][joint] - self.phase_bias[leg][joint][other_leg][other_joint]) #TODO: change phase bias structure
                    self.phases[leg][joint].append(self.prev_phases[leg][joint] + self.DT * (2*np.pi + summation))
            # we now set the prev_phases to current timeSTEPs phase so we progress
            for leg in range(6):
                for joint in range(2):
                    self.prev_phases[leg][joint] = self.phases[leg][joint][t]

    def getPhases(self):
        """
        Gets the theta-i values for the CPG (i.e., for each oscillator)

        This correspondes to the phases of the oscillators at each time step
        and is the output of the 1st ordinary differential equation. 

        Returns:
            the phases of the oscillators at each time step
        """
        return self.phases

if __name__ == "__main__":
    # do any debugging here
    for z in range(100):
        # get random phase bias instead of CPPN
        phase_bias = [[[[None, 3.5068019077724797], [3.728927645335739, 6.012972791916583], [2.197010599305194, 5.065503837841195], [4.730862129678579, 4.507256645104633], [1.2641833418580946, 4.203399690412492], [3.915382333750674, 3.1375775056921724]], [[5.062429061063832, None], [0.5169827321900529, 0.8610105130959634], [2.3780509266518606, 3.4583163396548944], [1.910505729723967, 5.231708114666355], [0.9136270592487555, 3.854002878133841], [2.406979435118308, 5.202182611486277]]], [[[3.7841216161022486, 5.387775635016577], [None, 0.9222664364827131], [5.0237163402614575, 3.734745044159147], [2.3796500639657023, 3.7954622906084268], [2.4699789866870643, 3.0017117955159867], [3.339579481018237, 2.7377795366269626]], [[5.489906983744749, 5.425031250402702], [5.491015248716992, None], [5.395093335779441, 1.7894941022156095], [0.1340489383736345, 3.055104039629926], [5.559272795860348, 3.617770350210537], [5.589117822875999, 1.4985064218100657]]], [[[5.034384704235012, 0.6023278999637408], [1.5534892719609912, 3.9842515077604963], [None, 6.225167610016697], [6.007996661756584, 1.352349559335411], [1.0946147883858746, 1.31933467670442], [5.2810054464945075, 5.983040765681909]], [[0.9167027318168743, 4.8799428061506465], [3.576446097604909, 4.46007286128076], [3.339159910291705, None], [5.081935638796572, 4.713585634197353], [1.2193042649239423, 4.053532252245812], [0.993963651325376, 5.302272628283921]]], [[[0.623390350986485, 3.664262003279918], [3.261802668901521, 3.4900470523372578], [5.1177095924255545, 6.008661899946037], [None, 1.6023226213650459], [3.057648962387964, 5.322222925742818], [1.623991072563872, 2.616889559942588]], [[1.0024840455166881, 2.2070912498169672], [2.931827464915592, 0.3489043743800492], [3.331252593679254, 5.3641803433072655], [3.5213267116103237, None], [0.39389732916472464, 5.236456389974808], [3.770795029832701, 1.5488104540003085]]], [[[4.492773171550187, 3.397578922082763], [6.087113105685607, 6.049534540089985], [5.790466193840465, 5.300175214994115], [2.3771124208950045, 4.87139212181459], [None, 3.096376668434617], [3.027901182067269, 4.514440295619322]], [[0.8546825416249778, 3.832229416306182], [2.4190018014058463, 2.323477116521129], [4.337071252531072, 4.401024077062711], [4.669624288583032, 0.6522369235529866], [0.5860895725350285, None], [1.4950478653861523, 4.078514205591889]]], [[[5.296341405915791, 1.9340501298641553], [2.2602430347939158, 0.94303443514123], [3.5125231478533325, 1.1707583349481119], [5.334849563931699, 5.316805717766203], [3.7809601074465333, 2.043774219475812], [None, 3.617781832474402]], [[2.6145427670231585, 5.070782709609786], [1.8272597729397413, 0.6762340963548555], [0.9105819201268035, 2.683203725945398], [0.331490269192378, 2.081679575898442], [1.321574577682836, 4.280588346137638], [3.350946817215021, None]]]]
        """
        phase_bias = [
            [ leg0
                [ joint0
                    [other_leg0_joint_0, other_leg0_joint_1],
                    [other_leg1_joint_0, other_leg1_joint_1],
                    [other_leg2_joint_0, other_leg2_joint_1],
                    [other_leg..._joint_0, other_leg..._joint_1],
                ],
                [],
                [],
                [],
                [],
                [],
            ],
            [],
            [],
            [],
            [],
            [],
        ]
        """
        # get random intrinsic amplitudes (radians) instead of CPPN
        A = np.random.rand(12)
        # A = np.zeros(18)+1

        # get amplitudes
        amplitude_ode = AmplitudeOde(10, A, 5, a_0=1, da_0=1, refresh_rate=240)
        a = amplitude_ode.getA()

        ode = PhaseODE(amplitudes=a, phase_bias=phase_bias, weights=20.0, seconds=5, refresh_rate=240)
        
        print(ode.getPhases())
        t = np.linspace(0,5,5*240)
        plt.plot(t,ode.getPhases()[0][0], 'b', label="leg0-joint-0")
        plt.plot(t,ode.getPhases()[1][0], 'g', label="leg1-joint-0")
        plt.plot(t,ode.getPhases()[0][1], 'b--', label="leg0-joint-1")
        plt.plot(t,ode.getPhases()[1][1], 'g--', label="leg1-joint-1")
        # plt.plot(t,ode.getPhases()[2], 'r--', label="osc-2")
        # plt.plot(t,ode.getPhases()[3], 'y--', label="osc-3")
        plt.legend(loc="best")
        plt.title("CPG Oscillator")
        plt.xlabel("time")
        plt.ylabel("theta-i (phase)")
        plt.show()
        # print(ode.getPhases()[0].size)


"""
Interesting phase bias:
[3.85114812 1.68515321 2.71091183 4.51732024 4.13272325 3.84634855
  0.38495832 0.06967663 5.76942305 5.77944091 3.99671837 6.1912333
  4.17360289 2.33876799 5.113237   0.08396763 4.24120175 0.07281165]

[1.08594541 5.93501584 5.87685209 1.62095235 0.60660627 4.50066679
  0.99790022 6.06347293 2.30730949 0.43500973 2.34190131 3.64737072
  0.46683061 5.8179908  2.19234595 0.01553603 3.2080396  1.02544959]
"""