import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class AmplitudeOde:
    """Calculates the amplitude values for the CPG controller

    This is an ordinary differential equation integrator class. 
    What is an ODE integrator? --> https://mathinsight.org/ordinary_differential_equation_linear_integrating_factor
    Refer to the research paper to understand what the different variables are and why they have been choosen.

    Attributes:
        b: the constant b (radians) default:10
        A: the intrinsic amplitudes from CPPN output (radians)
        seconds: number of 'seconds' to get values for
        a_0: initial amplitude (radians)
        da_0: initial amplitude derivative (radians/second)
    """

    def __init__(self, b, A, seconds, a_0=1, da_0=0):
        """
        Constructor for tool to get amplitude values for oscillators based on the
        differential equation.

        Args:
            b: the b constant (radians) default:10 
            A: the intrinsic amplitudes from CPPN output (radians)
            seconds: number of 'seconds' to get values for
            a_0: initial amplitude (radians)
            da_0: initial amplitude derivative (radians/second)
        """
        refresh_rate=240                            # how often to get values per second
        self.b = b                                  # radians constant 
        self.A = A                                  # intrinsic amplitudes
        self.seconds = np.linspace(0, seconds, refresh_rate*seconds)  # time points
        self.a_0 = a_0                              # intitial condition
        self.da_0 = da_0                            # intitial condition
        self.S_0 = (a_0, da_0)                      # intitial conditions
        self.a = []                                 # amplitudes

        for i in range(len(self.A)):
            # print(odeint(self.dSdt, self.S_0,self.seconds, args=(i,), tfirst=True).T[0])
            # as per mouret's C++ code, the intital condition for amplitude is the intrinsic amplitude
            self.a_0 = self.A[i]
            self.S_0 = (a_0, da_0)
            self.a.append( odeint(self.dSdt, self.S_0, self.seconds, args=(i,), tfirst=True).T[0] )

    def getA(self):
        """
        Returns the amplitude (a) values for all oscilators
        """
        shaped_a = [
            [],
            [],
            [],
            [],
            [],
            [],
        ]
        shaped_a[0].append(self.a[0])
        shaped_a[0].append(self.a[1])
        shaped_a[1].append(self.a[2])
        shaped_a[1].append(self.a[3])
        shaped_a[2].append(self.a[4])
        shaped_a[2].append(self.a[5])
        shaped_a[3].append(self.a[6])
        shaped_a[3].append(self.a[7])
        shaped_a[4].append(self.a[8])
        shaped_a[4].append(self.a[9])
        shaped_a[5].append(self.a[10])
        shaped_a[5].append(self.a[11])
        self.a = shaped_a
        return self.a

    def getAi(self, i):
        """
        Returns the amplitude (a) value for a given oscilator at a certain index
        
        Args:
            i: oscilator index
        """
        return self.a[i]

    def getT(self):
        """Returns the number of 'seconds' the ODE integrator is getting values for"""
        return self.seconds
    
    def dSdt(self,seconds,S,i):
        """Implementation of the differential equation for the amplitude"""
        # v = da/dt
        a, v = S
        return [
            v,
            self.b*(self.b/4 * (self.A[i]-a-v))
        ]

if __name__ == "__main__":
    """Runs code to test the ODE is working as expected
    """
    # get random intrinsic amplitudes (radians) instead of CPPN
    A = np.random.rand(12)
    print(A)

    # ode = AmplitudeOde(10, np.zeros(18)+1.5, 10, a_0=1, da_0=1, refresh_rate=100) # use constant for A
    ode = AmplitudeOde(10, A, 10, a_0=1, da_0=1, refresh_rate=50) # use random for A
    a0 = ode.getAi(0)
    a1 = ode.getAi(1)
    a2 = ode.getAi(2)
    a3 = ode.getAi(3)
    plt.plot(ode.getT(), a0, 'b-',label='a0')
    plt.plot(ode.getT(), a1, 'g--',label='a1')
    plt.plot(ode.getT(), a2, 'r--',label='a2')
    plt.plot(ode.getT(), a3, 'y--',label='a3')
    plt.legend(loc='best')
    plt.show()
