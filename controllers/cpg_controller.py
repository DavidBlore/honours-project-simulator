import sys
import os
sys.path.append(os.path.abspath("."))

import math
import numpy as np
from hexapod.controllers.amplitude_ode import AmplitudeOde
# from controllers.amplitude_ode import AmplitudeOde
# from amplitude_ode import AmplitudeOde
from hexapod.controllers.phase_ode import PhaseODE
# from controllers.phase_ode import PhaseODE
# from phase_ode import PhaseODE

class CPGController:
    """Implementation of Mouret's CPG model

    Refer to the research paper to understand what the different variables are and why they have been choosen.

    Attributes:
        intrinsic_amplitudes: (list) of the intrinsic amplitudes of the oscillators
        phase_biasess: (list) of the phase biases of the oscillators
        body_height: (float) height of the body in meters.
        velocity: (float) velocity of the body in meters per second.
        crab_angle: (float) angle of the crab in radians.
    """

    def __init__(self, intrinsic_amplitudes, phase_biases, seconds=5, body_height=0.15, velocity=0.46, crab_angle=-1.57) -> None:
        '''
        Creates CPG for specified time range.

        Args:
            intrinsic_amplitudes: (list) of the intrinsic amplitudes of the oscillators
            phase_biasess: (list) of the phase biases of the oscillators
            body_height: (float) height of the body in meters.
            velocity: (float) velocity of the body in meters per second.
            crab_angle: (float) angle of the crab in radians.
        '''
        self.body_height = body_height
        self.velocity = velocity
        self.crab_angle = crab_angle
        
        self.seconds = seconds
        self.intrinsic_amplitudes = intrinsic_amplitudes
        self.phase_biases = phase_biases
        self.amp_ode = AmplitudeOde(10,intrinsic_amplitudes,self.seconds)
        self.amplitudes = self.amp_ode.getA()
        self.phase_ode = PhaseODE(self.amplitudes,self.phase_biases,self.seconds)
        self.phases = self.phase_ode.getPhases()
        self.__generateServoValues()

    def joint_angles(self, t):
        '''
        Returns the joint angles for the given time t.

        Args
            t: (int) time in 1/240Hz steps (i.e., 0.7499999999999986= timestep 180)
        '''
        timestep = round(t*240) # convert t to timestep 
        # print("timestep:",timestep," t:",t) # FOR DEBUG: show timestep and t
        servo_values_t = np.array(self.getServoValues()[timestep]).flatten()
        return servo_values_t

    def getServoValues(self):
        """
        Returns all the yi values for the CPG (i.e., for each oscillator).

        This corresponds to the servo actuation values (i.e., the 2nd ordinary differential equation)
        yi = ai*cos(thetai)
        """
        return self.servo_values

    def __generateServoValues(self):
        """
        Calculates all the yi values for the CPG (i.e., for each oscillator).
        This method should only be called once in the constructor,
        use the getServoValues method to get the yi values instead.

        yi = ai*cos(thetai)
        """
        num_steps = math.floor(240*self.seconds) # = refresh rate * duration
        self.servo_values = [] 
        """
        Example values:
        self.servo_values = [
            [ #t0
                [ #leg0
                    [1.12, 0.2324, -1.2404] # joint0,1,2
                ],
                [ #leg1
                    [1.12, 0.2324, -1.2404] # joint0,1,2
                ],
                [ #leg2
                    [1.12, 0.2324, -1.2404] # joint0,1,2
                ],
                ...
            ],            
            [], #t2        
            [], #t3            
        ]
        """
        for t in range(1,num_steps):
            leg_values = []
            for leg in range(6):
                joint_values =[]
                for joint in range(3):
                    # calcuate the servo value for this oscillator
                    if (joint == 2):
                        # if this is the tibia oscillator (i.e., the last one) then 
                        # set its servo value to the negative of the femur joints angle - 80 degrees
                        # (to make the tibia point down to the ground)
                        raw_value = (-joint_values[1]) - 1.3962634
                        joint_values.append(CPGParameterHandler.scale_servo_value(raw_value,2))
                        # self.servo_values[i][t] = 0#-1.3962634 # FOR DEBUG: set tibia to 0
                    else:
                        # calculate value for oscilator femur/coxa joint (oscillator)
                        yi = self.amplitudes[leg][joint][t] * np.cos(self.phases[leg][joint][t]) # 'differential' equation for the controller 
                        joint_values.append(CPGParameterHandler.scale_servo_value(yi,joint))
                leg_values.append(joint_values)
            self.servo_values.append(leg_values)
        # print("servo_values:",self.servo_values)
        return self.servo_values

    def get_starting_joint_angles(self):
        '''
        Returns the starting joint angles for the CPG.

        Returns:
            values for a stance that is a neutral standing pose
        '''
        self.starting_joint_angles = [0.00000000e+00,1.83591547e-01,-1.94791007e+00,2.42563874e-01,6.26641372e-02,-2.08813169e+00,-2.42563874e-01,6.26641372e-02,-2.08813169e+00,2.44929360e-17,1.83591547e-01,-1.94791007e+00,2.42563874e-01,6.26641372e-02,-2.08813169e+00,-2.42563874e-01,6.26641372e-02,-2.08813169e+00]
        return self.starting_joint_angles

class CPGParameterHandler:
    """
    Scales the output values of oscillators/nerons to the legal range.
    """
    def bipolar_sigmoid(x):
        """
        Bipolar Sigmoid function. Converts x to a value between -1 and 1.

        Args:
            x: (float) value to convert to bipolar sigmoid.

        Returns:
            (float) output value between -1 and 1
        """
        if x>8:
            return 1
        if x<-8:
            return -1
        return (1-math.exp(-x))/(1+math.exp(-x))
    
    def sigmoid(x):
        """
        Sigmoid function. Converts x to a value between 0 and 1.
        
        Args:
            x: (float) value to convert to sigmoid.

        Returns:
            (float) value between 0 and 1.
        """
        if x>8:
            return 1
        if x<-8:
            return 0
        return 1/(1+math.exp(-x))

    def scale_intrinsic_amplitudes(intrinsic_amplitudes):
        """
        Converts the intrinsic amplitudes to the correct range.

        Args:
            intrinsic_amplitudes: (list) of the intrinsic amplitudes of the oscillators

        Returns:
            (list) of the scaled intrinsic amplitudes of the oscillators
        """
        intrinsic_amplitudes = np.reshape(intrinsic_amplitudes, (6,3))
        """
        ## Joint 1 (0) - coxa
        [-100, 100] degrees
        [-1.74533,1.74533] radians

        ## Joint 2 (1) - femur
        [-150, 130] degrees
        [-2.61799, 2.26893] radians

        ## Joint 3 (2) - tibia
        [-170, 130] degrees
        [-2.96706, 2.26893] radians
        """
        for leg in range(6):
            for joint in range(3):
                intrinsic_amplitudes[leg][joint] = CPGParameterHandler.scale_intrinsic_amplitude(intrinsic_amplitudes[leg][joint], joint)
        return intrinsic_amplitudes

    def scale_intrinsic_amplitude(intrinsic_amplitude, joint):
        """
        Converts the intrinsic amplitude to the correct range for the given joint.

        Args:
            intrinsic_amplitude: (float) the raw intrinsic amplitude for the joint/oscillator
            joint: (int) the joint (0/1/2)
        
        Returns:
            (float) the correct, adjusted intrinsic amplitude of the oscillator
        """
        if joint==0:
            return ((intrinsic_amplitude * 3.49066) - 1.74533)
        if joint==1:
            return ((intrinsic_amplitude * 4.88692) - 2.61799)
        if joint==2:
            return ((intrinsic_amplitude * 5.23599) - 2.96706)

    def scale_servo_value(servo_value, joint):
        """
        Scales the servo value to the correct range for the given joint.

        Args:
            servo_value: (float) the raw servo value for the joint/oscillator
            joint: (int) the joint (0/1/2)
        
        Returns:
            (float) the correct, adjusted servo output for the oscillator
        """
        if joint==0:
            if servo_value<-1.74533 or servo_value>1.74533:
                return 1.74533
            else:
                return servo_value
        if joint==1:
            if servo_value<-2.61799: 
                return 2.61799
            if servo_value>2.26893:
                return 2.26893
            else:
                return servo_value
        if joint==2:
            if servo_value<-2.96706:
                return 2.96706
            if servo_value>2.26893:
                return 2.26893
            else:
                return servo_value

    def scale_phase_bias(phase_bias):
        """
        Converts the phase bias to range [-2pi - 2pi].

        Args:
            phase_bias: (float) the phase bias for the joint/oscillator in any range
        
        Returns:
            (float) the correct, adjusted phase bias of the oscillator
        """
        return (CPGParameterHandler.sigmoid(phase_bias) * 4*np.pi) - 2*np.pi

class CPGParameterHandlerMAPElites:
    """
    Scales the output values of oscillators/nerons to the correct range.
    To be used with MAP-Elites which outputs values in the range [0,1]
    """ 

    def scale_intrinsic_amplitudes(intrinsic_amplitudes):
        """
        Converts the intrinsic amplitudes to the correct range.

        Args:
            intrinsic_amplitudes: (list) of the intrinsic amplitudes of the oscillators

        Returns:
            (list) of the scaled intrinsic amplitudes of the oscillators
        """
        intrinsic_amplitudes = np.reshape(intrinsic_amplitudes, (6,2))
        """
        ## Joint 1 (0) - coxa
        [-100, 100] degrees
        [-1.74533,1.74533] radians

        ## Joint 2 (1) - femur
        [-150, 130] degrees
        [-2.61799, 2.26893] radians

        ## Joint 3 (2) - tibia
        [-170, 130] degrees
        [-2.96706, 2.26893] radians
        """
        for leg in range(6):
            for joint in range(2):
                intrinsic_amplitudes[leg][joint] = CPGParameterHandler.scale_intrinsic_amplitude(intrinsic_amplitudes[leg][joint], joint)
        return intrinsic_amplitudes.flatten()

    def scale_intrinsic_amplitude(intrinsic_amplitude, joint):
        """
        Converts the intrinsic amplitude to the correct range for the given joint.

        Args:
            intrinsic_amplitude: (float) the raw intrinsic amplitude for the joint/oscillator
            joint: (int) the joint (0/1/2)
        
        Returns:
            (float) the correct, adjusted intrinsic amplitude of the oscillator
        """
        if joint==0:
            scaled_intrinsic_amplitude = (intrinsic_amplitude * 3.49066) - 1.74533
            if scaled_intrinsic_amplitude > 1.74533:
                return 1.74533
            elif scaled_intrinsic_amplitude < -1.74533:
                return -1.74533
            return scaled_intrinsic_amplitude
        if joint==1:
            scaled_intrinsic_amplitude = (intrinsic_amplitude * 4.88692) - 2.61799
            if scaled_intrinsic_amplitude > 2.26893:
                return 2.26893
            elif scaled_intrinsic_amplitude < -2.61799:
                return -2.61799     
            return scaled_intrinsic_amplitude
        if joint==2:
            scaled_intrinsic_amplitude = (intrinsic_amplitude * 5.23599) - 2.96706
            if scaled_intrinsic_amplitude > 2.26893:
                return 2.26893
            elif scaled_intrinsic_amplitude < -2.96706:
                return -2.96706
            return scaled_intrinsic_amplitude

    def sigmoid_intrinsic_amplitudes(intrinsic_amplitudes):
        """
        Converts the intrinsic amplitudes to between 0-1.

        Args:
            intrinsic_amplitudes: (float) the intrinsic amplitudes for the joint/oscillator
        
        Returns:
            (float) the correct, adjusted intrinsic amplitudes between [0-1]
        """
        for leg in range(6):
            intrinsic_amplitudes[2*leg] =    (intrinsic_amplitudes[2*leg] + 1.74533) / 3.49066
            intrinsic_amplitudes[2*leg+1] =  (intrinsic_amplitudes[2*leg+1] + 2.61799) / 4.88692
        return intrinsic_amplitudes

    def scale_phase_biases(phase_biases_flat):
        """
        Converts the phase biases to the correct range.

        Args:
            phase_biases: (list) of the phase biases of the oscillators in range [0,1]

        Returns:
            (list) of the scaled phase biases of the oscillators
        """
        phase_biases = [[[],[],],[[],[],],[[],[],],[[],[],],[[],[],],[[],[],],]
        phase_biases_flat = np.reshape(phase_biases_flat, (6,24))
        for leg in range(6):
            leg_biases = np.reshape(phase_biases_flat[leg], (2,12))
            for joint in range(2):
                other_leg_biases = np.reshape(leg_biases[joint], (6,2))
                for other_leg in range(6):
                    leg_values = []
                    for other_joint in range(2):
                        # phase_biases[leg][joint][other_leg][other_joint] = other_leg_biases[other_leg][other_joint]
                        raw_value = other_leg_biases[other_leg][other_joint]
                        leg_values.append(CPGParameterHandlerMAPElites.scale_phase_bias(raw_value) if raw_value is not None else 0.0)
                    phase_biases[leg][joint].append(leg_values)
        
        return phase_biases
    
    def scale_phase_bias(phase_bias):
        """
        Converts the phase bias from range in [0-1] to range [-2pi - 2pi].

        Args:
            phase_bias: (float) the phase bias for the joint/oscillator in range [0-1]
        
        Returns:
            (float) the correct, adjusted phase bias of the oscillator
        """
        scaled_phase_bias = (phase_bias * 4*np.pi) - 2*np.pi
        if scaled_phase_bias > 2*np.pi:
            return 2*np.pi
        elif scaled_phase_bias < -2*np.pi:
            return -2*np.pi
        else:
            return scaled_phase_bias

    def sigmoid_phase_bias(phase_bias):
        """
        Converts the phase bias from range in [-2pi - 2pi] to range [0-1].

        Args:
            phase_bias: (float) the phase bias for the joint/oscillator in range [-2pi - 2pi]
        Returns:
            (float) the correct, adjusted phase bias of the oscillator [0-1]
        """
        return (phase_bias + 2*np.pi) / (4*np.pi)

    def convert_non_mapelites_parameters(x_non01):
        """Takes CPG parammeters and converts parameters into range [0-1]

        Args:
            The parameters output by MBOA/stored in a map that arent in range 0-1

        Returns:
            Scaled parameters in range [0-1]
        """
        intrinsic_amplitudes = x_non01[:12]
        intrinsic_amplitudes = CPGParameterHandlerMAPElites.sigmoid_intrinsic_amplitudes(intrinsic_amplitudes.astype(float))
        
        phase_biases = x_non01[12:]
        x=np.concatenate((intrinsic_amplitudes,phase_biases))
        return x

if __name__ == '__main__':
    # do any debugging here
    pass