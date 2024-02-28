# Import essential libraries
import numpy as np
import matplotlib.pyplot as plt

# Set simulation parameters
Vthresh = -50   #mV. Voltage threshold-- at what voltage does a spike occur
Vreset  = -70   #mV. Reset voltage-- after a spike happens, what is the "baseline" membrane potential?
Vspike  =  20   #mV. Spike peak: When a spike happens, what is the maximum voltage it spikes to?
Rm      =  10   # Membrane Resistance, MOhms lower Resistance = lower spikes
tau     =  10   #ms
dt      =  0.1  #ms. Time step
counter =  0
i = 0 #counting the number of iterations

# Creates a vector of time points from 0 to 499 ms in steps of dt=0.01ms
timeVector = np.linspace(0, 500, 5001)

print(timeVector[3])

# Creates a placeholder for our voltages that is the same size as timeVector
voltageVector = np.zeros(len(timeVector))

voltVec = np.zeros(len(timeVector))

# Creates a placeholder for the external stimulation vector.
# It is also the same size as the time vector.
stimVector = np.zeros(len(timeVector))

#print(timeVector[4990:5001])

# Sets the external stimulation to 2.0001 nA for the first 500 ms
stimVector[0:] = 3.0001

#all entries are the same
#2.0001nA for V = -70mV
#1.01nA for V = -60mV

# Set the initial voltage to be equal to the resting potential
voltageVector[0] = Vreset

for S in range(len(timeVector) - 1):
    #derivaive evaluated at previous time step
    Vinf = ((Vreset + Rm * stimVector[S]) / tau) + (-voltageVector[S] / tau)  # ((E + I*R)/tau) + (-v[i]/tau) #WHAT DOES THIS EQUATION MEAN/DO???????

    voltageVector[S + 1] = Vinf * dt + voltageVector[S]  # v[i+1] = v[i] + dt *Vinf

    # The next voltage value is is equal to where we are going (Vinf)
    # plus the product of the different between the present voltage and
    # Vinf (how far we have to go) and e^-t/tau (how far we are going
    # in each step)

    # voltageVector[S+1] = Vinf + (voltageVector[S]-Vinf)*np.exp(-dt/tau)

    # This 'if' condition states that if the next voltage is greater than
    # or equal to the threshold, then to run the next section
    if voltageVector[S + 1] >= Vthresh:
        # This states that the next voltage vector will be the Vspike value
        voltageVector[S + 1] = Vspike

        # This 'if' statement checks if we are already at Vspike (this is
        # another way we can be above Vthresh)
        if voltageVector[S] == Vspike:
            # print(S) This prints the timing of spikes
            voltVec[i] = 1  # add a 1 to indicate a spike in voltVec

            # Set the next voltage equal to the reset value
            voltageVector[S + 1] = Vreset

            # This will count the number of observed spikes so that spike count
            # rate may be calculated later
            counter += 1
    i += 1



# This sets the plot object
plt.figure()

# This defines that we are plotting into the top plot
plt.subplot(2,1,1) # two rows, one column, first graph

# Plots time on the x-axis and current on the y-axis
plt.plot(timeVector, stimVector)

# Labels the x-axis
plt.xlabel('Time in ms')

# Labels the y-axis
plt.ylabel('External current in nA')

# Titles the plot
plt.title('External Stimulation vs Time')

# This defines that we are plotting into the top plot
plt.subplot(2,1,2) # 2 rows, 1 column, 2nd graph

# Plots time on the x-axis and voltage on the y-axis
plt.plot(timeVector, voltageVector)

# Labels the x-axis
plt.xlabel('Time in ms')

# Labels the y-axis
plt.ylabel('Membrane potential in mV')
plt.show()