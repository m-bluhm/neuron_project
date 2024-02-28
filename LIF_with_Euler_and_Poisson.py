import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Set simulation parameters
Vthresh = -50   #mV
Vreset  = -70   #mV resting potential
Vspike  =  20   #mV spike voltage
Rm      =  10   #MOhms lower Resistance = lower spikes
tau     =  10   #ms
dt      =  1 #ms
counter =  0
i = 0 #counting the number of iterations
time = 10000*4 #ms (run for 10 seconds)
spikeStrength = 0.01 #strength of spike
r = 30/1000 #constant (spikes/ms) asked for frequency of 30Hz
counter_poisson = 0 #counting number of spikes
counter_neuron = 0
tau_e = 5 #excitatory conductance time constant
Ve = 0 #excitatory reversal potential

#Sine funciton variables

I_max =  1.8 #external current
I_min =  1.7 #external current
I_mid = ((I_max - I_min)/2) + I_min
period = 4000 #in ms

timeVector = np.arange(0, time, dt) #creating time vector of intervals of size dt. TIME BINS

geVector = np.zeros(len(timeVector)) #amount spike jumps, excitatory conductance. CONDUCTANCE VECTOR
spikeVector = np.zeros(len(timeVector)) #holding 1s for spike and 0 for no spike at given interval
xVector = np.random.uniform(0,1,len(timeVector)) #create xVector of same length of timeVector to store uniform values between 0 and 1

# Creates a placeholder for our voltages that is the same size as timeVector
voltageVector = np.zeros(len(timeVector))

# Creates a placeholder for the external stimulation vector.
# It is also the same size as the time vector.
stimVector = np.zeros(len(timeVector))

#sine function. 1 cycle every 4 seconds
stimVector[0:] = ((I_max - I_min)/2) * (np.sin( ( timeVector/((period)/(2*np.pi)))  ) ) + I_mid

# Set the initial voltage to be equal to the resting potential
voltageVector[0] = Vreset

# Euler
# This line initiates the loop. "S" counts the number of loops.
# We are looping for 1 less than the length of the time vector
# because we have already calculated the voltage for the first iteration.
for S in range(len(timeVector) - 1):

    # update Vinf using purple sheet: what is PURPLE SHEET?
    Vinf = ((Rm * stimVector[S]) / tau) + (Vreset - voltageVector[S]) / tau + (Rm * geVector[S] * (
                Ve - voltageVector[S])) / tau  # ((E + I*R)/tau) + (-v[i]/tau) #differential equation
    #RELABEL: derivative

    voltageVector[S + 1] = Vinf * dt + voltageVector[S]  # v[i+1] = v[i] + dt *Vinf

    # upadate conductance decay as if no poisson spike-- stimulation current
    #assume there is no spike always

    geVector[S + 1] = geVector[S] * np.exp(-dt / tau_e)

    # This 'if' condition states that if the next voltage is greater than
    # or equal to the threshold, then to run the next section
    if voltageVector[S + 1] >= Vthresh:
        # This states that the next voltage vector will be the Vspike value
        voltageVector[S + 1] = Vspike

        # This 'if' statement checks if we are already at Vspike (this is
        # another way we can be above Vthresh
        if voltageVector[S] == Vspike:
            # print(S) This prints the timing of spikes
            spikeVector[S] = 1  # add a 1 to indicate a spike in spikeVector

            # upadate conductance as if no spike
            # geVector[i] = geVector[i-1] * np.exp(-dt/j)
            counter_neuron += 1

            # Set the next voltage equal to the reset value
            voltageVector[S + 1] = Vreset

    j = timeVector[S] #j = current_time_bin
    while j < timeVector[S + 1]:

        j = -np.log(np.random.uniform(0, 1)) / r + timeVector[S]  # r in ms Calculate wait time
        #predicted poisson spike time

        if j <= timeVector[S + 1]:  # see if wait time is in interval
            # if so, update conductance
            geVector[S + 1] = geVector[S] + spikeStrength #ultimatley interested in updating conductance vector
            counter_poisson += 1
            # spike will occur

print(sum(spikeVector))

#Vector of specific times of spikes
spike_time_Vector = np.zeros(counter_poisson)
z = 0 #initialize z to begin at index 0
for y in range(len(spikeVector)): #grab index of spike and compare to timeVector
    if spikeVector[y] == 1:
        temp_index = y
        spike_time_Vector[z] = timeVector[y] #add spike times in this vector
        z = z+ 1


#Voltage and Conductance Plots
plt.figure()
plt.plot(timeVector[:5000], voltageVector[:5000]) # This plots the voltage (y-axis) as a function of time (x-axis)
plt.title('Voltage versus time') # This sets the title
plt.figure()
plt.plot(timeVector, geVector) # This plots the voltage (y-axis) as a function of time (x-axis)
plt.title('Conductance versus time') # This sets the title
plt.show()

#Sine plot
plt.plot(timeVector, stimVector)
yVector = np.zeros(len(spike_time_Vector))
yVector[0:] = I_min
plt.plot(spike_time_Vector, yVector, '|', 'none')
plt.show()

print(stimVector) # each entry is updating with time
