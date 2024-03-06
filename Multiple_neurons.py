import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import pandas as pd
from scipy.interpolate import make_interp_spline ##new import

# Set simulation parameters
Vthresh = -50   #mV, AP threshold voltage
Vreset  = -70   #mV resting potential (going to value after spike)
Vspike  =  -50   #mV
Rm      =  10   #MOhms lower Resistance = lower spikes
tau     =  10   #ms
dt      =  1 #ms, time bins. SHOULD GO DOWN????
counter =  0
i = 0 #counting the number of iterations
time = 10000*4 #ms (run for 40 seconds)
Se = 0.01 #strength of spike
r = 30/1000 #constant (spikes/ms) asked for frequency of 30Hz
counter_poisson = 0 #counting number of spikes
tau_e = 5 #excitatory conductance time constant
Ve = 0 #excitatory reversal potential

#Sine funciton variables

I_max =  1.8 #external current
I_min =  1.7 #external current
I_mid = ((I_max - I_min)/2) + I_min
period = 4000 #in ms = 4 seconds

timeVector = np.arange(0, time, dt) #creating time vector of intervals of size dt
neuron_num = 1 #10 neurons being simulated
neuron_arr = np.zeros((neuron_num, len(timeVector))).astype(int) #2D array
neuron_volt = np.zeros((neuron_num, len(timeVector))) # Creates a placeholder for our voltages that is the same size as timeVector

# Creates a placeholder for the external stimulation vector.
# It is also the same size as the time vector.
stimVector = np.zeros(len(timeVector))

#sine function. 1 cycle every 4 seconds. Oscilating current
stimVector[0:] = ((I_max - I_min)/2) * (np.sin((timeVector/(period/(2*np.pi))))) + I_mid
# Euler---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

voltageVector_neurons = np.zeros((neuron_num, len(timeVector))).astype(int) #vector containing the voltages for each neuron at each time bin
voltageVector_neurons_avg = np.zeros(len(timeVector)) #vector containing the average voltage across neurons
spike_time_matrix = []
#WAYS TO IMPROVE:---------------------------------------------------------------------------------------------------------------------------------------------
#run neurons in parallel (only when not talking to eachother)
#compute external stimulus as you go. Don;t need to hold onto whole vector
#MAYBE: Hold onto time that the neuron spiked.


for n in range(neuron_num): #for each neuron
    counter_poisson = 0 #number of poisson spikes equals 0
    counter = 0
    voltageVector = np.zeros(len(timeVector)) #create a new voltage vector filled with all 0s
    voltageVector[0] = Vreset  # Set the initial voltage to be equal to the resting potential
    geVector = np.zeros(len(timeVector))  # amount spike jumps, excitatory conductance
    spikeVector = np.zeros(len(timeVector)).astype(int)  # holding 1s for spike and 0 for no spike at given interval
    spike_time_vector = []
    startTime = datetime.datetime.now()

    # This line initiates the loop. "S" counts the number of loops.
    # We are looping for 1 less than the length of the time vector
    # because we have already calculated the voltage for the first
    # iteration.
    for S in range(len(timeVector) - 1):

        # update Vinf using purple sheet. WHAT IS PURPLE SHEET? I HAVE FORGOTTEN
        Vinf = ((Rm * stimVector[S]) / tau) + (Vreset - voltageVector[S]) / tau + (Rm * geVector[S] * (
                    Ve - voltageVector[S])) / tau  # ((E + I*R)/tau) + (-v[i]/tau) #differential equation
        #MAX---------------------------------------------------------------------------------------------------
        #relabel vinf to what?

        #Vinf is the value of the membrane potential at the next time step. It is the sum of three components:

        #The first term represents the influence of the external stimulus (Istim) on the membrane potential. It's essentially the effect of the applied current over time.

        #The second term represents the influence of the difference between the reset potential (Vreset) and the current membrane potential (Vprevious).
        #This term introduces a form of negative feedback, pushing the membrane potential back towards the reset potential.

        #The third term represents the influence of the excitatory conductance (ge) on the membrane potential.
        # It accounts for the effect of excitatory synaptic inputs. Ve is the excitatory reversal potential.

        #Tau: Membrane time constant, representing the time it takes for the membrane potential to change in response to applied current.
        #Ve: Excitatory reversal potential, the equilibrium potential for excitatory inputs.

        voltageVector[S + 1] = Vinf * dt + voltageVector[S]  # v[i+1] = v[i] + dt *Vinf
        #Euler step-- equation of a line

        # upadate conductance decay as if no poisson input spike
        #conductance of external input into the neuron
        #MAX: WHY DO WE DO THIS?
        geVector[S + 1] = geVector[S] * np.exp(-dt / tau_e)

        # This 'if' condition states that if the next voltage is greater than
        # or equal to the threshold, then to run the next section
        if voltageVector[S + 1] >= Vthresh:
            # This states that the next voltage vector will be the Vspike value
            voltageVector[S + 1] = Vspike #neuron spike if above threshold

            # This 'if' statement checks if we are already at Vspike (this is
            # another way we can be above Vthresh)
            if voltageVector[S] == Vspike: #if the current voltage is a spike
                spikeVector[S] = 1  # add a 1 to indicate a spike in spikeVector
                # upadate conductance as if no spike
                #geVector[i] = geVector[i-1] * np.exp(-dt/curTime) #MAX: WHAT IS THE PURPOSE OF THIS?
                counter += 1

                # Set the next voltage equal to the reset value
                voltageVector[S + 1] = Vreset #MAYBE IMPLEMENT A "vRefractory--" less than Vreset

        curTime = timeVector[S]
        while curTime < timeVector[S + 1]: #MAX: I don't understand the purpose of this while loop?
            # responsible for simulating the arrival of excitatory spikes:
            #Generates a random waiting time according to a Poisson process

            curTime = -np.log(np.random.uniform(0, 1)) / r + timeVector[S]  # r in ms Calculate wait time. PROPOSED SPIKE TIME
            # generates a random number from an exponential distribution with a mean of 1/r. This random number represents the time until the next spike
            #If the generated curTime is within the current time interval (timeVector[S] to timeVector[S + 1]),
            # it updates the excitatory conductance geVector by adding the strength of a spike Se.
            # This simulates the effect of an excitatory spike arriving at the neuron during that time interval

            if curTime <= timeVector[S + 1]:  # see if wait time is in interval
                # if so, update conductance
                geVector[S + 1] = geVector[S] + Se
                # spike will occur
                # If the generated curTime is within the current time interval (timeVector[S] to timeVector[S + 1]),
                # it updates the excitatory conductance geVector by adding the strength of a spike Se.
                # This simulates the effect of an excitatory spike arriving at the neuron during that time interval
        #INEFFICENT
        neuron_arr[n] = spikeVector
        neuron_volt[n] = voltageVector
        voltageVector_neurons[n, ] = voltageVector

#END LOOP

for i in range(len(voltageVector)):
    voltageVector_neurons_avg[i, ] = voltageVector_neurons[:, i].mean()
    # voltageVector_neurons_avg[i] = np.mean(voltageVector_neurons, axis = 1)

print(spike_time_vector)
plt.plot(timeVector[:5000], voltageVector_neurons_avg[:5000])
#PLOT FULL THING ALONG WITH SINE WAVE
plt.show()
print(sum(spikeVector))

# Vector of specific times of spikes
for neuron in range(neuron_num):
    spikeVector = neuron_arr[neuron]
    spike_time_Vector = np.zeros(sum(spikeVector))
    z = 0  # initialize z to begin at index #MAX: Rename? What is z?
    for y in range(len(spikeVector)):  # grab index of spike and compare to timeVector

        if spikeVector[y] == 1:
            spike_time_Vector[z] = timeVector[y]  # add spike times in this vector
            z = z + 1
    print(spike_time_Vector)

    #sum across rows of 1s and 0s. To see how many spikes. Maximum number of spikes is how wide matrix needs to be
    #
    plt.plot(spike_time_Vector, np.zeros(len(spike_time_Vector)) + neuron, marker='|', linestyle='none')
    plt.xlabel("Time in ms")
    plt.ylabel("Neuron")
    plt.title("Spike Activity of 10 Neurons")
plt.show()

#TESTING--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2: Test

# #can likely remove this part-- everything that gets initialized in the loop does not really need to exist here
# max_bin = 500
# min_bin = 0
# total_spikes_bin = np.zeros(80)  # number of bins
# g = 0
# total_spikes_temp = 0
# q = 0
# bin_size = 500
# time_bin_vec = np.arange(0, time, bin_size)
# n = 10  # number of elements per bin
# fr_bin = np.zeros((neuron_num, 8))
# average_fr_bin = np.zeros(80)
#
# for neuron in range(neuron_num): #For each simulated neuron
#     spikeVector = neuron_arr[neuron] #grab it's spikeVector
#     spike_time_Vector = np.zeros(sum(spikeVector)) #create a new spike time vector for it
#     spikeCount = 0  # initialize z to begin at index 0 #MAX: BETTER VARIABLE NAMES PLZ
#     max_bin = 500
#     min_bin = 0
#     total_spikes_bin = np.zeros(80)  # number of bins
#     g = 0
#     total_spikes_temp = 0
#     q = 0
#
#     for y in range(len(spikeVector)):  # grab index of spike and compare to timeVector
#
#         if spikeVector[y] == 1: #if there is a spike
#             spike_time_Vector[spikeCount] = timeVector[y]  # add spike times in this vector
#             z = z + 1
#
#     for entry in spike_time_Vector:
#         print(entry)
#         #MAX: I DON'T UNDERSTAND THIS IF/ELSE STATEMENT. WHAT IS IT DOING?
#         if entry <= max_bin and entry >= min_bin:
#             total_spikes_temp += 1
#             q = q + 1 #WHAT IS Q
#         else:
#             total_spikes_bin[g] = total_spikes_temp
#
#             g = g + 1 #WHAT IS G AND WHAT DO MIN/MAX bins do?
#             max_bin = max_bin + 500  # change to variable to make reproducible
#             min_bin = min_bin + 500
#             total_spikes_temp = 1  # equal to 1 because entry must be placed in next time bin
#
#     firing_rate_bin = total_spikes_bin * 2
#     average_fr_bin = average_fr_bin + firing_rate_bin
#     plt.plot(time_bin_vec, firing_rate_bin)  # keep this
#     plt.xlabel("Time in ms")
#     plt.ylabel("Firing Rate (Hz)")
#     plt.title("Firing Rate of Ten Neurons")
#
#     # plt.plot(spike_time_Vector, np.zeros(len(spike_time_Vector))+neuron, marker = '|', linestyle = 'none')
#     # Average spikes at each phase and make histogram of averages at each phase. Bins of 4 seconds
#
#     start = 0
#     for i in range(8):
#         fr_bin[neuron, i] = np.mean(firing_rate_bin[start::7])  # grabbing spikes at same phase
#         start += 1
#
# # plot sine curve next to this
# plt.figure()
# plt.plot(time_bin_vec, average_fr_bin / neuron_num)  # keep this
# plt.xlabel("Time in ms")
# plt.ylabel("Firing Rate (Hz)")
# plt.title("Firing Rate of a Neuron")
# plt.show

# Step 2: Test

max_bin = 500
min_bin = 0
total_spikes_bin = np.zeros(80)  # number of bins
g = 0
total_spikes_temp = 0
q = 0
bin_size = 500
time_bin_vec = np.arange(0, time, bin_size)
n = 10  # number of elements per bin
fr_bin = np.zeros((neuron_num, 8))
average_fr_bin = np.zeros(80)

for neuron in range(neuron_num):
    spikeVector = neuron_arr[neuron]
    spike_time_Vector = np.zeros(sum(spikeVector))
    z = 0  # initialize z to begin at index 0
    max_bin = 500
    min_bin = 0
    total_spikes_bin = np.zeros(80)  # number of bins
    g = 0
    total_spikes_temp = 0
    q = 0

    for y in range(len(spikeVector)):  # grab index of spike and compare to timeVector

        if spikeVector[y] == 1:
            spike_time_Vector[z] = timeVector[y]  # add spike times in this vector
            z = z + 1

    for entry in spike_time_Vector:
        if entry <= max_bin and entry >= min_bin:
            total_spikes_temp = total_spikes_temp + 1
            q = q + 1
        else:
            total_spikes_bin[g] = total_spikes_temp

            g = g + 1
            max_bin = max_bin + 500  # change to varible to make reproducible
            min_bin = min_bin + 500
            total_spikes_temp = 1  # equal to 1 because entry must be placed in next time bin

    firing_rate_bin = total_spikes_bin * 2
    average_fr_bin = average_fr_bin + firing_rate_bin
    plt.plot(time_bin_vec, firing_rate_bin)  # keep this
    plt.xlabel("Time in ms")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Firing Rate of Ten Neurons")

    # plt.plot(spike_time_Vector, np.zeros(len(spike_time_Vector))+neuron, marker = '|', linestyle = 'none')

    # Average spikes at each phase and make histogram of averages at each phase. Bins of 4 seconds

    start = 0
    for i in range(8):
        fr_bin[neuron, i] = np.mean(firing_rate_bin[start::7])  # grabbing spikes at same phase
        start = start + 1

plt.show()
# plot sine curve next to this
plt.figure()
plt.plot(time_bin_vec, average_fr_bin / neuron_num)  # keep this
plt.xlabel("Time in ms")
plt.ylabel("Firing Rate (Hz)")
plt.title("Firing Rate of a Neuron")
plt.show()