






#subdivide time into short intervals of the same duration (dt). r = 20
#generate a sequence of random numbers (x[i]) uniformly distributed between 0 and 1
#if x[i] is greater than rst there will be a spike
# maybe set st equal to 1 millisecond
# simulation time should be 1000ms and frequency is 20HZ (there should be 20 spikes)

##Goal: Create list of spikes and spike times via Poisson Process


import numpy as np
import matplotlib.pyplot as plt
import random

dt = 1 #1 ms = 0.001 seconds- maybe decrease (given by poisson article): //MAX: what does this mean?
r = 0.02 #constant (spikes/ms) asked for frequency of 20Hz (20/1000 = 0.2). //MAX: Target firing rate of the poisson process
counter = 0 #counting number of spikes
time = 1000 #ms
timeVector = np.arange(0, time, dt) #creating time vector of intervals of size dt
spikeVector = np.zeros(len(timeVector)) #holding 1s for spike and 0 for no spike at given interval

#create xVector of same length of timeVector to store random values between 0 and 1. Will be compared to threshold value
xVector = np.random.uniform(0, 1, len(timeVector))

counter = 0  # reset counter
for i in range(len(xVector)):

    if xVector[i] <= r * dt:#WHAT DOES r * dt mean/do?. PROABABILITY THAT THERE WILL BE A SPIKE IN THAT TIME BIN
        spikeVector[i] = 1
        counter += 1
#vector of spike times.
#after this step, we have populated spikeVector with a number of "1s" equal to the number of randomly generated values
#greater than our threshold r * dt. 1s represent a spike, 0's represent no spike

spike_time_Vector = np.zeros(counter)   #create vector of spike times equal to length of spike count
                                        #This stores at which times (in milliseconds) a spike occured
z = 0 #initialize z to begin at index 0

for y in range(len(spikeVector)): #grab index of spike and compare to timeVector

    if spikeVector[y] == 1:
        temp_index = y
        spike_time_Vector[z] = timeVector[y] #add spike times in this vector
        z = z + 1 #I don't understand this section; What is z being used for?
        #print("%.1f" % timeVector[y]) ms

#Calculate ISI
isi_vec = np.diff(spike_time_Vector)
#print("ISI:")
#print(isi_vec)


print("The average isi is:")
print(np.mean(isi_vec)) #ms

print("The std is:")
print(np.std(isi_vec)) #ms
#Check that the time between spikes (ISI) is exponentially distributed – the mean(ISI) = std(ISI).

plt.hist(isi_vec)
plt.title('ISI Distribution')
plt.show()

########## Part 2: Conductance Figure
##Goal: Compute and Plot Excitatory Conductance
spikeStrength = 1  # strength of spike
decay = 5  # ms time to decay

geVector = np.zeros(len(timeVector))  # amount spike jumps, excitatory conductance
#This is where my understanding breaks dowm, I'm not sure what's going on here even though I understand the figure
for i in range(1, time, dt):
    if spikeVector[i] == 1:
        geVector[i] = geVector[i - 1] + spikeStrength

    else:
        geVector[i] = geVector[i - 1] * np.exp(-dt / decay)

yVector = np.zeros(len(spike_time_Vector)) #for plotting
# This sets the new plot object
plt.figure()

# This plots the voltage (y-axis) as a function of time (x-axis)
plt.plot(timeVector, geVector)
plt.plot(spike_time_Vector, yVector, 'o', 'none')
plt.xlabel('Time in ms') # This labels the x-axis
plt.ylabel("Spike Strength")
plt.title('Excitatory Conductance') # This sets the title
plt.show()