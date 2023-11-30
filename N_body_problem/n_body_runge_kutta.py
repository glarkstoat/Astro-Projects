import numpy as np
import matplotlib.pyplot as plt
import time, my_mod
import matplotlib.animation as animation
"""
#from mpl_toolkits.mplt3d import Axes3d # scatter nicht plot verwenden
#ax = plt.gca(proejction="3d")
"""

planets = {0 : "Sun" , 1 : "Mercury", 2 : "Venus", 3 : "Earth", 4 : "Mars", 5 : "Jupiter", 6 : "Saturn", 7 : "Uranus", 8 : "Neptune", 9 : "Pluto"}

start = time.time() # Starts the timer

# read input data file:
input_body = np.genfromtxt("initials.txt")
input = "initials.txt"

# These arrays contain data of the mass, position and velocity of every body for an arbitrary number of bodies
m = np.genfromtxt(input,usecols=(0))
r = np.genfromtxt(input,usecols=(1,2))
v = np.genfromtxt(input,usecols=(3,4))

dt = 1e+2  # length of timestep for numerical integration
n = 3*10**5  # number of output timesteps to compute

data = open('body_values_rk.output', 'w')

for x in range(n): # Solves the coupled ODEs via the Euler Method
    data.write(str(2*x*dt))
    for i in range(len(input_body)): # Checks the length of the array i.e the number of bodies
        dv = 0 # resets the change of velocity for every new index i (otherwise dv for every index i would be added)
        r[i] += v[i] * dt / 2 # Computes the HALF-STEP according to the Runge-Kutta 2nd Order method
        for j in range(len(input_body)): # Necessary to compute dv
            if i != j: # Same indices are excluded (if unclear check coupled ODEs)
                dv += my_mod.vel_change(m[j], r[i], r[j], dt) # Sums up every component of dv for a given index i
        v[i] += dv # Calculates the new v[i]
        r[i] += v[i] * dt # Computes the FULL-STEP according to the Runge-Kutta 2nd Order method
        r[0] = 0
        if -5*10**7 < r[i][1] < 5*10**7 and r[i][0] > 0 and i!=0:
            print(int(2*x*dt), "s : ", planets[i], "completed orbit.")
        for j in range(len(input_body)): # Necessary to compute dv
            if i != j: # Same indices are excluded (if unclear check coupled ODEs)
                dv += my_mod.vel_change(m[j], r[i], r[j], dt) # Sums up every component of dv for a given index i
        v[i] += dv # Again, calculates the new v[i]
        for h in [0,1]:
            data.write(" "+str(r[i][h])) # Writes only the components 
    data.write("\n")
    
    # Displays the current status of the computation
    if (10*x/n) % 1 == 0 and (x/n) != 0: # Only works if (x/n) > 1 (otherwise 0.6 % 0.1 != 0, also 0.3 % 0.1 != 0, no idea why!)
        print(str(round(100*x/n)) + "% Done")

data.close(); end = time.time()
print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

output = 'body_values_rk.output'

for i in range(1, 2*len(input_body)+1, 2):  # plots the positions of the bodies  
    plt.plot(np.genfromtxt(output,usecols=(int(i))), np.genfromtxt(output,usecols=int((i+1))), label='Mass '+str(round((i+1)/2))+': %s' %m[int(i/2)])
    plt.legend()
plt.show()

"""
fig = plt.figure()

# animation function
def animate(i):
    #fig.clear()
    #plt.axes()
    for k in range(1, 2*len(input_body)+1, 2):  # plots the positions of the bodies one by one  
        plt.plot(np.genfromtxt(output)[i][k], np.genfromtxt(output)[i][k+1], '*', color='b')        

body_animation = animation.FuncAnimation(fig, animate, interval=0.001) # call the animation

#body_animation.save('basic_animation.mp4', writer=writer)

plt.show()
"""
