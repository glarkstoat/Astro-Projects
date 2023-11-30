#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, my_mod
from numba import jit # Speeds up calculations

#from mpl_toolkits.mplt3d import Axes3d # scatter nicht plot verwenden
#import matplotlib.animation as animation
#ax = plt.gca(proejction="3d")

start = time.time() # Starts the timer

# read input data file:
input_body = np.genfromtxt("initials.txt")
input = "initials.txt"

# These arrays contain data of the mass, position and velocity of every body for an arbitrary number of bodies
m = np.genfromtxt(input,usecols=(0))
r = np.genfromtxt(input,usecols=(1,2))
v = np.genfromtxt(input,usecols=(3,4))

# Computed time [s] = dt * n
dt = 1e+2  # length of timestep for numerical integration
n = 320000  # number of timesteps to compute

data = open('body_values.output', 'w')
# @jit(nopython=True, fastmath=True)

for x in range(n): # Solves the coupled ODEs via the Euler Method
    data.write(str(x*dt)) # Writes the timesteps to the file. This is not used in the calculations. Just for refrence
    for i in range(len(input_body)): # Checks the length of the array i.e the number of bodies
        dv = 0 # resets the change of velocity for every new index i (otherwise dv for every index i would be added)
        r[i] += v[i] * dt # Calculates the new position of the particular body
        for j in range(len(input_body)): # Necessary to compute dv
            if i != j: # Same indices are excluded (if unclear check coupled ODEs)
                dv += my_mod.vel_change(m[j], r[i], r[j], dt) # Sums the change of velocity
        v[i] += dv # Computes new velocity
        for h in range(2): # Writes the new x- and y-coordinates to the file, for 3d it would be range(3)
            data.write(" "+str(r[i][h])) # Writes the components of r to the output-file
    data.write("\n") # New row in output-file
    
    # Displays the current status of the computation
    if (10*x/n) % 1 == 0 and (x/n) != 0: # Only works if (x/n) > 1 (otherwise 0.6 % 0.1 != 0, also 0.3 % 0.1 != 0, no idea why!)
        print(str(round(100*x/n)) + "% Done")
        
# Prints the elapsed time of computation
data.close(); end = time.time()
print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

#%%
output = 'body_values.output'

sns.set_style('darkgrid')
sns.set(color_codes=True)
sns.set_context('talk', font_scale=0.5)

for i in range(1, 2*len(input_body)+1, 2):  # plots the positions of the bodies  
    plt.plot(np.genfromtxt(output,usecols=(int(i))), np.genfromtxt(output,usecols=int((i+1))), label='Mass '+str(round((i+1)/2))+': %s' %m[int(i/2)])
    plt.legend()
plt.tight_layout()
plt.show()
"""
fig = plt.figure()

# animation function.  This is called sequentially
def animate(i):
    #fig.clear()
    #plt.axes()
    for k in range(1, 2*len(input_body)+1, 2):  # plots the positions of the bodies one by one  
        plt.plot(np.genfromtxt(output)[i][k], np.genfromtxt(output)[i][k+1], '*', color='b', markersize=1)        

body_animation = animation.FuncAnimation(fig, animate, interval=0.01) # call the animation

#body_animation.save('basic_animation.mp4', writer=writer)

plt.show()
"""
# %%
