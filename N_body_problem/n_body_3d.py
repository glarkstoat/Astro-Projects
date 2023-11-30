import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, my_mod

#import matplotlib.animation as animation
#ax = plt.gca(projection="3d")

# Starts the timer
start = time.time() 

# read input data file:
input_body = np.genfromtxt("initials_3d.txt")
input = "initials_3d.txt"

# These arrays contain data of the mass, position and velocity of every body for an arbitrary number of bodies
m = np.genfromtxt(input,usecols=(0))
r = np.genfromtxt(input,usecols=(1,2,3))
v = np.genfromtxt(input,usecols=(4,5,6))

dt = int(8e+2)  # length of timestep for numerical integration, doesn't affect the computation-time
n = int(5e+5)  # number of output timesteps to compute, directly affects the computation-time

data = open('body_values_3d.output', 'w')

# Solves the coupled ODEs via the Euler Method
for x in range(n): 
    data.write(str(x*dt))
    for i in range(len(input_body)): # Checks the length of the array i.e the number of bodies
        dv = 0 # resets the change of velocity for every new index i (otherwise dv for every index i would be added)
        r[i] += v[i] * dt # Computes the new r
        for j in range(len(input_body)): # Necessary to compute dv
            if i != j: # Same indices are excluded (if unclear check coupled ODEs)
                dv += my_mod.vel_change(m[j], r[i], r[j], dt) # Sums up every component of dv for a given index i
        v[i] += dv # Computes new v
        for h in range(3):
            data.write(" "+str(r[i][h])) # Writes the components of r to the output-file
    data.write("\n") # New row in output-file

    # Displays the current status of the computation
    if (10*x/n) % 1 == 0 and (x/n) != 0: # Only works if (x/n) > 1 (otherwise 0.6 % 0.1 != 0, also 0.3 % 0.1 != 0, no idea why!)
        print(str(round(100*x/n)) + "% Done")

# Prints the elapsed time of computation
data.close(); end = time.time()
print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

output = 'body_values_3d.output'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1, 3 * len(input_body) + 1, 3):  # plots the positions of the bodies i.e every thrid column 
    ax.plot(np.genfromtxt(output,usecols = int(i)), np.genfromtxt(output,usecols = int(i+1)), np.genfromtxt(output,usecols = int(i+2)), label='Mass '+str(round((i+1)/3))+': %s' %m[int(i/3)])
    plt.legend()
#ax.set_xlabel('X-Axis')
#ax.set_ylabel('Y-Axis')
#ax.set_zlabel('Z-Axis')

"""
for i in range(1, 3*len(input_body)+1, 3):  # plots the positions of the bodies  
    plt.plot(np.genfromtxt(output,usecols=(int(i))), np.genfromtxt(output,usecols=int((i+1))), np.genfromtxt(output,usecols=int((i+2))))
    #plt.legend()
plt.show()
"""
# Animation of rotation of z-axis
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)
"""
"""
fig = plt.figure()

# animation function.  This is called sequentially
def animate(i):
    fig.clear()
    plt.axes()
    for k in range(1, 2*len(input_body)+1, 2):  # plots the positions of the bodies one by one  
        plt.plot(np.genfromtxt(output)[i][k], np.genfromtxt(output)[i][k+1], '*', color='b', markersize=1)        

body_animation = animation.FuncAnimation(fig, animate, interval=0.01) # call the animation

#body_animation.save('basic_animation.mp4', writer=writer)

plt.show()
"""