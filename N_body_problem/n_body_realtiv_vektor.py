import numpy as np
import matplotlib.pyplot as plt
import time, my_mod
#from mpl_toolkits.mplt3d import Axes3d # scatter nicht plot verwenden
#import matplotlib.animation as animation
#ax = plt.gca(projection="3d")

start = time.time() # Starts the timer

# read input data file:
input_body = np.genfromtxt("initials_3d.txt")
input = "initials_3d.txt"

# These arrays contain data of the mass, position and velocity of every body for an arbitrary number of bodies
m = np.genfromtxt(input,usecols=(0))
r = np.genfromtxt(input,usecols=(1,2,3))
v = np.genfromtxt(input,usecols=(4,5,6))

print("WTF!")

# Sum of all masses
m_sum = 0
for i in range(2): # range(len(input_body)) for n-bodies
    m_sum += m[i]
    print("Still?")

x_r = r[1] - r[0]  
x_s = (1/m_sum) * (m[0]*r[0] + m[1]*r[1])

print




























dt = int(5e+3)  # length of timestep for numerical integration, doesn't affect the computation-time
n = int(5e+4)  # number of output timesteps to compute, directly affects the computation-time

data = open('body_values_3d.output', 'w')

for x in range(n): # Solves the coupled ODEs via the Euler Method
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
    if (x/n) % 0.05 == 0 and (x/n) != 0:  
        print(str(100 * (x/n)) + " " + "% Done!")

data.close(); end = time.time()
print("finished"); print(end-start) # Prints the elapsed time of computation

output = 'body_values_3d.output'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1, 3 * len(input_body) + 1, 3):  # plots the positions of the bodies  
    ax.plot(np.genfromtxt(output,usecols = int(i)), np.genfromtxt(output,usecols = int(i+1)), np.genfromtxt(output,usecols = int(i+2)))
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')

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
    #fig.clear()
    #plt.axes()
    for k in range(1, 2*len(input_body)+1, 2):  # plots the positions of the bodies one by one  
        plt.plot(np.genfromtxt(output)[i][k], np.genfromtxt(output)[i][k+1], '*', color='b', markersize=1)        

body_animation = animation.FuncAnimation(fig, animate, interval=0.01) # call the animation

#body_animation.save('basic_animation.mp4', writer=writer)

plt.show()
"""