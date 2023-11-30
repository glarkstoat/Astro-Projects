#%%
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt
import seaborn as sns
import sys
import random 

sns.set_style('darkgrid')
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')
sns.set_palette("viridis")

# %%
def pi_Monte_Carlo(M):
    """ Calculates the numerical value of pi via Monte Carlo method """
    x = np.linspace(-10,10, M)
    results = np.empty((0,3)) # Initializes the results array
    radius = (x.max() - x.min()) / 2 # radius of the cicle

    for n in range(1, M+1):
        # Calculates pi for every n until n=M
        particles_in_the_circle = 0
        x_coord = np.random.choice(x, n, replace=False) # random x-coordinates
        y_coord = np.random.choice(x, n, replace=False) # random y-coordinates
        for i in range(n): # Iterates over all n particles
            if np.linalg.norm([x_coord[i], y_coord[i]]) <= radius:
                """ Checks if particle is in the circle """
                particles_in_the_circle += 1
        
        pi_approx = particles_in_the_circle / n * 4
        difference = pi_approx - np.pi

        results = np.vstack((results, np.array([pi_approx, difference, n])))
    
    return results

# %%

M = int(sys.argv[1]) # max. number of particles
result = pi_Monte_Carlo(M) # calculates the pi values for all n=range(1,M+1) values

# Plots of the result
plt.figure()
plt.title('Numeric Value of Pi - Monte Carlo', fontsize=12 , fontweight='bold')
plt.plot(result[:,2], result[:,0], lw=0.2, c='r')
plt.plot(range(1,result.shape[0]+1), [np.pi] * result.shape[0], lw=1,
                    ls='dotted', c='black', label='True value')
plt.xlabel("Number of particles n", fontweight='bold')
plt.ylabel("Approximated pi-value" , fontweight='bold')
plt.legend()
plt.savefig('A.1.png', dpi=600)
plt.show()

# Plot 2
plt.figure()
plt.title('Numeric Value of Pi - Monte Carlo', fontsize=12 , fontweight='bold')
plt.plot(result[:,2], result[:,1], lw=0.2)
plt.plot(range(1,result.shape[0]+1), [0] * result.shape[0], lw=1,
                    ls='dotted', c='black')
plt.xlabel("Number of particles n", fontweight='bold')
plt.ylabel("Difference of approximated vs true value" , fontweight='bold')
plt.savefig('A.2.png', dpi=600)
plt.show()

# Plot 3
fig, ax = plt.subplots()

x = np.linspace(-10,10, 10000)
ax.set(xlim=(-10, 10), ylim = (-10, 10))
n=10000
a_circle = plt.Circle((0, 0), 10.0, alpha=0.3, color='orange')
ax.add_artist(a_circle)
ax.set_title('n=10000', fontsize=12 , fontweight='bold')
ax.scatter(np.random.choice(x, n, replace=True), np.random.choice(x, n, replace=True),
           s=0.1)
plt.gca().set_aspect('equal', adjustable='box')
ax.set_xlabel('x', fontweight='bold'); ax.set_ylabel('y', fontweight='bold')
plt.savefig('A.3.png', dpi=600)
plt.show()

# %%