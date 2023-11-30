#%%
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt
import seaborn as sns
import sys
import datetime
from multiprocessing import Pool
import os

sns.set_style('darkgrid')
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')
sns.set_palette("viridis")

#%%
def MC(n):
    """ Calculates the numerical value of pi via Monte Carlo method """
    radius = 10
    
    particles_in_the_circle = 0

    for i in range(n): # Iterates over all n particles    
        coordinates = np.random.uniform(low=-radius, high=radius, size=2)
        if np.sqrt(coordinates[0]**2 + coordinates[1]**2) <= radius:
            """ Checks if particle is in the circle """
            particles_in_the_circle += 1
    
    pi_approx = particles_in_the_circle / n * 4 # Approximate values of pi
    dif = pi_approx - np.pi # Difference to true value

    return pi_approx, dif

#%%

if __name__ == "__main__":
    
    M = int(sys.argv[1]) # max. number of particles
    procs = int(sys.argv[2]) # number of cores
    procs_max = os.cpu_count() # Maximum number of cores        
    
    # Error handling
    if procs > procs_max:
        print("Maximum number of cores exceeded!! \n"
                        "Changed number of cores to Max. number " 
                        "of cores:", procs_max, '\n')
        procs = procs_max # Uses the maximum number of cores
    
    runtimes = [] # runtimes for every iteration
    for proc in range(1, procs+1): # Result using 1,2,3,4 cores
        start = datetime.datetime.now() # Starts the timer for the calculation
        p = Pool(proc) # Initializes procs-number of processes
        
        # Calculates the function in parallel for all M-steps of the calculation
        pi = p.map(MC, range(1, M+1)) 
        # Finishes the process
        p.close()
        p.join()        

        runtime = (datetime.datetime.now() - start).total_seconds(); # seconds
        runtimes.append(runtime)
        
        print('Execution time using', proc, 'cores:',
            runtime, 's')
    
    # Prints the net spee-up
    print("Speedup using highest number of cores vs. serial version:",
          runtimes[0] / runtimes[-1])
    
    plt.figure()
    plt.title('Numeric Value of Pi - Monte Carlo - Parallel', fontsize=12 ,
              fontweight='bold')
    # approx. pi-values
    plt.plot(range(1,M+1), [el[0] for el in pi], lw=0.3, c='r')
    # true value
    plt.plot(range(1,M+1), [np.pi] * M, lw=1,
                        ls='dotted', c='black', label='True value')
    
    plt.xlabel("Number of particles n", fontweight='bold')
    plt.ylabel("Approximated pi-value" , fontweight='bold')
    plt.legend()
    plt.savefig('B.1.png', dpi=600)
    plt.show()
                    
    #%%
    plt.figure()
    plt.title('Numeric Value of Pi - Monte Carlo - Parallel', fontsize=12 , fontweight='bold')
    plt.plot(range(1,M+1), [el[1] for el in pi], lw=0.1) 
    plt.plot(range(1,M+1), [0] * M, lw=1,
                        ls='dotted', c='black')
    
    plt.xlabel("Number of particles n", fontweight='bold')
    plt.ylabel("Difference of approximated vs true value" , fontweight='bold')
    plt.savefig('B.2.png', dpi=600)
    plt.show()
       
#%%