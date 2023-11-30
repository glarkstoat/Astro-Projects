#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:11:55 2020

@author: Chris

Advection equation. Numerical solution via different methods.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time

def central_difference():

    start = time.time()
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            q[i,n+1] = q[i,n] - const * ( q[i+1,n] - q[i-1,n] )
        
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]
                
    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

def upwind():
    
    start = time.time() 
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            q[i,n+1] = q[i,n] - 2*const * ( q[i,n] - q[i-1,n] )
        
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]

    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

def lax_wendrof():
    
    start = time.time()
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            q[i,n+1] = q[i,n] - const * (q[i+1,n] - q[i-1,n]) + ((2*const) ** 2) / 2 * (q[i+1,n] - 2*q[i,n] + q[i-1,n]) 
        
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]
            
    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

def plot():
    
    sns.set_style('darkgrid')
    sns.set_context('paper')
    
    steps = int(N / 6) 
    sns.set_palette("rocket", 7)
    
    # iterates over timesteps and plots the respective results
    for n in range(0, N, steps):
        plt.plot(x_grid, q[2:M-2,n], label="t="+str(round(n*dt,2))+"s",
                 lw=3, alpha=0.6) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc=0, prop={'size': 10})
    plt.xlabel("x", fontweight="bold", fontsize=14)
    plt.ylabel("q(x,t)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    
# ---------------------------------------------------------
""" Initial condition of density """
def q0(x): 
    return 1 if x >= 2 and x <=4 else 0.1

""" Constants """
u = 0.5 # Velocity of the wave packet
x_grid = np.linspace(0,10,100)
dx = x_grid[1] - x_grid[0]
clf = 0.2
dt = clf * dx / u
const = u*dt / (2*dx)
methods = ["central", "upwind", "lax-wendrof"]

""" Simulation parameters """
M = len(x_grid)+4 # Number of grid points
N = 70 # Number of time steps
q = np.zeros((M,N)) # Stores results 

# Initial conditions 
q[2:M-2,0] = [q0(x) for x in x_grid] 
q[0:2,0] = q[3,0]
q[M-2:,0] = q[M-3,0]

for method in methods:
    if method == "central":
        central_difference()
        plt.figure()
        plt.title("Central Difference", fontweight="bold", fontsize=12)
        plot()
        plt.savefig('central.png', dpi=600)
        plt.show()

    elif method == "upwind":
        upwind()
        plt.figure()
        plt.title("Upwind Scheme", fontweight="bold", fontsize=12)
        plot()
        plt.savefig('upwind.png', dpi=600)
        plt.show()
        
    elif method == "lax-wendrof":
        lax_wendrof()
        plt.figure()
        plt.title("Lax-Wendrof", fontweight="bold", fontsize=12)
        plot()
        plt.savefig('lax.png', dpi=600)
        plt.show()
# %%
