#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:11:55 2020

@author: Christian Wiskott

Advection equation. Numerical solution via different methods.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time

def fromm():

    start = time.time()    
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            if u > 0:
                f_minus = u * ( q[i-1,n] + (q[i,n] - q[i-2,n]) / (2*dx) / 2 * (dx - u*dt) )
                f_plus = u * ( q[i,n] + (q[i+1,n] - q[i-1,n]) / (2*dx) / 2 * (dx - u*dt) )
                q[i,n+1] = q[i,n] + dt / dx * (f_minus - f_plus)
            
            elif u < 0:
                f_minus = u * ( q[i,n] + (q[i+1,n] - q[i-1,n]) / (2*dx) / 2 * (dx - u*dt) ) 
                f_plus = u * ( q[i+1,n] + (q[i+2,n] - q[i,n]) / (2*dx) / 2 * (dx - u*dt) )
                q[i,n+1] = q[i,n] + dt / dx * (f_minus - f_plus)
                
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]

    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 
                
def donor_cell():
    
    start = time.time()
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            if u > 0:
                q[i,n+1] = q[i,n] + dt / dx * u*( q[i-1,n] - q[i,n] )
            elif u < 0:
                q[i,n+1] = q[i,n] + dt / dx * u*( q[i,n] - q[i+1,n] )
        
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]

    end = time.time()
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

def beam():
    
    start = time.time() 
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            if u > 0:
                f_minus = u * ( q[i-1,n] + (q[i-1,n] - q[i-2,n]) / dx / 2 * (dx - u*dt) )
                f_plus = u * ( q[i,n] + (q[i,n] - q[i-1,n]) / dx / 2 * (dx - u*dt) )
                q[i,n+1] = q[i,n] + dt / dx * (f_minus - f_plus)

            elif u < 0:
                f_minus = u * ( q[i,n] + (q[i,n] - q[i-1,n]) / dx / 2 * (dx - u*dt) )
                f_plus = u * ( q[i+1,n] + (q[i+1,n] - q[i,n]) / dx / 2 * (dx - u*dt) )
                q[i,n+1] = q[i,n] + dt / dx * (f_minus - f_plus)
            
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]
                
    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

def lax_wendrof_flux():
    
    start = time.time() 
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            if u > 0:
                f_minus = u * ( q[i-1,n] + (q[i,n] - q[i-1,n]) / dx / 2 * (dx - u*dt) )
                f_plus = u * ( q[i,n] + (q[i+1,n] - q[i,n]) / dx / 2 * (dx - u*dt) )
                q[i,n+1] = q[i,n] + dt / dx * (f_minus - f_plus)

            elif u < 0:
                f_minus = u * ( q[i,n] + (q[i+1,n] - q[i,n]) / dx / 2 * (dx - u*dt) )
                f_plus = u * ( q[i+1,n] + (q[i+2,n] - q[i+1,n]) / dx / 2 * (dx - u*dt) )
                q[i,n+1] = q[i,n] + dt / dx * (f_minus - f_plus)
                    
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]
            
    end = time.time()
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

def plot():
    
    sns.set_style('darkgrid')
    sns.set_context('paper')
    
    steps = int(N / 6) 
    sns.set_palette("rocket", 6+1)
    
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
clf = 0.6
dt = clf * dx / u
const = u*dt / (2*dx)
methods = ["donor", "fromm", "beam", "lax-wendrof"]

""" Simulation parameters """
M = len(x_grid)+4 # Number of grid points on string
N = 70 # Number of time steps. If altered plot parameters must be altered as well!!!!
q = np.zeros((M,N)) # Stores all dislocations at all time steps. 

# Initial conditions 
q[2:M-2,0] = [q0(x) for x in x_grid] 
q[0:2,0] = q[3,0]
q[M-2:,0] = q[M-3,0]

for method in methods:
    if method == "donor":
        donor_cell()
        plt.figure()
        plt.title("Donor-Cell Method", fontweight="bold", fontsize=15)
        plot()
        plt.savefig('donor.png', dpi=1200)
        plt.show()
 
    elif method == "fromm":
        fromm()
        plt.figure()
        plt.title("Fromm's Method", fontweight="bold", fontsize=15)
        plot()
        plt.savefig('fromm.png', dpi=1200)
        plt.show()

    elif method == "beam":
        beam()
        plt.figure()
        plt.title("Beam-Warming Method", fontweight="bold", fontsize=15)
        plot()
        plt.savefig('beam.png', dpi=1200)
        plt.show()
        
    elif method == "lax-wendrof":
        lax_wendrof_flux()
        plt.figure()
        plt.title("Lax-Wendrof Flux Conservation", fontweight="bold",
                  fontsize=15)
        plot()
        plt.savefig('lax_flux.png', dpi=1200)
        plt.show()
        
        
