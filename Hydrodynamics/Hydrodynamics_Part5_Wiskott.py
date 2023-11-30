#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:11:55 2020

@author: Chris

Diffusion equation. Numerical solution via different methods.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time

def FTCS():

    start = time.time()
    for n in range(N-1): # time steps
        for i in range(2,M-2): # grid points (excluding the ghost cells)
            q[i,n+1] = q[i,n] + D*dt/(dx**2) * ( q[i+1,n] - 2*q[i,n] + q[i-1,n] )
        
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]
                
    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

    return q

def BTCS():
    
    a = c = - D*dt / dx**2
    b = 1 - 2*a
    
    # triangular solver
    # adapted from https://stackoverflow.com/a/43214907
    def solver(a,b,c,d):
        n = len(d)
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
        x_n = np.zeros(n)
        
        c_prime[0] = c/b
        d_prime[0] = d[0]/b
        
        for i in range(1,n-1):
            c_prime[i] = c/(b - a*c_prime[i-1])
        for i in range(1,n):
            d_prime[i] = (d[i] - a*d_prime[i-1])/(b - a*c_prime[i-1])
        x_n[n-1] = d_prime[n-1]
        for i in range(n-1,0,-1):
            x_n[i-1] = d_prime[i-1] - c_prime[i-1]*x_n[i]
        
        return x_n
    
    start = time.time()
    for n in range(N-1): # time steps
        
        # Computes q_i^n+1
        x_n = solver(a,b,c,q[2:M-2,n])
        q[2:M-2,n+1] = x_n
        
        # Resetting the ghost cells 
        q[0:2,n+1] = q[3,n+1]
        q[M-2:,n+1] = q[M-3,n+1]

    end = time.time() 
    print("Finished!"); print('Elapsed time of computation:\t' + str(end-start) + 's') 

    return q

def plot(m):
    
    sns.set_style('darkgrid')
    sns.set_context('paper')
    
    steps = int(1 / dt / 6) + 1 
    sns.set_palette("rocket", 7)

    # iterates over timesteps and plots the respective results
    for n in range(0, int(1/dt)+steps, steps):
        plt.plot(x_grid, m[2:M-2,n], label="t="+str(round(n*dt,3))+"s",
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
    return 1 if x >= 3.5 and x <= 6.5 else 0.1

""" Constants """
D = 1 
x_grid = np.linspace(0,10,100)
dx = x_grid[1] - x_grid[0]
clfs = [0.5, 1.0, 1.5]
methods = ["FTCS", "BTCS"]

""" Simulation parameters """
M = len(x_grid)+4 # Number of grid points
N = 1000 # Number of time steps
q = np.zeros((M,N)) 

# Initial conditions 
q[2:M-2,0] = [q0(x) for x in x_grid] 
q[0:2,0] = q[3,0]
q[M-2:,0] = q[M-3,0]

for method in methods:
    if method == "FTCS":
        for clf in clfs:
            dt = clf * dx**2 / (2*D)
            r = FTCS()
            plt.figure()
            plt.title("FTCS, c={}".format(clf), fontweight="bold", fontsize=15)
            plot(r)
            plt.savefig('FTCS{}.png'.format(clf), dpi=600)
            plt.show()

    elif method == "BTCS":
        for clf in clfs:
            dt = clf * dx**2 / (2*D)
            t = BTCS()
            plt.figure()
            plt.title("BTCS, c={}".format(clf), fontweight="bold", fontsize=15)
            plot(t)
            plt.savefig('BTCS{}.png'.format(clf), dpi=600)
            plt.show()
# %%
