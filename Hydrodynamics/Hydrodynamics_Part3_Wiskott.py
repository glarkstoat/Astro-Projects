#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:11:55 2020

@author: Christian Wiskott

Sod shock tube test via different methods.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time

def lax():

    for n in range(0,2*N,2): # time steps
        
        # Calculating dt
        for i in range(2,M-2,2): # grid points at cell centers (excluding the ghost cells)
            c_s = np.sqrt( gamma * P[i,n] / rho[i,n] )
            dt = dx / (np.abs(u[i,n]) + c_s) 
        dt = c * np.min(dt)

        # Evolving the system 
        for i in range(2,M-2,2): # grid points at cell centers (excluding the ghost cells)

            # Step 1
            # Cell-boundary gradients of P and Pu
            dP_dx[i+1,n] = (P[i+2,n] - P[i,n]) / dx     
            dP_dx[i-1,n] = (P[i,n] - P[i-2,n]) / dx
            
            dPu_dx[i+1,n] = (P[i+2,n] * u[i+2,n] - P[i,n] * u[i,n]) / dx
            dPu_dx[i-1,n] = (P[i,n] * u[i,n] - P[i-2,n] * u[i-2,n]) / dx
            
            # Step 2
            # Update half-stetp rho, rho_u, e
            rho[i+1,n+1] = 0.5 * (rho[i+2,n] + rho[i,n]) - 0.5*dt/dx * (rho[i+2,n]*u[i+2,n] - rho[i,n]*u[i,n])
            rho[i-1,n+1] = 0.5 * (rho[i,n] + rho[i-2,n]) - 0.5*dt/dx * (rho[i,n]*u[i,n] - rho[i-2,n]*u[i-2,n])
            
            rho_u[i+1,n+1] = 0.5 * (rho_u[i+2,n] + rho_u[i,n]) - 0.5*dt/dx * (rho_u[i+2,n]*u[i+2,n] - rho_u[i,n]*u[i,n]) - 0.5*dt*dP_dx[i+1,n]
            rho_u[i-1,n+1] = 0.5 * (rho_u[i,n] + rho_u[i-2,n]) - 0.5*dt/dx * (rho_u[i,n]*u[i,n] - rho_u[i-2,n]*u[i-2,n]) - 0.5*dt*dP_dx[i-1,n]

            e[i+1,n+1] = 0.5 * (e[i+2,n] + e[i,n]) - 0.5*dt/dx * (e[i+2,n]*u[i+2,n] - e[i,n]*u[i,n]) - 0.5*dt * dPu_dx[i+1,n]
            e[i-1,n+1] = 0.5 * (e[i,n] + e[i-2,n]) - 0.5*dt/dx * (e[i,n]*u[i,n] - e[i-2,n]*u[i-2,n]) - 0.5*dt * dPu_dx[i-1,n]
            
            # Step 3
            # Updated half-step pressures and speeds from updated rho, rho_u and e
            P[i+1,n+1] = (gamma - 1) * (e[i+1,n+1] - 0.5 * rho_u[i+1,n+1]**2 / rho[i+1,n+1])
            P[i-1,n+1] = (gamma - 1) * (e[i-1,n+1] - 0.5 * rho_u[i-1,n+1]**2 / rho[i-1,n+1])
            
            u[i+1,n+1] = rho_u[i+1,n+1] / rho[i+1,n+1]
            u[i-1,n+1] = rho_u[i-1,n+1] / rho[i-1,n+1]
            
            # Cell-center gradients of P and Pu
            dP_dx[i,n+1] = (P[i+1,n+1] - P[i-1,n+1]) / dx     
            dPu_dx[i,n+1] = (P[i+1,n+1] * u[i+1,n+1] - P[i-1,n+1] * u[i-1,n+1])
            
            # Step 4
            # Update full-stetp rho, rho_u, e
            rho[i,n+2] = rho[i,n] - dt/dx * (rho[i+1,n+1]*u[i+1,n+1] - rho[i-1,n+1]*u[i-1,n+1])
            rho_u[i,n+2] = rho_u[i,n] - dt/dx * (rho_u[i+1,n+1]*u[i+1,n+1] - rho_u[i-1,n+1]*u[i-1,n+1]) - dt * dP_dx[i,n+1]
            e[i,n+2] = e[i,n] - dt/dx * (e[i+1,n+1]*u[i+1,n+1] - e[i-1,n+1]*u[i-1,n+1]) - dt * dPu_dx[i,n+1]
            
            # Updated full-step pressures and speeds from updated rho, rho_u and e
            P[i,n+2] = (gamma - 1) * (e[i,n+2] - 0.5 * rho_u[i,n+2]**2 / rho[i,n+2])
            u[i,n+2] = rho_u[i,n+2] / rho[i,n+2]
            
        # Resetting the ghost cells 
        P[0:2,n+2] = P[2,n+2]
        P[M-2:,n+2] = P[M-3,n+2]
        
        rho[0:2,n+2] = rho[2,n+2]
        rho[M-2:,n+2] = rho[M-3,n+2]
        
        rho_u[0:2,n+2] = rho_u[2,n+2]
        rho_u[M-2:,n+2] = rho_u[M-3,n+2]
        
        e[0:2,n+2] = e[2,n+2]
        e[M-2:,n+2] = e[M-3,n+2]
        
        u[0:2,n+2] = u[2,n+2]
        u[M-2:,n+2] = u[M-3,n+2]

def upwind():

    for n in range(0,2*N,2): # time steps
        
        # Calculating dt
        for i in range(2,M-2,2): # grid points at cell centers (excluding the ghost cells)
            c_s = np.sqrt( gamma * P[i,n] / rho[i,n] )
            dt = dx / (np.abs(u[i,n]) + c_s) 
        dt = c * np.min(dt)

        for i in range(2,M-2,2): # grid points at cell centers (excluding the ghost cells)
                
            # Step 1
            # advection speeds at boundaries
            u[i+1,n] = 0.5 * (u[i,n] + u[i+2,n])
            u[i-1,n] = 0.5 * (u[i-2,n] + u[i,n])

            # Step 2 & 3
            # Fluxes at boundaries & updates full-steps
            if u[i+1,n] > 0:
                rho[i,n+2] = rho[i,n] + dt/dx * (u[i-1,n]*rho[i-2,n] - u[i+1,n]*rho[i,n])
                rho_u[i,n+2] = rho_u[i,n] + dt/dx * (u[i-1,n]*rho_u[i-2,n] - u[i+1,n]*rho_u[i,n])
                e[i,n+2] = e[i,n] + dt/dx * (u[i-1,n]*e[i-2,n] - u[i+1,n]*e[i,n])
            
            elif u[i+1,n] < 0:
                rho[i,n+2] = rho[i,n] + dt/dx * (u[i-1,n]*rho[i,n] - u[i+1,n]*rho[i+2,n])
                rho_u[i,n+2] = rho_u[i,n] + dt/dx * (u[i-1,n]*rho_u[i,n] - u[i+1,n]*rho_u[i+2,n])
                e[i,n+2] = e[i,n] + dt/dx * (u[i-1,n]*e[i,n] - u[i+1,n]*e[i+2,n])
            
            else: 
                rho[i,n+2] = rho[i,n]
                rho_u[i,n+2] = rho_u[i,n]
                e[i,n+2] = e[i,n]
          
            # Step 4
            # Cell-center gradients of P and Pu
            dP_dx[i,n] = (P[i+2,n] - P[i-2,n]) / (2*dx)
            dPu_dx[i,n] = (P[i+2,n] * u[i+2,n] - P[i-2,n] * u[i-2,n]) / (2*dx)
            
            # Update full-step rho_u, e
            rho_u[i,n+2] = rho_u[i,n+2] - dt * dP_dx[i,n]
            e[i,n+2] = e[i,n+2] - dt * dPu_dx[i,n]

            # Updated full-step pressures and speeds from updated rho, rho_u and e
            P[i,n+2] = (gamma - 1) * (e[i,n+2] - 0.5 * rho_u[i,n+2]**2 / rho[i,n+2])
            u[i,n+2] = rho_u[i,n+2] / rho[i,n+2]

        # Resetting the ghost cells 
        P[0:2,n+2] = P[2,n+2]
        P[M-2:,n+2] = P[M-3,n+2]
        
        rho[0:2,n+2] = rho[2,n+2]
        rho[M-2:,n+2] = rho[M-3,n+2]
        
        rho_u[0:2,n+2] = rho_u[2,n+2]
        rho_u[M-2:,n+2] = rho_u[M-3,n+2]
        
        e[0:2,n+2] = e[2,n+2]
        e[M-2:,n+2] = e[M-3,n+2]
        
        u[0:2,n+2] = u[2,n+2]
        u[M-2:,n+2] = u[M-3,n+2]
 
def plot_lax():
    
    sns.set_style('darkgrid')
    sns.set_context('paper')
    lss = ["dashed", "solid"]
    cs = ["black", "b"]
    metrics = [rho, u, P]
    labels = ["rho", "u", "P"]
    
    # iterates over timesteps and plots the respective results
    for metric, label in zip(metrics, labels):
        plt.figure()
        i=0
        for n in range(0, 2*N+1, 400):
            plt.plot(x_grid[::2], metric[2:M-2:2,n], ls=lss[i], lw=2,
                        c=cs[i])
            i+=1
        plt.title("Lax-Wendroff, c={}".format(c))
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        plt.xlabel("x", fontweight="bold", fontsize=14)
        plt.ylabel("{}".format(label), fontweight="bold", fontsize=14)
        plt.tight_layout()
        plt.savefig("{}_lax.png".format(label), dpi=1200)
        plt.show()
   
def plot_upwind():
    
    sns.set_style('darkgrid')
    sns.set_context('paper')

    metrics = [rho, u, P]
    labels = ["rho", "u", "P"]
    
    # iterates over timesteps and plots the respective results
    
    for metric, label in zip(metrics, labels):
        plt.figure()
        plt.plot(x_grid[::2], metric[2:M-2:2,0], ls="dashed", lw=2, c="black")                 
        plt.plot(x_grid[::2], metric[2:M-2:2,1000], ls="solid", lw=2, c="b")                 
        plt.title("Upwind, c={}".format(c))
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        plt.xlabel("x", fontweight="bold", fontsize=14)
        plt.ylabel("{}".format(label), fontweight="bold", fontsize=14)
        plt.tight_layout()
        plt.savefig("{}_upwind.png".format(label), dpi=1200)
        plt.show()

# ---------------------------------------------------------
""" Initial conditions """
def rho_init(x): 
    return 1 if x >= 0.35 and x <= 0.65 else 0.125

def P_init(x): 
    return 1 if x >= 0.35 and x <= 0.65 else 0.1

""" Constants """
gamma = 5/3
x_grid = np.linspace(0,1,2000)
dx = x_grid[2] - x_grid[0]
c = 0.97

""" Simulation parameters """
M = len(x_grid)+4 # Number of grid points
N = 300 # Number of time steps
dP_dx, dPu_dx = np.zeros((M,2*N+1)), np.zeros((M,2*N+1))  
P, u = np.zeros((M,2*N+1)), np.zeros((M,2*N+1)) 
rho, rho_u, e = np.zeros((M,2*N+1)), np.zeros((M,2*N+1)), np.zeros((M,2*N+1)) 

# Initial conditions 
rho[2:M-2,0] = [rho_init(x) for x in x_grid] 
P[2:M-2,0] = [P_init(x) for x in x_grid] 
e[2:M-2,0] = [P_init(x) / (gamma -1) for x in x_grid]

# Ghost cells kept constant
rho[0:2,0] = rho[2,0]
rho[M-2:,0] = rho[-3,0]
u[0:2,0] = u[2,0]
u[M-2:,0] = u[-3,0]
rho_u[0:2,0] = rho_u[2,0]
rho_u[M-2:,0] = rho_u[-3,0]
P[0:2,0] = P[2,0]
P[M-2:,0] = P[-3,0]
e[0:2,0] =e[2,0]
e[M-2:,0] =e[-3,0]

lax()
plot_lax()

""" Simulation parameters for upwind scheme """

c = 0.2
N = 1000 # Number of time steps

# reset of initial conditions
dP_dx, dPu_dx = np.zeros((M,2*N+1)), np.zeros((M,2*N+1))  
P, u = np.zeros((M,2*N+1)), np.zeros((M,2*N+1)) 
rho, rho_u, e = np.zeros((M,2*N+1)), np.zeros((M,2*N+1)), np.zeros((M,2*N+1)) 

# Initial conditions 
rho[2:M-2,0] = [rho_init(x) for x in x_grid] 
P[2:M-2,0] = [P_init(x) for x in x_grid] 
e[2:M-2,0] = [P_init(x) / (gamma -1) for x in x_grid]

# Ghost cells kept constant
rho[0:2,0] = rho[2,0]
rho[M-2:,0] = rho[-3,0]
u[0:2,0] = u[2,0]
u[M-2:,0] = u[-3,0]
rho_u[0:2,0] = rho_u[2,0]
rho_u[M-2:,0] = rho_u[-3,0]
P[0:2,0] = P[2,0]
P[M-2:,0] = P[-3,0]
e[0:2,0] =e[2,0]
e[M-2:,0] =e[-3,0]

upwind()
plot_upwind()

# %%
