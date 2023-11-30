# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:13:40 2020

@author: Christian Wiskott
"""
# %% 
import numpy as np

# ------- Constants in SI-Units-----------
kB = 1.380649e-23
m_sun = 1.9885e30
r_sun = 6.96342e8 
au = 1.49597e11 # astronomical unit
m_mol = 0.5 * 1.672621e-27 # 1/2 mass of proton
G = 6.67430e-11

speed_sound = lambda T : np.sqrt((kB*T)/m_mol) # speed of sound for given temperature
r_critical = lambda c_s : (G*m_sun) / (2*c_s**2) # critical radius for given temperature

def bisection(r, T, a, b, k): 
    """ Calculates the numerical solution of the parker wind equation via bisection.
        Choose a and be such that f(a)*f(b) < 0. k = convergence criterion for bisecton method."""

    c_s = speed_sound(T); r_c = r_critical(c_s) # Calculates the critical radius and the speed of sound for current temperature 
    func = lambda v : v * np.exp(-v**2 / (2*c_s**2)) - c_s * (r_c / r)**2 * np.exp(-2*r_c/r + 3/2) # Function that will be evaluated via bisection
    # parker wind equation set to f(v)=0
    
    if func(a)*func(b) >= 0: # Checks bisection condition
        raise Exception('Choose a and be such that, f(a)*f(b) < 0.'
                        ' Currently: f(a)*f(b)=', func(a)*func(b))
   
    i=0 # Serves as a counter for the maximum allowed iterations
    while (abs((b-a)) >= k and i < 10**7): # Continues the loop until convergence criterion is met
     # or maximum number of iterations is reached 
        
        c = (a+b)/2 # Defines the middle point
        if (func(c) == 0): # If the middle point is already the solution to f(v)=0 then loop is interrupted
            print("Exact solution!\n")
            break
   
        if (func(c)*func(a) < 0): # Checks on which side the bisection will continue
            b = c # root lies between a and c
        else:
            a = c # root lies between b and c
        i += 1 # Advance the counter
    return c # approximation of v for which f(v) = 0 

bins = np.linspace(2*r_sun, au, 100) # evenly spaced radii bins
temps = np.array([2,4,6,8,10]) * 1e+6 # temperatures to investigate. In MK.
v_solutions = np.array([]) # contains all solutions of bisection for each temp and radius

for T in temps: # loops over all temperatures
    c_s = speed_sound(T); r_c = r_critical(c_s) # Speed of sound and critical radius for given temperature
    for r in bins: # loops over all radii
        if r < r_c: # speed is subsonic i.e. v < c_s. a and b are chosen accordingly 
            v_solutions = np.append(v_solutions, bisection(r, T, 1e+4, c_s, 1e-7))
        elif r == r_c: # speed is at critical point i.e. v == c_s
            v_solutions = np.append(v_solutions, c_s)
        else: # speed is supersonic i.e. v > c_s. a and b are chosen accordingly 
            v_solutions = np.append(v_solutions, bisection(r, T, c_s, 1e+7, 1e-7))

bins /= r_sun # in units of solar radii
v_solutions /= 1000 # in km/s

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')
sns.set_palette("rocket")

plt.title("Solution of the parker wind equation using bisection", fontweight="bold")
plt.xlabel(r'r [$R_{\odot}$]', fontweight="bold"); plt.ylabel("v [km/s]", fontweight="bold")
for i in range(0,500,100): # plots all temperatures
    plt.plot(bins, v_solutions[i:i+100], label="T="+str(temps[int(i/100)]/1e+6)+"MK")

plt.xticks(range(0,250,25))
plt.legend(prop={'size': 6})
plt.tight_layout()
#plt.savefig('3.png', dpi=600)
# %%
