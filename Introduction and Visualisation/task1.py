# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:39:32 2020

@author: Christian Wiskott
"""

#%%
import numpy as np

f = lambda x : 2*np.sin(x) + 1 # Function to approximate

def simpson(f, lower_bound, upper_bound, n): 
    """ Calculates the approximate definite integral for a 
    given function f between the lower and the upper bound via Simpsons 1/3 rule"""

    if n %2 != 0:
        raise Exception('n has to be even!\n')
    
    dx = (upper_bound - lower_bound) / n # spacing between points
    x = np.linspace(lower_bound, upper_bound, n+1) # Array contains n+1 elements with x0=lower_bound and x_n = upper_bound
    F = f(x) # all y-values
    
    res = 0 # Initialization of result
    for i in range(n+1): # loops over whole array to compute estimate
        if i==0 or i==n: # first and very last element
            res += F[i]
        elif i % 2 == 0: # even elements
            res += 2 * F[i]
        else: # uneven elements
            res += 4 * F[i]

    return res * (dx / 3)

eval = np.arange(2,3002, 2) # the investigated number of bins

"""  absolute value of residual of true value and the computed estimate via Simpson's rule """
res_eval = [np.abs((4+np.pi) - simpson(f, 0, np.pi, x)) for x in eval] 

# ------------- Plot ------------------#
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette("rocket")
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')

plt.title("Number of Bins vs. Residual of Simpson\'s rule and true value", fontweight='bold')
plt.xlabel('Number of bins', fontweight='bold')
plt.ylabel('log(|F(x)-F\'(x)|)', fontweight='bold')
plt.plot(eval, res_eval, c="g", label="Residual", lw=0.5)
plt.yscale("log")
plt.xticks(range(0,3200,200))
plt.legend()
plt.tight_layout()

# %%
