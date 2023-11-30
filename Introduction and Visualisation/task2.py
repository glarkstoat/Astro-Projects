# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:39:32 2020

@author: Christian Wiskott
"""
#%%
import numpy as np

f = lambda x : x**3 - x + 1 # Function 1
f_prime = lambda x : 3*x**2 - 1  # f'

f2 = lambda x : np.cos(x) - 2*x # Function 2
f2_prime = lambda x : -np.sin(x) - 2 # f2'

def newton(f, f_prime, init, n=10**6): 
    """ Calculates the approximate solution of the root of function f starting from the 
    initial guess init. Utilizes Newton's method. """
    
    i=0; res=[]; inits=[] # Array of all values f(x_i) after each iteration
    while i <= n and np.abs(f(init)) > 1e-10: # As soon as either max-iteration or convergence-criterion is reached, loop is broken
       init = init - f(init) / f_prime(init) # Newton's method
       res.append(f(init))
       inits.append(init)
       i += 1
       
    return init, res, inits

print(newton(f, f_prime, 0)[0]) # root of f
print(newton(f2, f2_prime, 0)[0]) # root of f'

# ------------- Plot ------------------#
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('paper')

plt.subplot(221)
x = np.arange(-2,2,1e-5)
plt.title("Approx. root-finding \w Newton iteration", fontweight='bold')
#plt.xlabel('x', fontweight='bold')
plt.ylabel('y', fontweight='bold')
plt.scatter(newton(f, f_prime, 0)[0], 0, c="r", label="Approx. solution", s=15)
plt.plot(x, f(x), c="g",lw=1,label="f(x)=x^3-x+1")
plt.plot(x,np.zeros(len(x)),c="black",lw=1,label="y=0")
#plt.yscale("log")
plt.legend(prop={'size': 6})
plt.tight_layout()

plt.subplot(222)
res = newton(f,f_prime,0)[1]; x = range(1,len(res)+1)

plt.title("Convergence to 0", fontweight='bold')
#plt.xlabel('Number of iterations', fontweight='bold')
plt.ylabel('f(x_i)', fontweight='bold')
plt.plot(x, res, c="g",lw=1,label="f(x_i)")
#plt.yscale("log")
plt.legend(prop={'size': 6})
plt.tight_layout()

plt.subplot(223)
x = np.arange(-2,2,1e-5)
#plt.title("Approx. root-finding \w Newton iteration", fontweight='bold')
plt.xlabel('x', fontweight='bold')
plt.ylabel('y', fontweight='bold')
plt.scatter(newton(f2, f2_prime, 0)[0], 0, c="r", label="Approx. solution", s=15)
plt.plot(x, f2(x),lw=1,label="f(x)=cos(x)-2x")
plt.plot(x,np.zeros(len(x)),c="black",lw=1,label="y=0")
#plt.yscale("log")
plt.legend(prop={'size': 6})
plt.tight_layout()

plt.subplot(224)
res2 = newton(f2,f2_prime,0)[1]; x = range(1,len(res2)+1)

#plt.title("Convergence to 0", fontweight='bold')
plt.xlabel('Number of iterations', fontweight='bold')
plt.ylabel('f(x_i)', fontweight='bold')
plt.plot(x, res2,lw=1,label="f(x_i)")
plt.xticks(range(1,len(res2)+1))
#plt.yscale("log")
plt.legend(prop={'size': 6})
plt.tight_layout()

plt.show()

plt.figure()

ar = newton(f, f_prime, 0)[2]
plt.subplot(121)
#plt.title("", fontweight='bold')
plt.xlabel('Number of iterations', fontweight='bold')
plt.ylabel('root', fontweight='bold')
plt.plot(range(1,len(ar)+1), ar, c="g",label="f(x)=x^3-x+1")
plt.xticks(range(1,len(ar)+1,2))
plt.legend(prop={'size': 6})
plt.tight_layout()

ar = newton(f2, f2_prime, 0)[2]
plt.subplot(122)
plt.xlabel('Number of iterations', fontweight='bold')
plt.ylabel('root', fontweight='bold')
plt.plot(range(1,len(ar)+1), ar,label="f(x)=cos(x)-2x")
plt.xticks(range(1,len(ar)+1))
plt.legend(prop={'size': 6})
plt.tight_layout()

plt.show()

#%%