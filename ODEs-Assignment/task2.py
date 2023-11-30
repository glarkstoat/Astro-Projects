""" Solving initial values problem dy/dt = 0.1*y + np.sin(t), y(t=0)=1 
    via Runge-Kutta-Fehlberg method """

#%%
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib qt for plots in separate window
import seaborn as sns
sns.set_style('darkgrid')
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')
sns.set_palette("rocket_r")

""" ------------------ Functions ---------------------- """
# analytical solution
y = lambda t: 1.990099 * np.exp(0.1*t) - 0.0990099 * np.sin(t) - 0.990099 * np.cos(t)
# dy/dt
f = lambda t,y: 0.1*y + np.sin(t) 

def runge_kutta_45(y0, t0, tn, h, lte_tol):
    """ Calculates the numeric solution of the IVP via 
    Runge Kutta 45 scheme (5th-order). y0=intial value,
     t0 & tn range over which to calculate the solution.
     h = stepsize, lte_tol is the LTE tolerance for RK 45."""
    
    def y_update(y0, t0, h, lte_tol):
        """ Calculates the next y-value via the 6-steps of RK-45.
        Includes the 4th- and 5-th order approximation and the optimal 
        stepsize via the LTE-tolerance.
        """
        # 6 rk-steps
        k1 = f(t0, y0)
        k2 = f(t0 + 1/4 * h, y0 + 1/4 * h * k1)
        k3 = f(t0 + 3/8 * h, y0 + 3/32 * h * k1 + 9/32 * h * k2)
        k4 = f(t0 + 12/13 * h, y0 + 1932/2197 * h * k1 - 7200/2197 * h * k2 + 7296/2197 * h * k3)
        k5 = f(t0 + h, y0 + 439/216 * h * k1 - 8 * h * k2 + 3680/513 * h * k3 - 845/4104 * h * k4)
        k6 = f(t0 + 1/2 * h, y0 - 8/27 * h * k1 - 2 * h * k2 + 3544/2565 * h * k3 + 1859/4104 * h * k4 - 11/40 * h * k5)
        
        # 4-th order approximation
        yn1_approx = y0 + h * (25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
        # 5-th order update
        yn1 = y0 + h * (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
        
        # Approximation of the local truncation error (lte)
        lte = yn1 - yn1_approx
        # lte tolarance
        lte_tol = lte_tol * y0

        # optimal stepsize with safety parameter 0.999
        h_opt = 0.999 * h * (np.abs(lte_tol / lte))**(1/5)
        
        return yn1, h_opt
    
    y = [y0] # initial value. contains all computed y-values
    timesteps = [t0] # contains all timesteps during computation
    stepsizes = [h] # all computed optimal stepsizes
    
    while t0 < tn: # iterates over pre-defined t-range
        
        # computes the y-update via RK 45
        y_new, h_opt = y_update(y0, t0, h, lte_tol)
        
        while h > h_opt:
            """ If computed stepsize is not acceptable repeat the 
            computation of the y-update with optimal stepsize.
            Repeat until stepsize is acceptable. """
            
            h = h_opt
            y_new, h_opt = y_update(y0, t0, h, lte_tol)
    
        y0, h = y_new, h_opt # updates the y-value and the stepsize
        t0 = t0 + h # updates the t-value
        timesteps.append(t0) # records the taken timestep
        y.append(y0) # appends the computed y-value
        stepsizes.append(h_opt) # records the optimal stepsize
    
    return timesteps, y, stepsizes

#%%
""" Computing performance comparison of different methods """
stepsize = 0.5
lte_tolarances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
t0 = 0; tn = 20 # starting- and end-point
y0 = 1 # initial value y(t=0)

# Plot of the solutions for different stepsizes (y vs t)
xranges1 = np.arange(t0,tn+0.01, 0.01) # for analytical solution

plt.figure()
plt.title('RK 45 - Different '+ r'$LTE_{tol}$'+' - Numerical Solution', fontsize=12)
plt.plot(xranges1, y(xranges1), label='Analytic', 
            linestyle='dotted', c='black')

for lte_tol in lte_tolarances:
    # plot solutions for different tolerances
    res =  runge_kutta_45(y0, t0, tn, stepsize, lte_tol)
    plt.plot(res[0], res[1], label=r'$LTE_{tol}= $'+str(lte_tol))

plt.xlabel("t"); plt.ylabel("y(t)")
plt.legend()
plt.tight_layout()
plt.savefig('2.1.png', dpi=600)
plt.show()

# Plot of all timesteps verses dts
plt.figure()
plt.title('RK 45 - Different '+ r'$LTE_{tol}$'+' - dt vs t', fontsize=12)
for lte_tol in lte_tolarances:
    # plot solutions for different tolerances
    res =  runge_kutta_45(y0, t0, tn, stepsize, lte_tol)
    plt.plot(res[0], res[2], label=r'$LTE_{tol}= $'+str(lte_tol))

plt.xlabel("t"); plt.ylabel("dt")
plt.legend()
plt.tight_layout()
plt.savefig('2.2.png', dpi=600)
plt.show()
    
# Plots of all number of timesteps verses lte_tols
plt.figure()
plt.title('RK 45 - ' +r'$LTE_{tol}$'+' vs. Number of timesteps', fontsize=12)

number_of_timesteps = []
for lte_tol in lte_tolarances:

    res =  runge_kutta_45(y0, t0, tn, stepsize, lte_tol)
    number_of_timesteps.append(len(res[0])) # number of timesteps

plt.plot(lte_tolarances, number_of_timesteps)
plt.xscale('log')
plt.ylabel("numer of timesteps"); plt.xlabel(r'$LTE_{tol}$')
plt.tight_layout()
plt.savefig('2.3.png', dpi=600)
plt.show()
# %%