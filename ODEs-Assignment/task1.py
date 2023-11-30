""" Solving initial values problem dy/dt = -2ty, y(t=0)=1 
    via different methods (explicit and implicit) """

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
y = lambda t: np.exp(-t**2) # analytical solution
f = lambda t,y: -2*t*y # dy/dt

def forward_euler(y0, t0, tn, h):
    """ Calculates the numeric solution of the IVP via 
    forward euler scheme """
    
    y = [y0] # initial values. contains the computed y-values
    while t0 < tn: # iterates over pre-defined t-range
        y0 = y0 + h * f(t0, y0) # forward euler - step
        t0 = t0 + h # updates the t-value
        y.append(y0) # appends the computed y-value
    return y

def runge_kutta_4(y0, t0, tn, h):
    """ Calculates the numeric solution of the IVP via 
    Runge Kutta 4-th order scheme """
    
    y = [y0] # initial value
    while t0 < tn: # iterates over pre-defined t-range
        # computing the slopes at different positions
        k1 = f(t0, y0) # initial step
        k2 = f(t0 + h/2, y0 + k1*h/2) # Midpoint
        k3 = f(t0 + h/2, y0 + k2*h/2) # Midpoint
        k4 = f(t0 + h, y0 + k3*h) # Full step
        
        # Updating y-values via weighted average of slops
        y0 = y0 + h/6 * (k1 + 2*k2 + 2*k3+ k4)
        t0 = t0 + h # updates the t-value
        y.append(y0) # appends the computed y-value
    return y

def backward_euler(y0, t0, tn, h):
    """ Calculates the solution of the IVP via backward euler using 
    Newton-iteration as root-finding method """
    
    def G(y, t, yn, h):
        """ G_n+1^m """
        return y * (1 + 2 * h * (t+h)) - yn
    
    def G_prime(t, h):
        """ G'_n+1^m """
        return 1 + 2 * h * (t+h)

    res = [y0] # initial value. Contains all y-values
    while t0 < tn: # from starting to end-value  
        for i in range(5): # how often Newton will be perfomed
            """ Newton iteration. m-steps to find y_n+1^m"""
            y0 = y0 - G(y0, t0, res[-1], h) / G_prime(t0, h)
        res.append(y0)
        t0 = t0 + h # updates the timestep
    return res

def crank_nicolson(y0, t0, tn, h):
    """ Calculates the solution of the IVP via Crank-Nicolson scheme
    using Newton-iteration as root-finding method """
    
    def G(yn1, t, yn, h):
        """ G_n+1^m """
        return yn1 - yn + h * (t*yn + (t+h)*yn1)
        
    def G_prime(t, h):
        """ G'_n+1^m """
        return 1 + h * (t+h)

    res = [y0] # initial value. Contains all y-values
    while t0 < tn: # from starting to end-value  
        for i in range(5): # how often Newton will be perfomed
            """ Newton iteration. m-steps to find y_n+1^m"""
            y0 = y0 - G(y0, t0, res[-1], h) / G_prime(t0, h)
        res.append(y0)
        t0 = t0 + h # updates the timestep
    return res

#%%

""" Computing performance comparison of different methods """
stepsizes = [1, 0.5, 0.2, 0.1] # stepsizes
t0 = 0; tn = 3 # starting- and end-point
y0 = 1 # initial value y(t=0)

# Plot of the solutions for different stepsizes
xranges1 = np.arange(t0,tn+0.01, 0.01)
for h in stepsizes:
    plt.figure()
    plt.title(str(int(tn/h)) + ' steps', fontweight='bold', 
          fontsize=12)
    t = np.arange(t0, tn+h, h)
    
    plt.plot(xranges1, y(xranges1), label='Analytic', 
            linestyle='dotted', c='black')
    
    plt.plot(t, forward_euler(y0, t0, tn, h), label='Forward Euler')
    plt.plot(t, backward_euler(y0, t0, tn, h), label='Backward Euler')   
    plt.plot(t, runge_kutta_4(y0, t0, tn, h), label='Runge Kutta')   
    plt.plot(t, crank_nicolson(y0, t0, tn, h), label='Crank Nicolson')
             
    plt.xlabel("t"); plt.ylabel("y(t)")
    plt.legend()
    plt.savefig(str(int(tn/h))+' steps.png', dpi=600)
    plt.show()

# %%