#%%
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt
import seaborn as sns
import sys
import datetime
from multiprocessing import Pool
import os
from scipy.stats import norm

sns.set_style('darkgrid')
sns.set(color_codes=True)
sns.set_context('paper')
sns.set_palette("viridis")

def likelyhood(x, mu, sigma):
    """ Probability of data set given the model """
    return np.sum(
        np.log(
            1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp( -(x - mu)**2 / (2 * sigma**2))
            )
        )

#%%
# Loading the dataset
data = np.genfromtxt('./datasets/dataset.txt')

#%%

def MarchovChain(data, N=1000):
    """ Calculates the Marchov Chain for a given data set. The calculated mean
    and std should converge to the true values. """
    
     # stores the values for the plots
    means = []; stds = []
    
    # Random plus or minus operators 
    op1 = np.random.choice([1, -1])
    op2 = np.random.choice([1, -1])
    
    # Initial values with added random fraction
    mu = np.mean(data) + op1 * (np.mean(data) * np.random.rand())
    sigma = np.std(data) + op2 * (np.std(data) * np.random.rand())
    
    # Checks whether sigma is acceptable (> 0) 
    count = 0
    while sigma < 0:
        # Generates a new sigma if value is not acceptable
        op2 = np.random.choice([1, -1])
        sigma = np.std(data) + op2 * (np.std(data) * np.random.rand())
        if count > 100: # safety condition to prevent endless loop
            print('Sigma is smaller than 0! Endless loop.\n')
            return 0
        
    # Prints the starting values vs. the true values
    print('True Mean:', np.mean(data), '\nInitial value:', mu, '\n')
    print('True Std:', np.std(data), '\nInitial value:', sigma, '\n')

    # Initial likelyhood    
    L_old = likelyhood(data, mu, sigma)
     
    for i in range(N):
        """ N-steps of the Metropolis-Hastings Algorithm to calculate 
        the mean and the standard deviation """
         
        # Calculate new proposal values using noise sampled from Gauss distribution
        mu_prop = norm(mu, 0.1).rvs()
        sigma_prop = norm(sigma, 0.1).rvs()
        
        # Generates a new sigma_prob if value is not acceptable
        t = 0
        while sigma_prop < 0: # not acceptable
            # Calculate new sigma
            sigma_prop = norm(sigma, 0.1).rvs()
            t += 1
            if t > 100: # safety condition to prevent endless loop
                print('Sigma is smaller than 0! Endless loop.\n')
                return 0
            
        # Likelyhood of new proposal
        L_proposal = likelyhood(data, mu_prop, sigma_prop)
    
        """ Check whether to accept or reject the new proposal """
        if L_proposal > L_old: # Accept the proposal
            # Overwrites old with new values
            mu = mu_prop
            sigma = sigma_prop
            L_old = L_proposal
        else: # Maybe acceptable
            a = np.exp(L_proposal - L_old)
            r = np.random.rand()
            if a > r: # Accept the proposal
                # Overwrites old with new values
                mu = mu_prop
                sigma = sigma_prop 
                L_old = L_proposal
        
        # Record the calculated means and the standard deviations
        means.append(mu)
        stds.append(sigma)
        
    return means, stds

#%%

N=3000
means, stds = MarchovChain(data, N)
#sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})

# Means
plt.figure()
plt.title('MC - Convergence to true Mean (µ)', fontsize=12 ,
            fontweight='bold')
plt.plot(range(N), means, lw=2, c='b')
# true value
plt.plot(range(N), [np.mean(data)] * N, lw=3,
                    ls='dotted', c='black', label='True value')

plt.xlabel("Number of trials (N)", fontweight='bold')
plt.ylabel("Mean (µ)" , fontweight='bold')
plt.legend()
plt.savefig('4.1.png', dpi=600)
plt.show()

# Stds
plt.figure()
plt.title('MC - Convergence to true Standard Deviation', fontsize=12 ,
            fontweight='bold')
# approx. pi-values
plt.plot(range(N), stds, lw=2, c='b')
# true value
plt.plot(range(N), [np.std(data)] * N, lw=3,
                    ls='dotted', c='black', label='True value')

plt.xlabel("Number of trials (N)", fontweight='bold')
plt.ylabel("Standard Deviation (σ)" , fontweight='bold')
plt.legend()
plt.savefig('4.2.png', dpi=600)
plt.show()

# mu vs. sigma
plt.figure()
plt.title('Mean vs. Standard Deviation', fontsize=12 ,
            fontweight='bold')

plt.scatter(np.mean(data), np.std(data), c='g', s=50, label="True solution", zorder=2)
plt.scatter(means[0], stds[0], c='r', s=50, label="Initial proposal", zorder=2)
plt.plot(means, stds, lw=0.7, c='b', zorder=1, alpha=0.8)

plt.xlabel("µ", fontweight='bold')
plt.ylabel("σ" , fontweight='bold')
plt.legend()
plt.savefig('4.3.png', dpi=600)
plt.show()

# histogram
plt.figure()

# pdf-function using best fit  
x = np.linspace(min(data), max(data), 200)
plt.plot(x, norm(means[-1], stds[-1]).pdf(x),
         c='r', lw=2, label="PDF")

# Histogram
plt.hist(data, bins=30, density=True, 
         histtype='barstacked', alpha=0.5,
         color='b')

#plt.xlabel("µ", fontweight='bold')
plt.ylabel("Density" , fontweight='bold')
plt.legend()
plt.savefig('4.4.png', dpi=600)
plt.show()
#%%