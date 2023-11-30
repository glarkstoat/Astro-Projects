"""
Collection of useful, self-developed functions
"""
import numpy as np
G = 6.674*10**(-11)

def dist(x,y): 
    """ Calculates the distance of two points given as numpy arrays """
    return np.sqrt(sum((x - y) ** 2))

def check_for_prime(num):
    if num % 2 == 0:
        print(num, "is a prime number")
    for i in range(2,num):
        if (num % i) == 0:
            print(num,"is not a prime number")
            print(i,"times",num//i,"is",num)
            break
        elif (num % i) != 0 and (i == num - 1):
            print(num,"is a prime number")
 
def dist_cubed(x,y):
    return (np.linalg.norm(x-y)**3)

def vel_change(m, x, y, dt):
    """ computes the change in velocity for N=2, for a mass m, position vectors x and y, and timestep dt """
    return( -G * m / np.linalg.norm(x - y) ** 3 * (x - y) * dt )

"""def r_rel(x,y):
    ''' Realtive vector i.e. distance between them '''
    for i in range(len(input_body)): 
        for j in range(len(input_body)):
                    if i != j:
                        x_r = r[i] - r[j]"""
