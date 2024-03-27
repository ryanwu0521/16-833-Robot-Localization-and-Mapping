'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
'''
    Resampling implementation by Jeremy Kilbride and Ryan Wu for 16833 HW1.
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        
    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        # [Algorithm Low_variance_sampler Table 4.4]
        
        # Initialize variables
        X_bar_resampled =  np.zeros_like(X_bar)
        M = len(X_bar)                      # Number of particles   
        r = np.random.uniform(0, 1/M)       # Random number for low variance sampler
        c = X_bar[0, 3]                     # Cumulative weight
        i = 0                               # Index of the particle to be sampled
    
        # Resampling step
        for m in range(M):
            u = r + m  * (1/M)
            while u > c:
                i = (i + 1) % M
                c = c + X_bar[i, 3]
            X_bar_resampled[m, :] = X_bar[i, :]

        return X_bar_resampled