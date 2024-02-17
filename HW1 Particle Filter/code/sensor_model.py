'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
'''
    Sensor model implementation by Jeremy Kilbride and Ryan Wu for 16833 HW1.
'''

import numpy as np
import math
import time
import ray_casting_lib
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # Original numbers
        self._z_hit = 1.22
        self._z_short = 0.15
        self._z_max = 0.1
        self._z_rand = 550

        self._sigma_hit = 63.5
        self._lambda_short = 0.1

        # my numbers
        # self._z_hit = 1
        # self._z_short = 0.4
        # self._z_max = 0.002
        # self._z_rand = 0.5

        # self._sigma_hit = 50
        # self._lambda_short = 0.0027

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # occupancy map
        self.map=occupancy_map

        # map resolution
        self._map_resolution = 10

    # ray casting function in python
    # def ray_casting(self, x_t1, map, verbose=False):
    #     """
    #     param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    #     param[in] src_path_map : path to the map file
    #     param[out] z_t1_arr_star : expected range scan at time t
    #     """
    #     """
    #     TODO : Add your code here
    #     """

    #     # get the map and its size
    #     grid_size = self._map_resolution

    #     # get the number of particles
    #     num_particles = x_t1.shape[0]

    #     # get the subsampling
    #     subsample = self._subsampling

    #     # get the expected range values for each particle
    #     z_t1_arr_star = np.zeros((num_particles, int(180/subsample)))

    #     # iterate over each particle
    #     for particle in range(num_particles):
    #         dir = x_t1[particle, 2]        # get heading
    #         current_dir = dir - np.pi/2    # subtract 90 degrees
    #         start_point = x_t1[particle, :2]

    #         for beam in range(int(180/subsample)):
    #             signY_dir = 1
    #             signX_dir = 1
    #             if np.cos(current_dir) < 0:
    #                 signX_dir = -1
    #             if np.sin(current_dir) < 0:
    #                 signY_dir = -1
    #             grid_X = int(start_point[0] // grid_size)
    #             grid_Y = int(start_point[1] // grid_size)
    #             current_point = start_point
    #             current_dist = 0
    #             if verbose:
    #                 print(start_point)
    #             while map[grid_Y, grid_X] <= 0.01:
    #                 if verbose:
    #                     print(f"X:{grid_X}, Y:{grid_Y}")
    #                 dtX = ((grid_X) * grid_size - current_point[0]) / np.cos(current_dir)
    #                 dtY = ((grid_Y) * grid_size - current_point[1]) / np.sin(current_dir)
    #                 dt = np.min([dtX, dtY])
    #                 next_point = np.array([current_point[0] + dt * np.cos(current_dir), current_point[1] + dt * np.sin(current_dir)])
    #                 current_dist += dt
    #                 current_point = next_point
    #                 if dtX < dtY:
    #                     grid_X += signX_dir
    #                 else:
    #                     grid_Y += signY_dir
    #                 if grid_X >= 800 or grid_Y >= 800 or grid_X < 0 or grid_Y < 0 or map[grid_Y, grid_X] == -1:
    #                     break
    #             if current_dist==0:
    #                 current_dist=5
    #             z_t1_arr_star[particle, beam] = current_dist
    #             current_dir += (np.pi / (180 / subsample))
    
    #     return z_t1_arr_star

    # ray casting function in c++
    def ray_casting(self, x_t1, map):
        SensorModel = ray_casting_lib.SensorModel()
        z_t1_arr_star = SensorModel.ray_casting(x_t1, self.map)

        return z_t1_arr_star

          
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # [Beam range finder model Table 6.1]

        # get the number of particles
        num_particles = x_t1.shape[0]

        # z_t1_arr_star (expected) for Ray Casting
        # z_t1_arr_star = self.ray_casting(x_t1, self.map, verbose=False) 
        z_t1_arr_star = self.ray_casting(x_t1, self.map)
        z_t1_arr_star = np.array(z_t1_arr_star, dtype=np.float64)
        # z_t1_arr_star = self.ray_casting(x_t1)
        # subsample z_t1_arr_star
        # z_t1_subsampled = z_t1_arr[:180:2]
        z_t1_subsampled = z_t1_arr[:360: self._subsampling]

        # initialize the probability
        prob_zt1 = np.zeros(num_particles)

        # iterate over each particle
        for particle in range(num_particles): 
            # p_hit Gaussian distribution (Equations 6.4 - 6.6)
            N_exp = (-1/2) * (z_t1_subsampled - z_t1_arr_star[particle, :])**2 / self._sigma_hit**2
            N = 1 / np.sqrt(2 * np.pi * self._sigma_hit**2) * np.exp(N_exp)
            # eta_hit = 1 / (np.sum(N))        # normalizer (not sure if this is right)
            # p_hit = eta_hit * N
            p_hit = N
            
            # p_short exponential distribution (Equation 6.7 - 6.10)
            # eta_short = 1 / (1 - np.exp(-1*self._lambda_short * z_t1_arr_star[particle, :]))        # normalizer
            eta_short= 1
            #     eta=1/(1-np.exp(-1*ls*z_expected[particle,:]))
            p_short = eta_short * self._lambda_short * np.exp(-1*self._lambda_short * z_t1_subsampled)

            # p_max uniform distribution (Equation 6.11)
            p_max = (z_t1_subsampled == self._max_range).astype(float)

            # p_rand uniform distribution (Equation 6.12)
            p_rand = 1 / self._max_range * (z_t1_subsampled * (self._max_range > z_t1_subsampled) > 0).astype(float)

            # calculate the total probability
            prob_zt1_particle = self._z_hit * p_hit + self._z_short * p_short + self._z_max * p_max + self._z_rand * p_rand
            log_probs_zt1_particle = np.log(prob_zt1_particle)
            prob_zt1[particle] = np.exp(np.sum(log_probs_zt1_particle))

        #now normalize the probabilities
        prob_sum=np.sum(prob_zt1)
        prob_zt1=np.divide(prob_zt1,prob_sum)


        return prob_zt1
    
