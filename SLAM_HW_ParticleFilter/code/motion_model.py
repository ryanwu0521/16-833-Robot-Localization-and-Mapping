'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
'''
    Motion model implementation by Jeremy Kilbride and Ryan Wu for 16833 HW1.
'''

import sys
import numpy as np
import numpy.typing as npt

class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    # wrap angle to [-pi, pi] (all angular differences must lie in this range)
    def wrap_to_pi(self, angle):
        angle_wrap = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
        return angle_wrap

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        # [Odometry motion model Table 5.6]

        # declareing varibles for code readability
        num_particles = x_t0.shape[0]
        x = x_t0[:, 0]
        y = x_t0[:, 1]
        theta = x_t0[:, 2]
        x_bar = u_t0[0]
        y_bar = u_t0[1]
        theta_bar = u_t0[2]
        x_prime = u_t1[0]
        y_prime = u_t1[1]
        theta_prime = u_t1[2]

        # recover relative motion parameters
        delta_rot1 = np.arctan2(y_prime - y_bar, x_prime - x_bar) - theta_bar
        delta_rot1 = self.wrap_to_pi(delta_rot1)
        delta_trans = np.sqrt(np.square(x_prime - x_bar) + np.square(y_prime - y_bar))
        delta_rot2 = theta_prime - theta_bar - delta_rot1
        delta_rot2 = self.wrap_to_pi(delta_rot2)

        # calculate the corresponding relative motion parameters
        delta_hat_rot1 = delta_rot1 - np.random.normal(0, abs(self._alpha1 * delta_rot1 + self._alpha2 * delta_trans),x.shape)
        delta_hat_rot1 = self.wrap_to_pi(delta_hat_rot1)
        delta_hat_trans = delta_trans - np.random.normal(0, abs(self._alpha3 * delta_trans + self._alpha4 * (delta_rot1 + delta_rot2)),x.shape)
        delta_hat_rot2 = delta_rot2 - np.random.normal(0, abs(self._alpha1 * delta_rot2 + self._alpha2 * delta_trans),x.shape)
        delta_hat_rot2 = self.wrap_to_pi(delta_hat_rot2)

        # compute the error probabilities for the individual motion parameters
        x_t1 = np.zeros((num_particles, 3))
        x_t1[:, 0] = x + delta_hat_trans * np.cos(theta + delta_hat_rot1)
        x_t1[:, 1] = y + delta_hat_trans * np.sin(theta + delta_hat_rot1)
        x_t1[:, 2] = theta + delta_hat_rot1 + delta_hat_rot2
        x_t1[:, 2] = self.wrap_to_pi(x_t1[:, 2])
        
        return x_t1
        