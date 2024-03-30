'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def create_linear_system(odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)                 # number of odometry measurements
    n_obs = len(observations)           # number of landmark measurements

    M = (n_odom + 1) * 2 + n_obs * 2    # total rows in the linear system
    N = n_poses * 2 + n_landmarks * 2   # length of the state vector

    A = np.zeros((M, N))                # Jacobian matrix A
    b = np.zeros((M, ))                 # residual vector b

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    A[0:2, 0:2] = np.eye(2)

    # TODO: Then fill in odometry measurements
    # Jacobian function H0
    Ho = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    
    # loop through odometry measurements
    for i in range(n_odom):
        A[2 + i * 2:4 + i * 2, i * 2:i * 2 + 4] = sqrt_inv_odom @ Ho
        b[2 + i * 2:4 + i * 2] = sqrt_inv_odom @ odoms[i]

    # TODO: Then fill in landmark measurements
    # Jacobian function Hl
    Hl = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])

    # loop through landmark measurements
    for i in range(n_obs):
        pose_index = int(observations[i, 0])
        landmark_index = int(observations[i, 1])
        A[2 + n_odom * 2 + i * 2:4 + n_odom * 2 + i * 2, pose_index * 2:pose_index * 2 + 2] = sqrt_inv_obs @ Hl[:, 0:2]
        A[2 + n_odom * 2 + i * 2:4 + n_odom * 2 + i * 2, n_poses * 2 + landmark_index * 2:n_poses * 2 + landmark_index * 2 + 2] = sqrt_inv_obs @ Hl[:, 2:4]
        b[2 + n_odom * 2 + i * 2:4 + n_odom * 2 + i * 2] = sqrt_inv_obs @ observations[i, 2:4]
        
    return csr_matrix(A), b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to npz file')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help=
        'Number of repeats in evaluation efficiency. Increase to ensure stability.'
    )
    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt trajectory')
    plt.scatter(gt_landmarks[:, 0],
                gt_landmarks[:, 1],
                c='b',
                marker='+',
                label='gt landmarks')
    plt.legend()
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Build a linear system
    A, b = create_linear_system(odoms, observations, sigma_odom,
                                sigma_landmark, n_poses, n_landmarks)

    # Solve with the selected method
    for method in args.method:
        print(f'Applying {method}')

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f'{method} takes {total_time / total_iters}s on average')

        if R is not None:
            plt.spy(R)
            plt.show()

        traj, landmarks = devectorize_state(x, n_poses)

        # Visualize the final result
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)