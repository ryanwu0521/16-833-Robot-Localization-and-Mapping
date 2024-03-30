'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    '''
    Initialize the state vector given odometry and observations.
    '''
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    # landmarks_mask = np.zeros((n_landmarks), dtype=np.bool)
    landmarks_mask = np.zeros((n_landmarks), dtype=bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    '''
    # TODO: return odometry estimation
    odom = np.zeros((2, ))
    rx_t = x[2 * i]
    ry_t = x[2 * i + 1]
    lx_k = x[2 * (i + 1)]
    ly_k = x[2 * (i + 1) + 1]

    dx = lx_k - rx_t
    dy = ly_k - ry_t

    # odom = np.array([[dx], [dy]])

    odom = np.array([dx, dy])

    # print('odom shape:', odom.shape)
    
    return odom


def bearing_range_estimation(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    '''
    # TODO: return bearing range estimations
    obs = np.zeros((2, ))
    rx_t = x[2 * i]
    ry_t = x[2 * i + 1]
    lx_k = x[2 * n_poses + 2 * j]
    ly_k = x[2 * n_poses + 2 * j + 1]

    dx = lx_k - rx_t
    dy = ly_k - ry_t
    d = np.sqrt(dx**2 + dy**2)
    theta = warp2pi(np.arctan2(dy, dx))

    obs = np.array([theta, d])

    # print('obs shape:', obs.shape)

    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    '''
    # TODO: return jacobian matrix
    jacobian = np.zeros((2, 4))
    rx_t = x[2 * i]
    ry_t = x[2 * i + 1]
    lx_k = x[2 * n_poses + 2 * j]
    ly_k = x[2 * n_poses + 2 * j + 1]

    dx = lx_k - rx_t
    dy = ly_k - ry_t
    d = np.sqrt(dx**2 + dy**2)
    theta = warp2pi(np.arctan2(dy, dx))

    jacobian = np.array([[dy / d**2, -dx / d**2, -dy / d**2, dx / d**2], [-dx / d, -dy / d, dx / d, dy / d]])

    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param x State vector x at which we linearize the system.
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M, ))

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    A[0:2, 0:2] = np.eye(2)

    # TODO: Then fill in odometry measurements
    # Jacobian function H0 for odometry
    Ho = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])

    for i in range(n_odom):
        A[2 + i * 2:4 + i * 2, i * 2:i * 2 + 4] = sqrt_inv_odom @ Ho
        b[2 + i * 2:4 + i * 2] = sqrt_inv_odom @ (odoms[i] - odometry_estimation(x, i))
    
    # TODO: Then fill in landmark measurements
    for i in range(n_obs):
        pose_index = int(observations[i, 0])
        landmark_index = int(observations[i, 1])
        # Jacobian functions for landmark observations H_l and bearing range estimations H_b
        H_l= compute_meas_obs_jacobian(x, pose_index, landmark_index, n_poses)
        H_b = bearing_range_estimation(x, pose_index, landmark_index, n_poses)
        
        # calculate the error between measurements and estimates for d and theta
        theta_diff = warp2pi(observations[i, 2] - H_b[0])
        d_diff = observations[i, 3] - H_b[1]

        A[2 + n_odom * 2 + i * 2:4 + n_odom * 2 + i * 2, pose_index * 2:pose_index * 2 + 2] = sqrt_inv_obs @ H_l[:, 0:2]
        A[2 + n_odom * 2 + i * 2:4 + n_odom * 2 + i * 2, n_poses * 2 + landmark_index * 2:n_poses * 2 + landmark_index * 2 + 2] = sqrt_inv_obs @ H_l[:, 2:4]
        b[2 + n_odom * 2 + i * 2:4 + n_odom * 2 + i * 2] = sqrt_inv_obs @ np.array([theta_diff, d_diff])

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='../data/2d_nonlinear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')

    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-')
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='b', marker='+')
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx
        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

# (2d_nonlinear.npz))
# python code/nonlinear.py data/2d_nonlinear.npz --method default pinv lu qr lu_colamd qr_colamd