'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
'''
    The EKF Algorithm : implementaion by Ryan Wu for 16-833 HW2.
    References: Probabilistic Robotics by Thrun et al.
                24-677 Modern Control Theory (Personal Past Class Projects)
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor((angle_rad + np.pi) / (2 * np.pi))

    return angle_rad

# non-linear measurement fucntion
def _h (X, k):
    
    # extract pose x y theta
    x, y, theta = X[:3].ravel()
    
    # initialize measurement function h
    h = np.zeros((2 * k, 1))
    
    # loop calculation for each landmark
    for i in range(k):
        # landmark position
        lx, ly = X[3 + 2 * i:3 + 2 * i + 2].ravel()
        
        # calculate the expected landmark measurements
        dx = lx - x
        dy = ly - y
        pow_dxdy = dx**2 + dy**2
        sqr_dxdy = np.sqrt(pow_dxdy)
        b_hat = np.arctan2(dy, dx) - theta
        r_hat = sqr_dxdy
        
        # update the measurement function h
        h[2 * i] = warp2pi(b_hat)
        h[2 * i + 1] = r_hat

    return h


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''
    # number of landmarks
    k = init_measure.shape[0] // 2
    # initial pose x y theta
    x, y, theta = init_pose.ravel()

    # initialize landmarks arrays and covariance
    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    # loop calculation for landmark position and covariance
    for i in range(k):
        # measurements beta (bearing angle) and r (range)
        beta = warp2pi(init_measure[2 * i, 0])
        r = init_measure[2 * i + 1, 0]
        
        # calculate landmark position
        lx = x + r * np.cos(theta + beta) 
        ly = y + r * np.sin(theta + beta)
        landmark[2 * i] = lx
        landmark[2 * i + 1] = ly

        # motion model with respect to the state
        Lp = np.array([[-r * np.sin(theta + beta), np.cos(theta + beta)],
                      [r * np.cos(theta + beta), np.sin(theta + beta)]])
        # measurement model with respect to the state
        Ll = np.array([[1, 0, -r * np.sin(theta + beta)],
                       [0, 1, r * np.cos(theta + beta)]])
    
        # update landmark covariance
        landmark_cov[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Lp @ init_measure_cov @ Lp.T + Ll @ init_pose_cov @ Ll.T

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # extract initial pose x, y, theta
    x, y, theta = X[:3].ravel()

    # extract control inputs delta and alpha
    delta, alpha = control.ravel()

    # initialize state vector X and Convariance matrix P
    X_pre = np.zeros((3 + 2 * k, 1))
    P_pre = np.zeros((3 + 2 * k, 3 + 2 * k))

    # predicted next pose
    x_pre = x + delta * np.cos(theta)
    y_pre = y + delta * np.sin(theta)
    theta_pre = warp2pi(theta + alpha)

    # Jacobian of the motion model
    Gt = np.eye (3 + 2 * k)
    Gt[:3, :3] = np.array([[1, 0, -delta * np.sin(theta)],
                  [0, 1, delta * np.cos(theta)],
                  [0, 0, 1]])

    # Jacobian of the control model (rotaion matrix)
    At = np.zeros((3 + 2 * k, 3 + 2 * k))
    At[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    
    # predict state vector X
    X_pre = np.vstack([x_pre, y_pre, theta_pre, X[3:]])
    
    # expand control_cov matrix to match size (15 by 15)
    control_cov_expand = np.zeros((3 + 2 * k, 3 + 2 * k))
    control_cov_expand[:3, :3] = control_cov

    # predict covariance P
    P_pre = Gt @ P @ Gt.T + At @ control_cov_expand @ At.T
    
    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # extract predicted pose x, y, theta
    x, y, theta = X_pre[:3].ravel()

    # initialize state vector X and Convariance matrix P
    X = X_pre.copy()
    P = P_pre.copy()

    # intialize measurement Jacobian Ht and Hl
    Ht = np.zeros((2 * k, 3 + 2 * k))   # (12 by 15)
    diff = np.zeros((2 * k, 1))         # (12 by 1)

    # loop calculation for each landmark
    for i in range(k):
        # landmark position
        lx, ly = X_pre[3 + 2 * i:3 + 2 * i + 2].ravel()

        # calculate the expected landmark measurements
        dx = lx - x
        dy = ly - y
        pow_dxdy = dx**2 + dy**2
        sqr_dxdy = np.sqrt(pow_dxdy)
        
        # calculate the Jacobians
        # with respect to the robot pose 
        Hp = np.array([[dy / pow_dxdy, -dx / pow_dxdy, -1],
                       [-dx / sqr_dxdy, -dy / sqr_dxdy, 0]])
        # with respect to the landmark measurement
        Hl = np.array([[-dy / pow_dxdy, dx / pow_dxdy],
                       [dx / sqr_dxdy, dy / sqr_dxdy]])
        
        # Combine the Jacobians into the Jacobian Ht
        Ht[2 * i:2 * i + 2, 0:3] = Hp
        Ht[2 * i:2 * i + 2, 3 + 2 * i:3 + 2 * i + 2] = Hl

        # calculate the expected landmark measurements
        diff = measure - _h(X_pre, k)
    
    # measurement noise covariance matrix Qt (12 by 12)
    Qt = np.kron(np.eye(k), measure_cov)
  
    # calculate the Kalman gain (15 by 12)
    Kt = P_pre @ Ht.T @ np.linalg.inv(Ht @ P_pre @ Ht.T + Qt)

    # update the state vector X
    X = X_pre + Kt @ diff

    # update the covariance matrix P
    P = (np.eye(3 + 2 * k) - Kt @ Ht) @ P_pre

    return X, P


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    # ground truth landmarks
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    
    # loop Euclidean and Mahalanobis calculation for each landmark
    for i in range(k):
        lx, ly = X[3 + 2 * i:3 + 2 * i + 2].ravel()
        euc = np.linalg.norm(l_true[2 * i:2 * i + 2] - np.array([lx, ly]), 2)
        mah = np.sqrt((l_true[2 * i:2 * i + 2] - np.array([lx, ly])).T @ np.linalg.inv(P[3 + 2 * i:3 + 2 * i + 2, 3 + 2 * i:3 + 2 * i + 2]) @ (l_true[2 * i:2 * i + 2] - np.array([lx, ly])))
        print(f"Landmark {i+1}:")
        print(f"\tEuclidean distance: {euc}")
        print(f"\tMahalanobis distance: {mah}")

    # print the final state converiance matrix P
    print ("P" , P)

    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    # sig_x = 2.5
    # sig_x = 0.025
    sig_y = 0.1
    # sig_y = 1
    # sig_y = 0.01
    sig_alpha = 0.1
    # sig_alpha = 1
    # sig_alpha = 0.01
    sig_beta = 0.01
    # sig_beta = 0.1
    # sig_beta = 0.001
    sig_r = 0.08
    # sig_r = 0.8
    # sig_r = 0.008


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("SLAM_HW_EKF/data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)

if __name__ == "__main__":
    main()