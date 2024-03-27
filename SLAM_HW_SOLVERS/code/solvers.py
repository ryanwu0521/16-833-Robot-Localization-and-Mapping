'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    x = inv(A.T @ A) @ A.T @ b
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    LU = splu(A.T @ A, permc_spec='NATURAL')
    x = LU.solve(A.T @ b)
    # U = eye(A.shape[1])
    U = LU.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    LU_comlamd = splu(A.T @ A, permc_spec='COLAMD')
    x = LU_comlamd.solve(A.T @ b)
    U = LU_comlamd.U
    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    Q, R, E, rank = rz(A, b, permc_spec='NATURAL')
    x = spsolve_triangular(R, Q, lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    Q, R, E, rank = rz(A, b, permc_spec='COLAMD')
    x = spsolve_triangular(R, Q, lower=False)   
    
    # E is the permutation matrix
    E = permutation_vector_to_matrix(E)
    x = E @ x
    return x, R

def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matrix
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)

# command line to run the code
# (2d_linear.npz)) 
# python code/linear.py data/2d_linear.npz --method default pinv lu qr lu_colamd qr_colamd

# (2d_linear_loop.npz))
# python code/linear.py data/2d_linear_loop.npz --method default pinv lu qr lu_colamd qr_colamd

# (2d_nonlinear.npz))
# python code/nonlinear.py data/2d_nonlinear.npz --method default pinv lu qr lu_colamd qr_colamd