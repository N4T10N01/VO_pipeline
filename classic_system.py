import numpy as np
import cv2
from lie_algebra_utilities import *

def projection_jacobian_normalized(X):
    X, Y, Z = X

    J = np.array([
        [1.0 / Z, 0.0, -X / (Z * Z)],
        [0.0, 1.0 / Z, -Y / (Z * Z)]
    ])
    return J

def projection_jacobian(K, x):
    X, Y, Z = x
    fx, fy = K[0,0], K[1,1]

    J = np.array([
        [fx/Z, 0, -fx*X/(Z*Z)],
        [0, fy/Z, -fy*Y/(Z*Z)]
    ])
    return J

def pose_jacobian(x):
    return np.hstack([np.eye(3), -1*hat(x)])
    

def build_system(T, points_3d, pts2d, K):
    
    J_list = []
    r_list = []

    R = T[:3,:3]
    t = T[:3,3]

    for P, z in zip(points_3d, pts2d):
        # x = R @ P + t
        x = P
        z_hat = project(K, x)
        
        r = z - z_hat #classic constraint
        J_proj = projection_jacobian(K, x)
        J_pose = pose_jacobian(x)

        J = J_proj @ J_pose

        J_list.append(J)
        r_list.append(r)

    J = np.vstack(J_list)
    r = np.hstack(r_list)

    return J, r

def build_system_normalized(T, points_3d, pts2d, K):
    J_list = []
    r_list = []

    norm_pts2d = normalize_points(pts2d, K)

    R = T[:3,:3]
    t = T[:3,3]

    for P, z in zip(points_3d, norm_pts2d):
        # x = R @ P + t
        x = P
        J_proj = projection_jacobian_normalized(x)
        J_pose = pose_jacobian(x)

        z_hat = x/x[2]

        r = z[:2] - z_hat[:2] 

        J = J_proj @ J_pose

        J_list.append(J)
        r_list.append(r)

    J = np.vstack(J_list)
    r = np.hstack(r_list)

    return J, r


def estimate_motion(pts2d, points_3d, K, T):
    
    J, r = build_system_normalized(T, points_3d, pts2d, K)

    # delta = np.linalg.lstsq(J, -r, rcond=None)[0] #Original without damping
 
    tau = 1e-3  # initial damping scale
    H = J.T @ J
    g = J.T @ r
    lambda_ = tau * np.max(np.diag(H))

    # --- Solve damped system ---
    delta = np.linalg.solve(H + lambda_ * np.eye(H.shape[0]), -g)
    
    # Threshold: on a scale of meters, motion below this is on the scale of mm and akin to noise
    epsilon = 1e-3

    delta = delta * (np.abs(delta) >= epsilon)

    T_final = exp_se3(delta)

    # noise modelling: compute empirical covariance
    residuals = r.flatten()  # ensure 1D
    dof = len(residuals) - 6
    sigma2 = (residuals @ residuals) / max(dof, 1)
    Sigma = sigma2 * np.linalg.pinv(H)
    Sigma = np.clip(Sigma, -1e3, 1e2)  # Clip to prevent extreme values

    return T_final, Sigma
