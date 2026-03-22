import numpy as np
import cv2
from lie_algebra_utilities import *

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
        x = R @ P + t
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

def estimate_motion(kp2, points_3d, K, T):
    # pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = kp2 #np.array([kp2[m.trainIdx].pt for m in matches])
    # print(pts2)
    # print(points_3d)
    # pts1 = normalize_points(pts1,K)
    # pts2 = normalize_points(pts2, K)

    J, r = build_system(T, points_3d, pts2, K)

    delta = np.linalg.lstsq(J, -r, rcond=None)[0]

    # Threshold: on a scale of m, motion below this is on the scale of mm and akin to noise
    epsilon = 1e-2

    delta_filtered = delta * (np.abs(delta) >= epsilon)

    T_final = exp_se3(delta_filtered) @ T

    return T_final
