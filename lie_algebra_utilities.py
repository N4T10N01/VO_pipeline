import numpy as np
import cv2

def vee(M):
    return np.array([
        M[2,1],
        M[0,2],
        M[1,0]
    ])

def log_so3(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        # First-order approximation: log(R) ≈ (R - R^T)/2
        return vee(0.5 * (R - R.T))

    elif np.pi - theta < 1e-5:
        # Near pi: special handling to avoid instability
        # Extract axis from diagonal
        A = (R + np.eye(3)) / 2.0

        axis = np.array([
            np.sqrt(max(A[0, 0], 0)),
            np.sqrt(max(A[1, 1], 0)),
            np.sqrt(max(A[2, 2], 0))
        ])

        # Fix signs using off-diagonal entries
        if R[2, 1] - R[1, 2] < 0: axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0: axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0: axis[2] = -axis[2]

        axis = axis / np.linalg.norm(axis)
        return theta * axis

    else:
        # Standard case
        phi_hat = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        return vee(phi_hat)

def log_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]

    phi = log_so3(R)
    theta = np.linalg.norm(phi)

    if theta < 1e-8:
        K = hat(phi)
        V_inv = np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)

    else:
        K = hat(phi)
        theta2 = theta * theta

        A = 1.0 / theta2
        B = (1 + np.cos(theta)) / (2 * theta * np.sin(theta))

        V_inv = (
            np.eye(3)
            - 0.5 * K
            + (A - B) * (K @ K)
        )

    rho = V_inv @ t

    xi = np.zeros(6)
    xi[:3] = rho
    xi[3:] = phi
    return xi

def hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)
    k = w / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

def exp_se3(xi):
    rho = xi[:3]
    phi = xi[3:]

    R = exp_so3(phi)

    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = hat(phi/theta)
        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K)
        )

    t = V @ rho

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def project(K, x):
    X, Y, Z = x
    #project to camera plane using focal lengths and optical ceneters of intrinsic matrix
    u = K[0,0]*X/Z + K[0,2] 
    v = K[1,1]*Y/Z + K[1,2]
    return np.array([u, v])


def normalize_points(pts, K):
    K_inv = np.linalg.inv(K)
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    return (K_inv @ pts_h.T).T


