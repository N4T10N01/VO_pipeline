import cv2
import numpy as np
from dbscan_ransac import dbscan_ransac
from classic_system import estimate_motion

img1 = cv2.imread('test_imgs\handheld_check1.jpg')
img2 = cv2.imread('test_imgs\handheld_check2.jpg')

img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_bw = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# --Camera Intrinsics-----
K = np.array([
    [525.0, 0, 320.0],
    [0, 525.0, 240.0],
    [0, 0, 1]
])

# -----------------------------
# Feature detector
# -----------------------------
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# -----------------------------
# Pose state
# -----------------------------
R_global = np.eye(3)
t_global = np.zeros((3,1))

trajectory = []

# out1 = cv2.VideoWriter(
#     "output1.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     30,
#     (640, 568)
# )

out2 = cv2.VideoWriter(
    "output2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (600, 580)
)


# -----------------------------
# Visualization canvas
# -----------------------------

def draw_trajectory(traj_img, t, prev_point, scale=5):
    center_x = traj_img.shape[1] // 2
    center_y = traj_img.shape[0] // 2

    x = int(scale * t[0] + center_x)
    y = int(center_y - scale * t[2])

    if prev_point is not None:
        cv2.line(traj_img, prev_point, (x, y), (0,0,255), 2)

    return traj_img, (x, y)


# -----------------------------
# Main VO loop
# -----------------------------
def run_vo(video_path):
    positions = []
    global R_global, t_global

    cap = cv2.VideoCapture(video_path)
    prev_point = None
    traj_img = np.ones((580,600,3), dtype=np.uint8) * 255
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    current_pose = np.eye(4)
    frame_id = 0
    # --- Initial features (replace ORB) ---
    pts_prev = cv2.goodFeaturesToTrack(
    prev_gray,
    maxCorners=2000,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
    )
    lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #----------- tracking with ORB --------------------
        # kp2, des2 = orb.detectAndCompute(gray, None)

        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)[:1000]
        #---------------------------------------------------------

        #-----------tracking with KLT -----------------------------
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)

        mask = status.flatten() == 1
        pts1 = pts_prev[mask].reshape(-1,2)
        pts2 = pts_curr[mask].reshape(-1,2)

        # --- Optional: forward-backward check (strongly recommended) ---
        # pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        #     gray, pts1, pts2, None, **lk_params
        # )

        # fb_error = np.linalg.norm(pts1 - pts_back, axis=1)
        # fb_mask = fb_error < 1.0
        # fb_mask = status.flatten()

        # pts1 = pts1[fb_mask].reshape(-1, 2)
        # pts2 = pts2[fb_mask].reshape(-1, 2)

        #---------------------------------------------------
        
        #-----------filter points---------------------------
        matches = dbscan_ransac(pts1, pts2)
        #---------------------------------------------------
        # --- Estimate motion via epipolar geometry ---
        pts1 = pts1[matches]
        pts2 = pts2[matches]

        # --- Estimate motion via epipolar geometry ---
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

        if E is None:
            continue

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

        perturbation = np.eye(4)
        perturbation[:3,:3] = R
        perturbation[:3,3] = t.flatten()
        # perturbation =  estimate_motion_epipolar(kp1, kp2, matches, K, T = current_pose)
        current_pose = perturbation @ current_pose

        R = current_pose[:3,:3]
        t = current_pose[:3,3].reshape(3,1)

        # --- Normalize translation (fix scale) ---
        t = t / np.linalg.norm(t)

        # --- Integrate pose ---
        t_global += R_global @ t
        R_global = R @ R_global

        trajectory.append(t_global.copy())

        # --- Draw matches ---

          # --- Draw trajectory ---
        traj_img, prev_point = draw_trajectory(traj_img.copy(), t_global.flatten(), prev_point)
        positions.append((int(t.flatten()[0] * 50 + 300), int(t.flatten()[2] * 50 + 300)))
        traj_vis = traj_img

        # --- Show ---
        cv2.imshow("Trajectory", traj_vis)

        out2.write(traj_vis)

        # --- Prepare next iteration ---
        prev_frame = frame
        prev_gray = gray
        pts_prev = pts_curr
        # des1 = des2
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.imwrite("./ORB_cv2.png", traj_vis)
    cap.release()
    # out1.release()
    out2.release()
    cv2.destroyAllWindows()

run_vo("./test_imgs/trail_walk.mp4")