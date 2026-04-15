import gtsam
from gtsam.symbol_shorthand import X, L
import numpy as np
from process_camera import *


graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
K_gtsam = gtsam.Cal3_S2(fx, fy, 0, cx, cy)

# Pixel noise
pixel_sigma = 2.0
base_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)

noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber(1.345),
    base_noise
)

# Anchor first pose
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), prior_noise))
initial.insert(X(0), gtsam.Pose3())

# Landmark tracking
landmark_ids = None
next_landmark_id = 0

current_pose = np.eye(4)

perturbation = np.eye(4)


Q = 1e-4 * np.eye(6)

kf = SE3KalmanFilter(
    init_pose=current_pose,
    init_cov= Q
)

# Get first frame
prev_frame = pipeline.wait_for_frames()
prev_frame = align.process(prev_frame)
prev_color = np.asanyarray(prev_frame.get_color_frame().get_data())
prev_depth = prev_frame.get_depth_frame()

prev_depth = complete_depth_filter(prev_depth, depth_scale, spatial, temporal, hole_filling)

prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

pts_prev = cv2.goodFeaturesToTrack(
prev_gray,
maxCorners=2000,
qualityLevel=0.01,
minDistance=6,
blockSize=6
)
loop_count =0

while True:
    loop_count+=1

    match_img = None
    if len(pts_prev) < 20: #tunable

        prev_frame = pipeline.wait_for_frames()
        if not prev_frame:
            continue
        prev_frame = align.process(prev_frame)
        prev_color = np.asanyarray(prev_frame.get_color_frame().get_data())
        prev_depth = prev_frame.get_depth_frame()

        prev_depth = complete_depth_filter(prev_depth, depth_scale, spatial, temporal, hole_filling)

        prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

        pts_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=2000,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=6
        )

    curr_frame = pipeline.wait_for_frames()
    if not curr_frame:
        continue


    curr_frame = align.process(curr_frame)
    curr_color = np.asanyarray( curr_frame.get_color_frame().get_data())
    curr_depth = curr_frame.get_depth_frame()

    curr_depth = complete_depth_filter(curr_depth,  depth_scale, spatial, temporal, hole_filling)

    depth = np.nan_to_num(curr_depth, nan=0.0, posinf=0.0, neginf=0.0)

    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    depth_uint8 = depth_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
    #-----------------done fetching info of images here------------------

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
    # pixel smoothing
    pts_curr = 0.7 * pts_curr + 0.3 * pts_prev
    mask = status.flatten() == 1
    pts_prev = pts_prev[mask].reshape(-1,2)
    pts_curr = pts_curr[mask].reshape(-1,2)

    points_3d, mask = project_to_3D(pts_prev, prev_depth, K)
    #points that have associated depth
    pts_prev_valid = pts_prev[mask]
    pts_curr_valid = pts_curr[mask]

    if landmark_ids is None:
        landmark_ids = np.arange(len(pts_prev))
        next_landmark_id = len(landmark_ids)

    # Keep only valid tracked points
    landmark_ids = landmark_ids[mask]
    ids_valid = landmark_ids

    for i, pid in enumerate(ids_valid):
        key = L(int(pid))
        if not initial.exists(key):
            X3, Y3, Z3 = points_3d[i]
            if Z3 > 1e-3:  # avoid bad depth
                initial.insert(key, gtsam.Point3(X3, Y3, Z3))


    # Initialize current pose
    if not initial.exists(X(loop_count)):
        initial.insert(X(loop_count), gtsam.Pose3())

    # Add projection factors
    for i, pid in enumerate(ids_valid):

        if i >= len(points_3d):
            continue

        Z3 = points_3d[i][2]
        if Z3 < 1e-3:
            continue

        landmark_key = L(int(pid))

        #depth-dependent noise 
        sigma = pixel_sigma + 0.01 * (Z3 ** 2)
        depth_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
        robust_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.345),
            depth_noise
        )

        u1, v1 = pts_prev_valid[i]
        graph.add(
            gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(float(u1), float(v1)),
                robust_noise,
                X(loop_count - 1),
                landmark_key,
                K_gtsam
            )
        )

        # Current frame observation
        u2, v2 = pts_curr_valid[i]
        graph.add(
            gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(float(u2), float(v2)),
                robust_noise,
                X(loop_count),
                landmark_key,
                K_gtsam
            )
        )

    if loop_count % 10 == 0 and loop_count > 0:

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        initial = result
        
        pose = result.atPose3(X(loop_count))
        current_pose = pose.matrix()

        print("Optimized pose:\n", current_pose)

    traj_img, prev_point = draw_trajectory(traj_img.copy(), current_pose[:3,3], prev_point)
    traj_vis = traj_img
    cv2.imshow("Trajectory", traj_vis)
    vis = draw_keypoints(vis, pts_curr)
    cv2.imshow("keypoints", vis)
    if cv2.waitKey(1) == 27:
        break

    prev_depth = curr_depth
    prev_gray = curr_gray
    prev_color = curr_color
    pts_prev = pts_curr

        