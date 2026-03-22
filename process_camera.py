import numpy as np
import cv2 
import pyrealsense2 as rs
import numpy as np
from dbscan_ransac import dbscan_ransac
from classic_system import estimate_motion

prev_point = None
traj_img = np.ones((1600,1600,3), dtype=np.uint8) * 255

def draw_trajectory(traj_img, t, prev_point):
    x = int(5*t[0] + 300)
    y = int(5*t[2] + 300)

    if prev_point is not None:
        cv2.line(traj_img, prev_point, (x, y), (0,0,255), 2)
    
    return traj_img, (x, y)

def project_to_3D(pts, depth, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u = pts[:,0]
    v = pts[:,1]

    # integer pixel indices
    u_i = np.clip(u.astype(int), 0, depth.shape[1]-1)
    v_i = np.clip(v.astype(int), 0, depth.shape[0]-1)

    z = depth[v_i, u_i]

    valid = z > 0

    X = (u[valid] - cx) * z[valid] / fx
    Y = (v[valid] - cy) * z[valid] / fy
    Z = z[valid]

    pts_3d = np.stack([X, Y, Z], axis=1)

    return pts_3d, valid

def complete_depth_filter(depth_frame, depth_scale, spatial, temporal):
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)

    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

    mask = depth > 0

    depth[~mask] = np.nan

    depth_filled = np.nan_to_num(depth)

    filtered = cv2.bilateralFilter(depth_filled, 5, 0.05, 5)

    return filtered

# Acquire provided depth filters
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# Open stream to obtain frames
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Obtain intrinsics
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

K = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])
print(K)
# Initialize Pose
current_pose = np.eye(4)

# Get first frame
prev_frame = pipeline.wait_for_frames()
prev_frame = align.process(prev_frame)
prev_color = np.asanyarray(prev_frame.get_color_frame().get_data())
prev_depth = prev_frame.get_depth_frame()

prev_depth = complete_depth_filter(prev_depth, depth_scale, spatial, temporal)

prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

pts_prev = cv2.goodFeaturesToTrack(
prev_gray,
maxCorners=2000,
qualityLevel=0.01,
minDistance=6,
blockSize=6
)

while True:
    #prev points are monotonically decreasing from fitlers, must refetch once count gets too low for tracking
    match_img = None
    if len(pts_prev) < 20: #tunable
        print("FAILURE OCCURRED")
        prev_frame = pipeline.wait_for_frames()
        if not prev_frame:
            continue
        print("made it here")
        prev_frame = align.process(prev_frame)
        prev_color = np.asanyarray(prev_frame.get_color_frame().get_data())
        prev_depth = prev_frame.get_depth_frame()

        prev_depth = complete_depth_filter(prev_depth, depth_scale, spatial, temporal)

        prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

        pts_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=2000,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=6
        )
        print(len(pts_prev))

    curr_frame = pipeline.wait_for_frames()
    if not curr_frame:
        continue


    curr_frame = align.process(curr_frame)
    curr_color = np.asanyarray( curr_frame.get_color_frame().get_data())
    curr_depth = curr_frame.get_depth_frame()

    curr_depth = complete_depth_filter(curr_depth,  depth_scale, spatial, temporal)

    curr_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)
    #-----------------done fetching info of images here------------------

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
    mask = status.flatten() == 1
    pts_prev = pts_prev[mask].reshape(-1,2)
    pts_curr = pts_curr[mask].reshape(-1,2)

    points_3d, mask = project_to_3D(pts_prev, prev_depth, K)
    #points that have associated depth
    pts_prev_valid = pts_prev[mask]
    pts_curr_valid = pts_curr[mask]

    if False and len(pts_prev_valid) < 20:

        # --- Estimate motion via epipolar geometry of cv2 ---
        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, K, method=cv2.RANSAC)

        pts_prev = pts_prev[mask]
        pts_curr = pts_curr[mask]

        if E is None:
            continue

        _, R, t, mask = cv2.recoverPose(E, pts_prev[mask], pts_curr[mask], K)

        pts_curr = pts_curr[mask] #may not want to re-mask here

        perturbation = np.eye(4)
        perturbation[:3,:3] = R
        perturbation[:3,3] = t.flatten()

    else:
        # ---With sufficient depth, do Lie Algebra -----

        
        print(len(pts_prev_valid))
        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_prev_valid]
        kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_curr_valid]
    

        matches = []
        for i in range(len(pts_prev_valid)):
            m = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0,
                        _distance=float(np.linalg.norm(pts_curr[i] - pts_prev[i])))
            matches.append(m)
        print(len(matches))

        if len(pts_prev_valid) > 200:
            matches = dbscan_ransac(kp1, kp2, matches)
            pts_curr = np.array([pts_curr[m.trainIdx] for m in matches], dtype=np.float32) # may not want to re-mask here

        match_img = cv2.drawMatches(
            prev_color, kp1,
            curr_color, kp2,
            matches[:50], None,
            flags=2
        )

        perturbation  = estimate_motion(pts_curr, points_3d, K, current_pose)

    current_pose = perturbation @ current_pose

    traj_img, prev_point = draw_trajectory(traj_img.copy(), current_pose[:3,3], prev_point)
    traj_vis = traj_img
    if pts_prev.shape == pts_curr.shape:
        print(np.linalg.norm(pts_prev-pts_curr))
    cv2.imshow("Trajectory", traj_vis)


    if match_img is not None:
        
        cv2.imshow("live_feed", match_img)
    if cv2.waitKey(1) == 27:
        break

    prev_depth = curr_depth
    prev_gray = curr_gray
    prev_color = curr_color
    pts_prev = pts_curr

cv2.imwrite("./KLT_classic.png", traj_vis)
cv2.destroyAllWindows()


