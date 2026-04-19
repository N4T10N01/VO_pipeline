import numpy as np
import cv2 
import pyrealsense2 as rs
import numpy as np
from scipy import spatial
from dbscan_ransac import dbscan_ransac
from classic_system import estimate_motion
from lie_algebra_utilities import *

out3 = cv2.VideoWriter(
    "classic_system_tracking.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (640, 480)
)

prev_point = None
traj_img = np.ones((600,600,3), dtype=np.uint8) * 255

def draw_keypoints(img, pts, color=(0,255,0)):
    for p in pts:
        x, y = int(p[0]), int(p[1])
        cv2.circle(img, (x, y), 2, color, -1)
    return img

def draw_trajectory(traj_img, t, prev_point, scale=5):
    center_x = traj_img.shape[1] // 2
    center_y = traj_img.shape[0] // 2

    x = int(scale * t[0] + center_x)
    y = int(center_y - scale * t[1])

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

    valid = z > 1e-2

    X = (u[valid] - cx) * z[valid] / fx
    Y = (v[valid] - cy) * z[valid] / fy
    Z = z[valid]

    pts_3d = np.stack([X, Y, Z], axis=1)

    return pts_3d, valid

def complete_depth_filter(depth_frame, depth_scale, spatial, temporal, hole_filling):
    depth_frame = temporal.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) #* depth_scale

    # mask = depth > 0

    # depth[~mask] = np.nan

    # depth_filled = np.nan_to_num(depth)

    # filtered = cv2.bilateralFilter(depth, 9, 0.1, 5)

    return depth

# Acquire provided depth filters
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)

temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 10)

hole_filling.set_option(rs.option.holes_fill, 2)

# Open stream to obtain frames
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()

depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
depth_sensor.set_option(rs.option.visual_preset, 3)

depth_sensor.set_option(rs.option.laser_power, 200)

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

if __name__ == "__main__":
    current_pose = np.eye(4)

    # No motion until sensed
    perturbation = np.eye(4)


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

        #prev points are monotonically decreasing from fitlers, must refetch once count gets too low for tracking
        if len(pts_prev) < 100: #tunable

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
        # Optional: ignore invalid values (NaN / inf)
        depth = np.nan_to_num(curr_depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to 0–255
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8
        depth_uint8 = depth_norm.astype(np.uint8)

        # Apply heatmap
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
        #-----------------done fetching info of images here------------------

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
        mask = status.flatten() == 1
        pts_prev = pts_prev[mask].reshape(-1,2)
        pts_curr = pts_curr[mask].reshape(-1,2)

        pts_curr = 0.9 * pts_curr + 0.1 * pts_prev

        points_3d, mask = project_to_3D(pts_prev, prev_depth, K)
        #points that have associated depth
        pts_prev = pts_prev[mask]
        pts_curr = pts_curr[mask]

        if len(pts_prev) > 20:
            # ---With sufficient depth, do Lie Algebra -----

            # kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_prev]
            # kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_curr]
        

            # matches = []
            # for i in range(len(pts_prev)):
            #     m = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0,
            #                 _distance=float(np.linalg.norm(pts_curr[i] - pts_prev[i])))
            #     matches.append(m)

            # if len(pts_prev[0]) > 200:
            #     matches = dbscan_ransac(kp1, kp2, matches)
            #     pts_curr = np.array([pts_curr[m.trainIdx] for m in matches], dtype=np.float32) # may not want to re-mask here


            perturbation, sigma = estimate_motion(pts_curr,  points_3d,K, current_pose)

            # if (log_se3(perturbation)[0]>0.1):
            #     print(log_se3(perturbation))

        # optional: filter out smal,l motions that are likely noise (tunable threshold)
        # if np.linalg.norm(log_se3(perturbation)) < 0.02:
        #     perturbation = np.eye(4)
        current_pose = perturbation @ current_pose

        traj_img, prev_point = draw_trajectory(traj_img.copy(), current_pose[:3,3], prev_point)
        traj_vis = traj_img

        cv2.imshow("Trajectory", traj_vis)

        vis = curr_color.copy()
        vis = draw_keypoints(vis, pts_curr)
        cv2.imshow("keypoints", vis)
        out3.write(vis)
        cv2.imshow("heatmap", heatmap) 


        if match_img is not None:
            
            cv2.imshow("live_feed", match_img)
        if cv2.waitKey(1) == 27:
            break

        prev_depth = 0.5 * prev_depth + 0.5 * curr_depth
        prev_gray = curr_gray
        prev_color = curr_color
        pts_prev = pts_curr

    out3.release()
    cv2.imwrite("./KLT_classic.png", traj_vis)
    cv2.destroyAllWindows()


