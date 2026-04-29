import numpy as np
import cv2
import pyrealsense2 as rs
import gtsam
from gtsam import symbol

from dbscan_ransac import dbscan_ransac
from classic_system import estimate_motion
from lie_algebra_utilities import *

BA_MAX_FRAMES = 60
BA_MIN_LANDMARKS = 60

USE_MOTION_PRIORS = True
DEFAULT_MEASUREMENT_SIGMA_PX = 1.0


# ============================================================
# GTSAM BA FUNCTIONS
# ============================================================

def np_to_pose3(T):
    T = np.asarray(T, dtype=float)
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return gtsam.Pose3(R, t)


def run_batch_bundle_adjustment(
    window_data,
    default_sigma_px=1.0,
    huber_k=1.345,
    max_iterations=50,
    use_motion_priors=True,
):


    K = np.asarray(window_data["K"], dtype=float)
    poses_init = window_data["poses_init"]
    landmarks_init = window_data["landmarks_init"]
    observations = window_data["observations"]

    measurement_sigmas = window_data.get("measurement_sigmas", [])
    relative_motions = window_data.get("relative_motions", [])
    motion_covariances = window_data.get("motion_covariances", [])

    if len(poses_init) == 0:
        raise ValueError("poses_init is empty")
    if len(observations) == 0:
        raise ValueError("observations is empty")
    if len(poses_init) != len(observations)+1:
        raise ValueError(
            f"poses_init and observations must have same length, got "
            f"{len(poses_init)} and {len(observations)}"
        )

    if measurement_sigmas and len(measurement_sigmas) != len(observations):
        raise ValueError("measurement_sigmas must have one entry per observation frame")

    num_landmarks = len(landmarks_init)
    if num_landmarks == 0:
        raise ValueError("landmarks_init is empty")

    for t, obs in enumerate(observations):
        kps = np.asarray(obs["keypoints"])
        if kps.ndim != 2 or kps.shape[1] != 2:
            raise ValueError(f'observations[{t}]["keypoints"] must be shape (N,2)')
        if kps.shape[0] != num_landmarks:
            raise ValueError(
                f'observations[{t}] has {kps.shape[0]} keypoints but landmarks_init has {num_landmarks}'
            )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    calib = gtsam.Cal3_S2(float(fx), float(fy), 0.0, float(cx), float(cy))

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    pose_key = lambda t: symbol('x', int(t))
    lm_key = lambda j: symbol('p', int(j))

    for t, T in enumerate(poses_init):
        initial.insert(pose_key(t), np_to_pose3(T))

    for j in range(num_landmarks):
        pt = np.asarray(landmarks_init[j], dtype=float).reshape(3)
        initial.insert(lm_key(j), gtsam.Point3(float(pt[0]), float(pt[1]), float(pt[2])))

    # Anchor first pose
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype=float)
    )
    graph.add(
        gtsam.PriorFactorPose3(
            pose_key(0),
            np_to_pose3(poses_init[0]),
            prior_noise
        )
    )

    # Reprojection factors
    for t, obs in enumerate(observations):
        sigma_px = float(measurement_sigmas[t]) if measurement_sigmas else float(default_sigma_px)

        base_meas = gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
        robust_meas = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(float(huber_k)),
            base_meas
        )

        kps = np.asarray(obs["keypoints"], dtype=float)
        for j in range(num_landmarks):
            u, v = kps[j]
            z = gtsam.Point2(float(u), float(v))
            graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    z,
                    robust_meas,
                    pose_key(t+1),
                    lm_key(j),
                    calib
                )
            )

    # Between factors from relative motion measurements
    if use_motion_priors:
        if len(relative_motions) != len(observations):
            raise ValueError("relative_motions length must equal len(observations)")
        if len(motion_covariances) != len(observations):
            raise ValueError("motion_covariances length must equal len(observations)")

        for t in range(len(relative_motions)):
            dT = np.asarray(relative_motions[t], dtype=float)
            Sigma = np.asarray(motion_covariances[t], dtype=float)

            if Sigma.shape != (6, 6):
                raise ValueError(f"motion_covariances[{t}] must be 6x6")

            Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * np.eye(6)
            motion_noise = gtsam.noiseModel.Gaussian.Covariance(Sigma)

            graph.add(
                gtsam.BetweenFactorPose3(
                    pose_key(t),
                    pose_key(t + 1),
                    np_to_pose3(dT),
                    motion_noise
                )
            )

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iterations))
    params.setVerbosityLM("SILENT")
    params.setVerbosity("SILENT")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    optimized_poses = []
    for t in range(len(poses_init)):
        pose = result.atPose3(pose_key(t))
        optimized_poses.append(pose.matrix())

    optimized_landmarks = {}
    for j in range(num_landmarks):
        pt = np.asarray(result.atPoint3(lm_key(j)), dtype=float).reshape(3)
        optimized_landmarks[j] = pt.copy()

    return {
        "poses": optimized_poses,
        "landmarks": optimized_landmarks,
        "initial_error": graph.error(initial),
        "final_error": graph.error(result),
        "result": result,
        "graph": graph,
        "initial": initial,
    }


def maybe_run_ba_and_reset(
    current_pose,
    K,
    landmarks_3d,
    observations,
    poses_window,
    measurement_sigmas,
    frame_since_ba,
    pts_prev,
    relative_motions,
    motion_covariances,
    ba_min_landmarks=60,
    ba_max_frames=60,
    use_motion_priors=True,
):
    should_run = (
        landmarks_3d is not None and
        (len(landmarks_3d) < ba_min_landmarks or frame_since_ba >= ba_max_frames)
    )

    if not should_run:
        return (
            False,
            current_pose,
            landmarks_3d,
            observations,
            poses_window,
            measurement_sigmas,
            frame_since_ba,
            pts_prev,
            relative_motions,
            motion_covariances,
        )

    window_data = {
        "K": K,
        "poses_init": poses_window,
        "landmarks_init": {i: landmarks_3d[i] for i in range(len(landmarks_3d))},
        "observations": observations,
        "measurement_sigmas": measurement_sigmas,
        "relative_motions": relative_motions,
        "motion_covariances": motion_covariances,
    }

    ba_out = run_batch_bundle_adjustment(
        window_data,
        default_sigma_px=DEFAULT_MEASUREMENT_SIGMA_PX,
        huber_k=1.345,
        max_iterations=5,
        use_motion_priors=use_motion_priors,
    )

    current_pose = ba_out["poses"][-1]

    print(
        f"[BA] frames={frame_since_ba}, landmarks={len(landmarks_3d)}, "
        f"initial_error={ba_out['initial_error']:.4f}, final_error={ba_out['final_error']:.4f}"
    )

    landmarks_3d = None
    observations = []
    poses_window = [current_pose.copy()]
    measurement_sigmas = []
    frame_since_ba = 0
    pts_prev = None
    relative_motions = []
    motion_covariances = []

    return (
        True,
        current_pose,
        landmarks_3d,
        observations,
        poses_window,
        measurement_sigmas,
        frame_since_ba,
        pts_prev,
        relative_motions,
        motion_covariances,
    )


# ============================================================
# VO HELPERS
# ============================================================

def draw_keypoints(img, pts, color=(0, 255, 0)):
    for p in pts:
        x, y = int(p[0]), int(p[1])
        cv2.circle(img, (x, y), 2, color, -1)
    return img


def draw_trajectory(traj_img, t, prev_point, scale=50):
    center_x = traj_img.shape[1] // 2
    center_y = traj_img.shape[0] // 2

    x = int(scale * t[0] + center_x)
    y = int(scale * t[1] + center_y)

    if prev_point is not None:
        cv2.line(traj_img, prev_point, (x, y), (0, 0, 255), 2)

    return traj_img, (x, y)


def bilinear_depth_sample(depth, u, v):
    h, w = depth.shape

    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    valid = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)

    z = np.zeros_like(u, dtype=np.float32)
    if not np.any(valid):
        return z, valid

    u_valid = u[valid]
    v_valid = v[valid]

    x0 = np.floor(u_valid).astype(np.int32)
    y0 = np.floor(v_valid).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    du = u_valid - x0
    dv = v_valid - y0

    z00 = depth[y0, x0]
    z01 = depth[y0, x1]
    z10 = depth[y1, x0]
    z11 = depth[y1, x1]

    z_valid = (
        (1 - du) * (1 - dv) * z00 +
        du * (1 - dv) * z01 +
        (1 - du) * dv * z10 +
        du * dv * z11
    )

    z[valid] = z_valid
    return z, valid


def project_to_3D(pts, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = pts[:, 0]
    v = pts[:, 1]

    z, valid_interp = bilinear_depth_sample(depth, u, v)

    valid_depth = z > 1e-4
    valid = valid_interp & valid_depth

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

    pts_3d = np.stack([X, Y, Z], axis=1)
    return pts_3d, valid


def complete_depth_filter(depth_frame, depth_scale, spatial, temporal, hole_filling):
    depth_frame = temporal.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    return depth


# ============================================================
# PIPELINE / CAMERA SETUP
# ============================================================

out3 = cv2.VideoWriter(
    "classic_system_tracking_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (640, 480)
)

out2 = cv2.VideoWriter(
    "classic_system_tracking_trajectory.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (600, 600)
)

prev_point = None
traj_img = np.ones((600, 600, 3), dtype=np.uint8) * 255

#================== BA state ==================#
landmarks_3d = None
observations = []
poses_window = []

measurement_sigmas = []
relative_motions = []
motion_covariances = []

frame_since_ba = 0
#================= BA state ==================#

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

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

K = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])


# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":
    current_pose = np.eye(4)
    perturbation = np.eye(4)
    poses_window = [current_pose.copy()]
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

    loop_count = 0
    while True:
        loop_count += 1
        match_img = None

        if pts_prev is None or len(pts_prev) < BA_MIN_LANDMARKS:
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
        curr_color = np.asanyarray(curr_frame.get_color_frame().get_data())
        curr_depth = curr_frame.get_depth_frame()

        curr_depth = complete_depth_filter(curr_depth, depth_scale, spatial, temporal, hole_filling)

        depth = np.nan_to_num(curr_depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)

        mask_1 = status.flatten() == 1
        pts_prev = pts_prev[mask_1].reshape(-1, 2)
        pts_curr = pts_curr[mask_1].reshape(-1, 2)

        points_3d_full, mask_2 = project_to_3D(pts_prev, prev_depth, K)

        pts_prev_valid = pts_prev[mask_2]
        pts_curr_valid = pts_curr[mask_2]

        points_3d_track = points_3d_full.copy()
        pts_curr_track = pts_curr.copy()
  

        #================== only mask_1 and mask_2 affect BA state ==================#
        if landmarks_3d is not None:
            combined_mask = mask_1.copy()
            # combined_mask[combined_mask] = mask_2
            landmarks_3d = landmarks_3d[combined_mask]

            for obs in observations:
                obs["keypoints"] = obs["keypoints"][combined_mask]


        if len(pts_prev_valid) > 20:

            T_prev = current_pose.copy()

            points_3d_solve = points_3d_full[mask_2].copy()
            pts_curr_solve = pts_curr_valid.copy()

            if len(pts_prev_valid) > 80:
                mask_3 = dbscan_ransac(pts_prev_valid, pts_curr_valid)

                pts_curr_solve = pts_curr_solve[mask_3]
                points_3d_solve = points_3d_solve[mask_3]

                pts_curr_solve = np.asarray(pts_curr_solve).reshape(-1, 2)
                points_3d_solve = np.asarray(points_3d_solve).reshape(-1, 3)

            perturbation_world, sigma = estimate_motion(
                pts_curr_solve,
                points_3d_solve,
                K,
                current_pose
            )

            T_next = perturbation_world @ T_prev

            Z_between = np.linalg.inv(T_prev) @ T_next

            if landmarks_3d is None:
                landmarks_3d = points_3d_track.copy()

            observations.append({
                "keypoints": pts_curr_track.copy()
            })

            poses_window.append(T_next.copy())

            measurement_sigma_this_frame = DEFAULT_MEASUREMENT_SIGMA_PX
            measurement_sigmas.append(float(measurement_sigma_this_frame))

            relative_motions.append(Z_between.copy())

            if np.ndim(sigma) == 2 and np.shape(sigma) == (6, 6):
                motion_covariances.append(sigma.copy())
            else:
                raise ValueError("estimate_motion must return a 6x6 pose covariance")

            frame_since_ba += 1


            assert all(
                obs["keypoints"].shape[0] == landmarks_3d.shape[0]
                for obs in observations
            )

            (
                did_run_ba,
                current_pose,
                landmarks_3d,
                observations,
                poses_window,
                measurement_sigmas,
                frame_since_ba,
                pts_prev,
                relative_motions,
                motion_covariances,
            ) = maybe_run_ba_and_reset(
                current_pose=T_next,
                K=K,
                landmarks_3d=landmarks_3d,
                observations=observations,
                poses_window=poses_window,
                measurement_sigmas=measurement_sigmas,
                frame_since_ba=frame_since_ba,
                pts_prev=pts_prev,
                relative_motions=relative_motions,
                motion_covariances=motion_covariances,
                ba_min_landmarks=BA_MIN_LANDMARKS,
                ba_max_frames=BA_MAX_FRAMES,
                use_motion_priors=USE_MOTION_PRIORS,
            )

            if not(did_run_ba):
                current_pose = T_next

        traj_img, prev_point = draw_trajectory(
            traj_img.copy(),
            current_pose[:3, 3],
            prev_point
        )
        traj_vis = traj_img

        cv2.imshow("Trajectory", traj_vis)
        out2.write(traj_vis)

        vis = curr_color.copy()
        vis = draw_keypoints(vis, pts_curr)
        cv2.imshow("keypoints", vis)
        out3.write(vis)
        cv2.imshow("heatmap", heatmap)

        if match_img is not None:
            cv2.imshow("live_feed", match_img)

        if cv2.waitKey(1) == 27:
            break

        prev_gray = curr_gray
        prev_color = curr_color
        prev_depth = curr_depth
        pts_prev = pts_curr

    out2.release()
    out3.release()
    cv2.imwrite("./KLT_classic.png", traj_vis)
    cv2.destroyAllWindows()