import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def dbscan_ransac(kp1, kp2, matches):
    # motion version
    vectors = []
    match_map = []

    for m in matches:
        (x1, y1) = kp1[m.queryIdx].pt
        (x2, y2) = kp2[m.trainIdx].pt
        
        dx = x2 - x1
        dy = y2 - y1
        
        v = [dx, dy]
        vectors.append(v)
        match_map.append(m)

    X = np.array(vectors)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    db = DBSCAN(eps=0.3, min_samples=6).fit(X)
    labels = db.labels_

    best_inliers = []
    best_H = None
    best_cluster = None

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # noise

        idxs = np.where(labels == cluster_id)[0]

         #need at least 8 points for H
        if len(idxs) < 8:
            continue
        
        src_pts = np.float32([kp1[match_map[j].queryIdx].pt for j in idxs]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[match_map[j].trainIdx].pt for j in idxs]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            continue

        inliers = mask.ravel().tolist()
        inlier_count = sum(inliers)

        # Track best cluster/least affected by filtering
        if inlier_count > len(best_inliers):
            best_inliers = inliers 
            best_H = H
            best_cluster = cluster_id

    if best_H is None:
        print("No valid homography found")
        return None

    else: 
        idxs = np.where(labels == best_cluster)[0]
        cluster_matches = [match_map[j] for j in idxs]
        final_matches = [m for m, keep in zip(cluster_matches, best_inliers) if keep]
        return final_matches 

