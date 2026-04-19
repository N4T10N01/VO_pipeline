import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def dbscan_ransac(pts1, pts2):

    # motion version
    vectors = []

    for i in range(len(pts1)):
        (x1, y1) = pts1[i]
        (x2, y2) = pts2[i]
        
        dx = x2 - x1
        dy = y2 - y1
        
        v = [dx, dy]
        vectors.append(v)

    X = np.array(vectors)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    db = DBSCAN(eps=0.3, min_samples=6).fit(X)
    labels = db.labels_

    best_inlier_count = 0
    best_H = None
    best_cluster = None
    best_idxs = None 

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # noise

        idxs = np.where(labels == cluster_id)[0]

         #need at least 8 points for H
        if len(idxs) < 8:
            continue
    
        H, mask = cv2.findHomography(pts1[idxs], pts2[idxs], cv2.RANSAC, 5.0)
        if H is None:
            continue

        inliers = mask.ravel().tolist()
        
        inlier_count = sum(inliers)
        
        # Track best cluster/least affected by filtering

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = mask.ravel().astype(bool)
            best_H = H
            best_idxs = idxs

    if best_H is None:
        print("No valid homography found")
        return None

    else: 
        return best_idxs[best_inliers]
