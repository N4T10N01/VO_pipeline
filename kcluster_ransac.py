
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kcluster_ransac(kp1, kp2, matches):
    vectors = []
    match_map = []

    # h1, w1 = img1.shape
    # h2, w2 = img2.shape

    # for m in matches:
    #     (x1, y1) = kp1[m.queryIdx].pt
    #     (x2, y2) = kp2[m.trainIdx].pt
        
    #     v = [x1/w1, y1/h1, x2/w2, y2/h2]  # normalized
    #     vectors.append(v)
    #     match_map.append(m)

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

    k = 6 #tune as fit
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_

    best_inliers = []
    best_H = None
    best_cluster = None

    for i in range(k):
        idxs = np.where(labels == i)[0]

         #need at least 8 points for H
        if len(idxs) < 8:
            continue


        src_pts = np.float32([kp1[match_map[j].queryIdx].pt for j in idxs]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[match_map[j].trainIdx].pt for j in idxs]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if H is None:
            continue

        inliers = mask.ravel().tolist()
        inlier_count = sum(inliers)

        if inlier_count > len(best_inliers):
            best_inliers = inliers
            best_H = H
            best_cluster = i

    if best_H is None:
        print("No valid homography found.")
        return None
    
    else:
    
        idxs = np.where(labels == best_cluster)[0]
        cluster_matches = [match_map[j] for j in idxs]
        final_matches = [m for m, keep in zip(cluster_matches, best_inliers) if keep]
        return final_matches



# final_img = cv2.drawMatches(
#     query_img, kp1,
#     train_img, kp2,
#     final_matches, None,
#     matchColor=(0,255,0),
#     singlePointColor=(0,0,255),
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )


# final_img = cv2.resize(final_img, (1000, 650))

# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
# plt.title("Filtered Matches (KMeans + RANSAC)")
# plt.axis('off')
# plt.show()
