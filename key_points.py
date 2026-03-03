import numpy as np
import cv2
import matplotlib.pyplot as plt

query_img = cv2.imread('test_imgs\handheld_check1.jpg')
train_img = cv2.imread('test_imgs\handheld_check2.jpg')

query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

img1 = query_img_bw
img2 = train_img_bw

detector = cv2.ORB_create()

queryKeypoints, queryDescriptors = detector.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = detector.detectAndCompute(train_img_bw, None)

kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)

# final_img = cv2.drawMatches(query_img, queryKeypoints, 
#                              train_img, trainKeypoints, matches[:20], None)
# final_img = cv2.resize(final_img, (1000, 650))


# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)) 
# plt.title("Feature Matches")
# plt.axis('off')  
# plt.show()

vectors = []
match_map = []

h1, w1 = img1.shape
h2, w2 = img2.shape

for m in matches:
    (x1, y1) = kp1[m.queryIdx].pt
    (x2, y2) = kp2[m.trainIdx].pt
    
    v = [x1/w1, y1/h1, x2/w2, y2/h2]  # normalized
    vectors.append(v)
    match_map.append(m)

X = np.array(vectors)

from sklearn.cluster import KMeans

k = 5  # tune this
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
labels = kmeans.labels_

best_inliers = []
best_H = None

for i in range(k):
    idxs = np.where(labels == i)[0]
    if len(idxs) < 8:
        continue

    cluster_matches = [match_map[j] for j in idxs]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in cluster_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in cluster_matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        continue

    inliers = mask.ravel().tolist()
    inlier_count = sum(inliers)

    if inlier_count > len(best_inliers):
        best_inliers = inliers
        best_H = H
        best_cluster = i


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

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=6).fit(X)
labels = db.labels_

for cluster_id in set(labels):
    if cluster_id == -1:
        continue  # noise

    idxs = np.where(labels == cluster_id)[0]
    if len(idxs) < 8:
        continue

    # build src_pts, dst_pts
    src_pts = np.float32([kp1[match_map[j].queryIdx].pt for j in idxs]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[match_map[j].trainIdx].pt for j in idxs]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        continue

    inliers = mask.ravel().tolist()
    inlier_count = sum(inliers)

    print(f"Cluster {cluster_id}: {inlier_count} inliers")