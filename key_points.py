import numpy as np
import cv2
import matplotlib.pyplot as plt
from dbscan_ransac import dbscan_ransac
from kcluster_ransac import kcluster_ransac


class keypoint_matcher: 

    def __init__(self, detector_make=cv2.ORB_create):

        self.detector = detector_make()
        

    def match_keypoints(self, img1, img2):
        """
        Expects BW images

        """

        kp1, desc1 = self.detector.detectAndCompute(img1, None)
        kp2, desc2 = self.detector.detectAndCompute(img2, None)

        matcher = cv2.BFMatcher()
        matches = matcher.match(desc1, desc2)
        return matches, kp1, kp2


# -----------------------------------------------------------

matcher = keypoint_matcher()
matches, kp1, kp2 = matcher.match_keypoints(img1_bw,img2_bw)

best_inlier_matches = kcluster_ransac(img1_bw, img2_bw, kp1, kp2, matches)


final_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    best_inlier_matches, None
)

final_img = cv2.resize(final_img, (1000, 650))

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title(f"Best Cluster (Kmeans + RANSAC)")
plt.axis('off')
plt.show()