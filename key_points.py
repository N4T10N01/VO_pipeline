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