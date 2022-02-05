import cv2
import numpy as np
from imread_from_url import imread_from_url

from glpdepth import GLPDepth

# Initialize model
max_dist = 5.0
model_path='models/glpdepth_kitti_480x640.onnx'
depth_estimator = GLPDepth(model_path)

# Read inference image
img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/401_Gridlock.jpg/1280px-401_Gridlock.jpg")

# Estimate depth and colorize it
depth_map = depth_estimator(img)
color_depth = depth_estimator.draw_depth(max_dist)

combined_img = np.hstack((img, color_depth))

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
cv2.imshow("Estimated depth", combined_img)
cv2.waitKey(0)