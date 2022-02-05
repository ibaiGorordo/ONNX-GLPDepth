import cv2
import numpy as np

from glpdepth import GLPDepth

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize model
max_dist = 3.0
model_path='models/glpdepth_nyu_480x640.onnx'
depth_estimator = GLPDepth(model_path)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the video
	ret, frame = cap.read()
	if not ret:	
		break
	
	# Estimate depth and colorize it
	depth_map = depth_estimator(frame)
	color_depth = depth_estimator.draw_depth(max_dist)

	combined_img = np.hstack((frame, color_depth))
	
	cv2.imshow("Estimated depth", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		cv2.imwrite("out.jpg", combined_img)
		break

cap.release()
cv2.destroyAllWindows()

