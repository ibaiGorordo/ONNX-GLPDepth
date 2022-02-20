import sys
import cv2
import time
import numpy as np
import onnx
import onnxruntime

class GLPDepth():

	def __init__(self, model_path):

		self.first_zero_row = 0

		# Initialize model
		self.initialize_model(model_path)

	def __call__(self, image):

		return self.estimate_depth(image)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path)

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_depth(self, image):

		input_tensor = self.prepare_input(image)

		outputs = self.inference(input_tensor)
		
		self.depth_map = self.process_output(outputs)

		return self.depth_map

	def prepare_input(self, img):

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		self.img_height, self.img_width = img.shape[:2]

		img_input = cv2.resize(img, (self.input_width,self.input_height))

		img_input = img_input/255
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	def inference(self, input_tensor):
		# start = time.time()
		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

		# print(time.time() - start)
		return outputs

	def process_output(self, output): 

		return np.squeeze(output)

	def draw_depth(self, max_depth = 5.0):

		# Normalize estimated depth to color it
		depth_min = 0.0 #self.depth_map.min()
		norm_depth_map = 255 * (1.0-(self.depth_map-depth_min)/(max_depth-depth_min)) 
		norm_depth_map[norm_depth_map < 0] =0

		# Issue: KITTI model fails to inference properly the upper part of the image (KITTI is missing Lidar data in that region)
		# Find where there is a high number of zeros (background), and fill the upper part
		zero_count_thres = int(self.input_width*0.3)
		row_zero_count = np.count_nonzero(norm_depth_map==0, axis=1)
		zero_rows = np.argwhere(row_zero_count>zero_count_thres)
		if(zero_rows.shape[0] > 0):
			self.first_zero_row = self.first_zero_row//2 + zero_rows[0][0]//2 # Smooth cut region
		
		norm_depth_map[:self.first_zero_row,:] = 0

		# Normalize and color the image
		color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

		# Resize the depth map to match the input image shape
		return cv2.resize(color_depth, (self.img_width,self.img_height))

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape
		self.output_height = self.output_shape[2]
		self.output_width = self.output_shape[3]

if __name__ == '__main__':
	
	from imread_from_url import imread_from_url

	# Initialize model
	model_path='../models/glpdepth_kitti_480x640.onnx'
	depth_estimator = GLPDepth(model_path)

	# Read inference image
	img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/401_Gridlock.jpg/1280px-401_Gridlock.jpg")

	# Estimate depth and colorize it
	depth_map = depth_estimator(img)
	color_depth = depth_estimator.draw_depth()

	combined_img = np.hstack((img, color_depth))

	cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
	cv2.imshow("Estimated depth", combined_img)
	cv2.waitKey(0)
