# ONNX-GLPDepth
Python scripts for performing monocular depth estimation using the GLPDepth model in ONNX

![GLPDepth monocular depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-GLPDepth/blob/main/doc/img/out.jpg)
*Original image:https://commons.wikimedia.org/wiki/File:401_Gridlock.jpg*

# Requirements

 * Check the **requirements.txt** file. Additionally, **pafy** and **youtube-dl** are required for youtube video inference.

# Known issues
The KITTI model (for road scenes) has poor performance in the upper parts of the image due to the lack of ground truth lidar data in the dataset. The code tries to find where the depth starts to fail, and fills the depth with zeros that upper part.
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube_dl==2021.12.17
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/245_GLPDepth) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-GLPDepth/tree/main/models)** folder. 

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/vinvino02/GLPDepth).
 
# Examples

 * **Image inference**:
 
 ```
 python image_depth_estimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python video_depth_estimation.py
 ```
 
 * **Webcam inference**:
 
 ```
 python webcam_depth_estimation.py
 ```
 
# [Inference video Example](https://youtu.be/k5HZ_USROpU) 
 ![GLPDepth monocular depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-GLPDepth/blob/main/doc/img/glpdepth_test.gif)
*Original video: https://www.youtube.com/watch?v=jc3uSpXYZqY*

# References:
* GLPDepth model: https://github.com/vinvino02/GLPDepth
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* Original paper: https://arxiv.org/abs/2201.07436
 
