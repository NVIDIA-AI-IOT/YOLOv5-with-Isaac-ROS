# YOLOv5 Isaac ROS DNN

Integrating YOLOv5 with Isaac ROS DNN Inference.

<figure>
   <img src="/images/workflow_with_camera.PNG" height="75%" width="75%">
   <figcaption>Workflow</figcaption>
</figure>

## Usage

Based on [isaac_ros_dnn_inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference).
Tested with ROS2 Humble running on Jetson Orin with JetPack 5.0.2. 

Refer to the license terms for the YOLOv5 project before using this software and ensure you are using YOLOv5 under license terms compatible with your project requirements.

## Docker
Use the [Isaac ROS Dev Docker](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference#docker) for development. 

## Model preparation
- Download [yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt) from the YOLOv5 project.
- Export to ONNX following steps [here](https://github.com/ultralytics/yolov5/issues/251) and visualize the ONNX model using [Netron](https://netron.app/). Note `input` and `output` names - these will be used to run the node. 
- Copy the ONNX model to a location accessible from the container.

## Running the YOLOv5 node
- Copy [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) from the official YOLOv5 project to the container.
- Enter the container and run the following: 
```
pip install -r requirements.txt
pip install python-dateutil==2.8.1
pip install torch==1.12.1
pip install torchvision==0.13.1
```
- Download the [utils](https://github.com/ultralytics/yolov5/tree/master/utils) folder from the Ultralytics YOLOv5 project and put it in the `yolov5_isaac_ros` folder. File structure should look something like this:
```
.
+- yolov5-isaac-ros-dnn
   +- README
   +- launch
   +- images
   +- yolov5_isaac_ros
      +- utils
      +- Yolov5Decoder.py  
      +- Yolov5DecoderUtils.py    
```
Refer to the license terms for the YOLOv5 project before using this software and ensure you are using YOLOv5 under license terms compatible with your project requirements.
- Build, source and run the `yolov5_isaac_ros` node. This subscribes to input images from the Realsense camera on topic `/camera/color/image_raw`. It performs inference and publishes results on topic `/object_detections` as [Detection2DArray](http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/Detection2DArray.html) messages. Use the names noted above in `Model preparation` as `input_binding_names` and `output_binding_names`:
```
colcon build --packages-up-to yolov5_isaac_ros
source install/setup.bash

ros2 launch isaac_ros_tensor_rt isaac_ros_tensor_rt.launch.py model_file_path:=/workspaces/isaac_ros-dev/src/yolov5s.onnx engine_file_path:=/workspaces/isaac_ros-dev/src/yolov5s.plan input_binding_names:=['images'] output_binding_names:=['output0']
```
- For subsequent runs, use the following command as the engine file is saved after the first run:
```
ros2 launch yolov5_isaac_ros isaac_ros_yolov5_tensor_rt.launch.py network_image_width:=640 network_image_height:=640 engine_file_path:=/workspaces/isaac_ros-dev/src/yolov5s.plan input_binding_names:=['images'] output_binding_names:=['output0']
```

## Using Triton
<figure>
   <img src="/images/triton_workflow.PNG" height="75%" width="75%">
   <figcaption>Workflow</figcaption>
</figure>

- Convert the ONNX model to a TRT plan file (named `model.plan`) using `trtexec`. To do this, run the following command from `/usr/src/tensorrt/bin` and save the generated file under `yolov5/1/`. 
```
./trtexec --onnx=yolov5s.onnx --saveEngine=model.plan  --fp16
```
- File structure should look like this:
```
.
+- yolov5_isaac_ros
   +- config
   +- yolov5
      +- config.pbtxt
      +- 1
         +- model.plan
   +- launch
      +- isaac_ros_yolov5_triton.launch.py
      
```
- To launch the pipeline using Triton for inference: 
```
ros2 launch yolov5_isaac_ros isaac_ros_yolov5_triton.launch.py network_image_width:=640 network_image_height:=640
```

## Output visualization
The `yolov5_visualizer_node` subscribes to the topics below and publishes images with resulting bounding boxes on topic `yolov5_processed_image`:
- `camera/color/image_raw`
- `object_detections`

On running the pipeline, an rqt window will pop up showing bounding boxes and labels around detected objects.

## Launching Realsense camera for input
- Follow [Isaac ROS Realsense Setup](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md) to setup the camera.
- Follow steps [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline#quickstart) to launch the camera node (skip step 6).
- Verify that images are being published on `/camera/color/image_raw`, can echo or use rqt for this.

## Support
Please reach out regarding issues and suggestions here.
