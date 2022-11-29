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

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab-master.nvidia.com/asawareeb/yolov5-isaac-ros-dnn.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab-master.nvidia.com/asawareeb/yolov5-isaac-ros-dnn/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
