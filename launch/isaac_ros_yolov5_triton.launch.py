# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """Launch the DNN Image encoder, Triton node and Yolov5 decoder node."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    config = launch_dir_path + '/../config/params.yaml'
    model_dir_path = '/workspaces/isaac_ros-dev/src/yolov5_isaac_ros'

    launch_args = [
        DeclareLaunchArgument(
            'network_image_width',
            default_value='960',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='544',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'encoder_image_mean',
            default_value='[0.5, 0.5, 0.5]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'encoder_image_stddev',
            default_value='[0.5, 0.5, 0.5]',
            description='The standard deviation for image normalization'),
    ]

    # DNN Image Encoder parameters
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    encoder_image_mean = LaunchConfiguration('encoder_image_mean')
    encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

    
    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': encoder_image_mean,
            'image_stddev': encoder_image_stddev,
        }],
        remappings=[('encoded_tensor', 'tensor_pub'), ('image', '/camera/color/image_raw')]
    )

    triton_node = ComposableNode(
        name='triton_node',
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'yolov5',
            'model_repository_paths': [model_dir_path],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['images'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['output_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'log_level': 0
        }])
     


    resize_node = ComposableNode(
        name='isaac_ros_resize',
        package='isaac_ros_image_proc',
        plugin='isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'use_relative_scale': False,
            'height': 640,
            'width': 640,
        }]
    )

    
    rclcpp_container = ComposableNodeContainer(
        name='yolov5_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            encoder_node, triton_node],
        output='screen',
    )


    yolov5_decoder_node = Node(
        name='yolov5_decoder_node',
        package='yolov5_isaac_ros',
        executable='Yolov5Decoder',
        output='screen',
    )


    yolov5_visualizer_node = Node(
            name='yolov5_visualizer_node',
            package='yolov5_isaac_ros',
            executable='isaac_ros_yolov5_visualizer',
            output='screen',
    )
    
    rqt_node = Node(
            name='image_view',
            package='rqt_image_view',
            executable='rqt_image_view',
            arguments=['/yolov5_processed_image']
    )


    final_launch_description = launch_args + [rclcpp_container] + [yolov5_decoder_node] + [yolov5_visualizer_node] + [rqt_node]
    return launch.LaunchDescription(final_launch_description)

