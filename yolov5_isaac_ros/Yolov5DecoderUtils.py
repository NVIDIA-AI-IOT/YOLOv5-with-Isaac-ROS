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

from enum import IntEnum

import cv2
import numpy as np
import torch
from torch import nn


def tensor_to_torch_array(tensor):
    shape = tuple(tensor.shape.dims)
    x = None
    if tensor.data_type == 9:  # float32
        x = torch.frombuffer(bytearray(tensor.data), dtype=torch.float32)
    elif tensor.data_type == 10:  # float64
        x = torch.frombuffer(bytearray(tensor.data), dtype=torch.float64)
    else:
        print('Received tensor of incorrect type:', tensor.data_type)
        return None
    x = torch.reshape(x, shape)
    return x
