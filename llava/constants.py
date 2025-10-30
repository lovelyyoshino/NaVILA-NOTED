# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

"""
全局常量定义

定义了NaVILA模型中使用的各种常量，包括：
- 分布式训练相关的心跳参数
- 特殊token索引和字符串
- 图像/视频处理相关的标记
"""

# ==================== 分布式系统常量 ====================
# 控制器心跳过期时间（秒）- 用于分布式训练中检测节点是否存活
CONTROLLER_HEART_BEAT_EXPIRATION = 30

# 工作节点心跳间隔（秒）- 工作节点定期向控制器发送心跳的时间间隔
WORKER_HEART_BEAT_INTERVAL = 15

# 日志目录
LOGDIR = "."

# ==================== 模型核心常量 ====================
# 忽略索引 - 在计算损失时忽略的标签索引（通常用于padding或不需要计算梯度的位置）
IGNORE_INDEX = -100

# 图像token索引 - 在token序列中表示图像的特殊索引
IMAGE_TOKEN_INDEX = -200

# ==================== 特殊Token字符串 ====================
# 默认图像token - 在文本序列中表示图像位置的占位符
DEFAULT_IMAGE_TOKEN = "<image>"

# 图像patch token - 表示图像patch的特殊token
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

# 图像开始token - 标记图像内容的开始
DEFAULT_IM_START_TOKEN = "<im_start>"

# 图像结束token - 标记图像内容的结束
DEFAULT_IM_END_TOKEN = "<im_end>"

# 图像占位符 - 用于在预处理阶段标记图像位置
IMAGE_PLACEHOLDER = "<image-placeholder>"

# ==================== VILA特定常量 ====================
# 哨兵token - VILA架构中用于特殊标记的token
SENTINEL_TOKEN = "<vila/sentinel>"
