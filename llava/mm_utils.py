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

"""
多模态工具函数模块

提供图像和视频处理的各种工具函数，包括：
- 视频帧采样（用于导航任务和通用视频处理）
- 图像加载和转换
- Token化和Tensor处理
- 模型输入预处理
- 停止条件判断
"""

import base64
import os
import tempfile
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import StoppingCriteria

from llava.constants import IMAGE_TOKEN_INDEX


def vlnce_frame_sampling(frames, num_frames=8):
    """
    VLN-CE（视觉-语言导航）任务的帧采样函数
    
    从视频帧序列中采样固定数量的帧，确保包含最新的一帧。
    采样策略：均匀采样 + 保留最后一帧
    
    参数:
        frames (List): 帧列表，可以是PIL Image对象或图像路径
        num_frames (int): 需要采样的帧数，默认8帧
    
    返回:
        List[Image]: 采样后的PIL Image列表（RGB格式）
    
    采样逻辑:
        1. 如果没有帧：返回num_frames个黑色图像
        2. 如果帧数不足：在开头填充黑色图像
        3. 正常情况：均匀采样(num_frames-1)帧 + 最后一帧
    
    注意:
        - 最后一帧始终被保留（对导航任务很重要）
        - 采样使用线性插值，保证时间均匀性
    """

    # 没有帧的情况：返回黑色图像
    if len(frames) == 0:
        print("No frames found. Returning empty images.")
        return [Image.new("RGB", (448, 448), (0, 0, 0))] * num_frames

    # 帧数不足的情况：在开头填充黑色图像
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (448, 448), (0, 0, 0)))

    # 保留最后一帧（导航任务中当前观测很重要）
    latest_frame = frames[-1]
    
    # 从前面的帧中均匀采样 (num_frames-1) 个
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    
    # 转换为PIL Image对象
    images = []
    for frame in sampled_frames:
        # 如果是路径字符串，先加载图像
        if isinstance(frame, str):
            frame = Image.open(frame)
        assert isinstance(frame, Image.Image)
        images.append(frame.convert("RGB"))  # 确保是RGB格式
    
    return images


def get_frame_from_vcap_vlnce(vidcap, num_frames=10, fps=None, frame_count=None):
    """
    从OpenCV VideoCapture对象中提取帧（用于VLN-CE任务）
    
    参数:
        vidcap: cv2.VideoCapture对象
        num_frames (int): 需要提取的帧数，默认10
        fps: 帧率（当前版本会被重新读取，此参数未使用）
        frame_count: 总帧数（当前版本会被重新读取，此参数未使用）
    
    返回:
        List[Image]: 提取的PIL Image列表
    
    功能说明:
        1. 读取视频的FPS和总帧数
        2. 读取所有视频帧
        3. 调用vlnce_frame_sampling进行采样
        4. 将OpenCV格式(BGR)转换为PIL Image(RGB)
    """

    # 从VideoCapture对象获取视频属性
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 视频无效的情况：返回空白图像
    if fps == 0 or frame_count == 0:
        print("Video file not found. Returning empty images.")
        return [Image.new("RGB", (720, 720))] * num_frames

    images = []
    success = True
    frames = []

    # 读取所有视频帧
    while success:
        success, frame = vidcap.read()
        if success:
            frames.append(frame)

    # 没有成功读取任何帧
    if len(frames) == 0:
        print("No frames found. Returning empty images.")
        return [Image.new("RGB", (720, 720))] * num_frames

    # 帧数不足时在开头填充零帧
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, np.zeros_like(frames[0]))

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    # 将OpenCV格式(BGR)转换为PIL Image(RGB)
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in sampled_frames]
    return images, len(frames)


def get_frame_from_vcap(vidcap, num_frames=10, max_fps=0.0, fps=None, frame_count=None, video_file_name=None):
    """
    从VideoCapture对象中提取固定数量的帧（通用版本）
    
    参数:
        vidcap: cv2.VideoCapture对象
        num_frames (int): 需要提取的帧数，默认10
        max_fps (float): 最大FPS限制（当前版本未使用）
        fps (float): 视频的帧率，None时自动获取
        frame_count (int): 视频总帧数，None时自动获取
        video_file_name (str): 视频文件名（用于日志输出）
    
    返回:
        Tuple[List[Image], int]: (提取的PIL Image列表, 实际提取的帧数)
    
    采样策略:
        - 使用np.linspace在视频中均匀采样指定数量的帧
        - 如果视频帧数不足，会进行左填充（重复使用已有帧）
    
    注意:
        - 与get_frame_from_vcap_vlnce不同，这个函数不保留最后一帧
        - 适用于通用视频理解任务
    """
    import cv2

    # 获取或验证视频属性
    if fps == None or frame_count == None:
        # if one of fps or frame_count is None, still recompute
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 视频无效检查
    if fps == 0 or frame_count == 0:
        print(f"Video file not found. return empty images. {video_file_name}")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames, 0

    # 计算视频时长和采样间隔
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    
    # 视频太短，无法采样
    if frame_interval == 0 and frame_count <= 1:
        print(f"frame_interval is equal to 0. return empty image. {video_file_name}")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames, 0

    images = []
    count = 0
    success = True
    # 计算要采样的帧索引（均匀分布）
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    while success:
        # 情况1：视频帧数充足，按索引采样
        if frame_count >= num_frames:
            success, frame = vidcap.read()
            if count in frame_indices:
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                if len(images) >= num_frames:
                    return images, num_frames
            count += 1
        else:
            # 情况2：视频帧数不足，读取所有帧（左填充策略）
            success, frame = vidcap.read()
            if success:
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                count += 1
            else:
                break
    
    # 没有成功提取任何帧
    if len(images) == 0:
        raise ValueError("Did not find enough frames in the video. return empty image.")

    return images, len(images)


def get_frame_from_vcap_with_fps(vidcap, num_frames=10, max_fps=0.0, fps=None, frame_count=None, video_file_name=None):
    """
    从VideoCapture对象中提取帧（考虑FPS限制）
    
    根据视频时长和模型支持的最大FPS，智能调整提取的帧数。
    这是一个更高级的帧提取函数，适用于长视频处理。
    
    参数:
        vidcap: cv2.VideoCapture对象
        num_frames (int): 模型能支持的最大帧数（默认10）
        max_fps (float): 模型能支持的最大FPS（默认0.0，表示无限制）
        fps (float): 输入视频的帧率，None时自动获取
        frame_count (int): 输入视频的总帧数，None时自动获取
        video_file_name (str): 视频文件名（用于日志）
    
    返回:
        Tuple[List[Image], int]: (提取的PIL Image列表, 实际提取的帧数)
    
    采样策略:
        1. 如果视频时长 >= num_frames/max_fps（视频太长）:
           - 提取num_frames帧（达到模型最大帧数限制）
        
        2. 如果视频时长 < num_frames/max_fps（视频较短）:
           - 提取 duration * max_fps 帧（保持原始FPS）
           - 确保至少提取2帧
    
    特殊处理:
        - 视频无效或太短时，返回随机数量（2-8*max_fps）的空白图像
        - 使用vidcap.grab()跳过不需要的帧，提高效率
    
    示例:
        假设max_fps=1.0, num_frames=10:
        - 20秒视频 -> 提取10帧（每2秒1帧）
        - 5秒视频 -> 提取5帧（保持原始FPS）
    """

    import random
    import cv2

    # 获取或验证视频属性
    if fps == None or frame_count == None:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 视频无效检查
    if fps == 0 or frame_count == 0:
        print(f"Video file not found. return empty images. {video_file_name}")
        # 返回随机数量的空白帧（避免训练时batch size问题）
        empty_video_frames = int(random.uniform(2, 8 * max_fps))
        return [
            Image.new("RGB", (720, 720)),
        ] * empty_video_frames, 0

    # 计算视频时长（秒）
    duration = frame_count / fps
    
    # 策略1: 视频太长（超过模型支持的时长）
    # 降低采样FPS，提取固定数量的帧
    if duration >= num_frames / max_fps:
        frame_interval = frame_count // num_frames

        # 视频太短（只有1帧或更少）
        if frame_interval == 0 and frame_count <= 1:
            print(f"frame_interval is equal to 0. return empty image. {video_file_name}")
            empty_video_frames = int(random.uniform(2, 8 * max_fps))
            return [
                Image.new("RGB", (720, 720)),
            ] * empty_video_frames, 0

        images = []
        count = 0
        success = True
        # 计算均匀分布的帧索引
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

        while success:
            if frame_count >= num_frames:
                # 如果当前帧在采样索引中，读取并处理
                if count in frame_indices:
                    success, frame = vidcap.read()
                    try:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(img)
                        images.append(im_pil)
                    except:
                        continue
                    if len(images) >= num_frames:
                        return images, num_frames
                else:
                    # 跳过不需要的帧（grab比read更快）
                    success = vidcap.grab()
                count += 1
            else:
                # 视频帧数不足：读取所有可用帧
                success, frame = vidcap.read()
                if success:
                    try:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(img)
                        images.append(im_pil)
                    except:
                        continue
                    count += 1
                else:
                    break
    
    # 策略2: 视频较短（在模型支持范围内）
    # 按max_fps采样，保持较高的时间分辨率
    else:
        # 计算需要的帧数：duration * max_fps
        frames_required = int(duration * max_fps)
        frame_indices = np.linspace(0, frame_count - 1, frames_required, dtype=int)
        
        # 视频太短，无法提取足够的帧
        if frames_required == 0:
            print(f"frames_required is fewer than 2. Duration {duration}, return empty image.")
            empty_video_frames = int(random.uniform(2, 8 * max_fps))
            return [
                Image.new("RGB", (720, 720)),
            ] * empty_video_frames, 0
        # 至少需要2帧（避免单帧情况）
        elif frames_required == 1:
            frame_indices = np.linspace(0, frame_count - 1, 2, dtype=int)
        
        images = []
        count = 0
        looked = 0  # 已查看的帧数
        success = True

        while success:
            success, frame = vidcap.read()
            if success and (looked in frame_indices):
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except:
                    continue
                count += 1
            looked += 1

    # 最终检查：如果没有提取到任何帧
    if len(images) == 0:
        print("empty videos...")
        empty_video_frames = int(random.uniform(2, 8 * max_fps))
        return [
            Image.new("RGB", (720, 720)),
        ] * empty_video_frames, 0
    else:
        return images, len(images)


def opencv_extract_frames(vpath_or_bytesio, frames=6, max_fps=0.0, fps=None, frame_count=None):
    """
    使用OpenCV从视频中提取帧（统一接口）
    
    这是一个高级封装函数，可以处理多种输入格式的视频。
    根据max_fps参数自动选择合适的帧提取策略。

    参数:
        vpath_or_bytesio (str or BytesIO): 视频输入，可以是：
            - str: 视频文件路径
            - BytesIO: 包含视频数据的BytesIO对象
        frames (int): 要提取的帧数（默认6）
        max_fps (float): 最大FPS限制（默认0.0表示无限制）
            - 如果 > 0.0: 使用get_frame_from_vcap_with_fps（智能FPS调整）
            - 如果 = 0.0: 使用get_frame_from_vcap（简单均匀采样）
        fps (float): 视频帧率（可选，None时自动获取）
        frame_count (int): 视频总帧数（可选，None时自动获取）

    返回:
        Tuple[List[Image], int]: (提取的PIL Image列表, 实际提取的帧数)

    异常:
        NotImplementedError: 如果输入类型不被支持

    使用示例:
        # 从文件路径提取
        images, count = opencv_extract_frames("/path/to/video.mp4", frames=8)
        
        # 从BytesIO提取
        with open("video.mp4", "rb") as f:
            video_bytes = BytesIO(f.read())
        images, count = opencv_extract_frames(video_bytes, frames=8)
        
        # 使用FPS限制（适合长视频）
        images, count = opencv_extract_frames("/path/to/long_video.mp4", 
                                              frames=100, max_fps=1.0)
    """
    import cv2

    # 处理文件路径输入
    if isinstance(vpath_or_bytesio, str):
        vidcap = cv2.VideoCapture(vpath_or_bytesio)
        # 根据max_fps选择提取策略
        if max_fps > 0.0:
            return get_frame_from_vcap_with_fps(
                vidcap, frames, max_fps, fps=fps, frame_count=frame_count, video_file_name=vpath_or_bytesio
            )
        return get_frame_from_vcap(
            vidcap, frames, max_fps, fps=fps, frame_count=frame_count, video_file_name=vpath_or_bytesio
        )
    
    # 处理BytesIO输入
    elif isinstance(vpath_or_bytesio, (BytesIO,)):
        # 将BytesIO内容写入临时文件（OpenCV需要文件路径）
        # 假设是MP4格式
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(vpath_or_bytesio.read())
            temp_video_name = temp_video.name
            vidcap = cv2.VideoCapture(temp_video_name)
            # 根据max_fps选择提取策略
            if max_fps > 0.0:
                return get_frame_from_vcap_with_fps(
                    vidcap, frames, max_fps, fps=fps, frame_count=frame_count, video_file_name=temp_video_name
                )
            return get_frame_from_vcap(
                vidcap, frames, max_fps, fps=fps, frame_count=frame_count, video_file_name=temp_video_name
            )
    else:
        raise NotImplementedError(type(vpath_or_bytesio))


def load_image_from_base64(image):
    """
    从base64编码的字符串加载图像
    
    参数:
        image (str): base64编码的图像字符串
    
    返回:
        Image: PIL Image对象
    
    使用场景:
        - 从Web API接收图像数据
        - 从JSON格式的数据集加载图像
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """
    将PIL图像扩展为正方形（通过填充）
    
    在图像的短边添加padding，使其成为正方形。
    这对某些需要正方形输入的视觉编码器很有用。

    参数:
        pil_img (Image): 要扩展的PIL图像
        background_color (tuple or int): 填充颜色
            - RGB模式: (R, G, B) 元组
            - 灰度模式: 单个整数值

    返回:
        Image: 扩展后的正方形PIL图像

    处理逻辑:
        - 如果已经是正方形: 直接返回
        - 如果宽>高: 在上下添加填充
        - 如果高>宽: 在左右添加填充
    
    使用示例:
        # RGB图像
        img = Image.open("photo.jpg")  # 800x600
        square_img = expand2square(img, (122, 116, 104))  # 800x800
        
        # 灰度图像
        gray_img = Image.open("photo.jpg").convert("L")
        square_gray = expand2square(gray_img, 128)
    """
    width, height = pil_img.size
    
    # 灰度图像：background_color应该是单个值
    if pil_img.mode == "L":
        background_color = background_color[0]
    
    # 已经是正方形
    if width == height:
        return pil_img
    # 宽度大于高度：在上下添加填充
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    # 高度大于宽度：在左右添加填充
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_image(image_file, data_args, image_folder):
    """
    处理单个图像文件，准备模型输入
    
    根据data_args中的配置对图像进行预处理，包括：
    - 加载图像
    - 调整尺寸/宽高比
    - 归一化
    - 转换为tensor
    
    参数:
        image_file: 图像输入，可以是：
            - str: 图像文件名或路径
            - PIL.Image: 已加载的图像对象  
            - bytearray: 图像的字节数据
        data_args: 数据参数对象，包含：
            - image_processor: 图像处理器（CLIP/SigLIP等）
            - image_aspect_ratio: 宽高比处理方式
                * "resize": 直接resize到目标尺寸
                * "pad": 填充为正方形
                * 其他: 使用编码器的默认行为
        image_folder (str): 图像文件夹路径（如果image_file是相对路径）
    
    返回:
        torch.Tensor: 预处理后的图像tensor，形状为 [C, H, W]
    
    支持的视觉编码器:
        - CLIP: 默认中心裁剪，有crop_size属性
        - SigLIP: 默认resize，有size属性
        - InternViT: 默认resize
        - Radio: 默认中心裁剪
    """
    processor = data_args.image_processor
    
    # 加载图像
    if isinstance(image_file, str):
        # 字符串路径：可能是相对路径或绝对路径
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # 已经是图像对象（PIL Image或bytearray）
        image = image_file
    
    # 确保是RGB格式
    image = image.convert("RGB")
    
    # 宽高比处理方式1: 直接resize到目标尺寸
    if data_args.image_aspect_ratio == "resize":
        # 获取目标尺寸（不同编码器的属性名不同）
        if hasattr(data_args.image_processor, "crop_size"):
            # CLIP等编码器使用crop_size
            crop_size = data_args.image_processor.crop_size
        else:
            # SigLIP等编码器使用size
            assert hasattr(data_args.image_processor, "size")
            crop_size = data_args.image_processor.size
        # 直接resize，可能改变宽高比
        image = image.resize((crop_size["height"], crop_size["width"]))
    
    # 宽高比处理方式2: 填充为正方形（保持宽高比）
    if data_args.image_aspect_ratio == "pad":

        # 内部定义expand2square函数（局部作用域）
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        # 使用processor的均值作为填充颜色（通常接近背景色）
        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        # 应用processor预处理
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        # 宽高比处理方式3: 使用视觉编码器的默认行为
        # - CLIP: 中心裁剪
        # - Radio: 中心裁剪  
        # - SigLIP: resize
        # - InternViT: resize
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    
    return image


def process_images(images, image_processor, model_cfg):
    """
    批量处理多个图像
    
    参数:
        images (List): 图像列表，每个元素可以是文件路径或PIL Image
        image_processor: 图像处理器
        model_cfg: 模型配置对象
    
    返回:
        torch.Tensor or List[torch.Tensor]: 
            - 如果所有图像shape相同: 返回堆叠的tensor [B, C, H, W]
            - 否则: 返回tensor列表
    
    注意:
        - 这个函数会修改model_cfg.image_processor
        - 用于处理批量图像输入
    """
    model_cfg.image_processor = image_processor
    # 逐个处理图像
    new_images = [process_image(image, model_cfg, None) for image in images]

    # 如果所有图像尺寸相同，可以stack成batch
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None, lstrip=False):
    """
    将包含图像标记的提示文本转换为token IDs
    
    处理包含<image>标记的文本，将其转换为模型可以理解的token序列。
    <image>标记会被替换为特殊的image_token_index。
    
    参数:
        prompt (str): 包含<image>标记的提示文本
            例如: "描述这张图片<image>中的内容"
        tokenizer: HuggingFace tokenizer对象
        image_token_index (int): 图像token的索引值（默认-200）
        return_tensors (str): 返回格式
            - "pt": 返回PyTorch tensor
            - None: 返回Python列表
        lstrip (bool): 是否去除开头的特殊token
    
    返回:
        List[int] or torch.Tensor: Token ID序列
    
    处理逻辑:
        1. 按<image>分割文本
        2. 对每个文本块进行tokenize
        3. 在文本块之间插入image_token_index
        4. 处理BOS token（如果存在）
    
    使用示例:
        prompt = "用户: <image>这是什么？\n助手:"
        input_ids = tokenizer_image_token(
            prompt, tokenizer, 
            image_token_index=-200,
            return_tensors="pt"
        )
        # 结果: tensor([1, 1234, ..., -200, ..., 5678])
    """
    # 按<image>分割并tokenize每个文本块
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        """在列表元素之间插入分隔符"""
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    
    # 处理BOS token
    if lstrip:
        offset = 1
    else:
        # 如果第一个chunk包含BOS token，单独处理
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

    # 插入image token并拼接所有chunks
    for chunk_id, x in enumerate(insert_separator(prompt_chunks, [image_token_index] * (offset + 1))):
        if chunk_id == 0 and lstrip:
            input_ids.extend(x)
        else:
            input_ids.extend(x[offset:])

    # 转换为tensor（如果需要）
    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    
    return input_ids


def is_gemma_tokenizer(tokenizer):
    """
    检查是否为Gemma tokenizer
    
    参数:
        tokenizer: tokenizer对象
    
    返回:
        bool: 如果是Gemma tokenizer返回True
    
    用途:
        Gemma模型的tokenizer有特殊行为，需要特殊处理
    """
    return "gemma" in tokenizer.__class__.__name__.lower()


def get_model_name_from_path(model_path):
    """
    从模型路径中提取模型名称
    
    参数:
        model_path (str): 模型路径
            例如: "/path/to/llava-v1.5-7b"
            或: "/path/to/llava-v1.5-7b/checkpoint-1000"
    
    返回:
        str: 模型名称
            例如: "llava-v1.5-7b"
            或: "llava-v1.5-7b_checkpoint-1000"
    
    处理逻辑:
        - 如果路径最后是checkpoint-xxx，返回"父目录_checkpoint-xxx"
        - 否则返回路径的最后一部分
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    
    # 检查是否是checkpoint目录
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    基于关键词的停止条件类
    
    在模型生成过程中，当检测到指定的关键词时停止生成。
    这对于控制生成长度和避免无意义的重复很有用。
    
    属性:
        keywords (List[str]): 停止关键词列表
        keyword_ids (List[torch.Tensor]): 关键词的token ID序列
        max_keyword_len (int): 最长关键词的长度
        tokenizer: tokenizer对象
        start_len (int): 输入序列的初始长度
    
    使用示例:
        stopping_criteria = KeywordsStoppingCriteria(
            keywords=["</s>", "###"],
            tokenizer=tokenizer,
            input_ids=input_ids
        )
        outputs = model.generate(
            input_ids,
            stopping_criteria=[stopping_criteria]
        )
    """
    
    def __init__(self, keywords, tokenizer, input_ids):
        """
        初始化停止条件
        
        参数:
            keywords (List[str]): 停止关键词列表，如["</s>", "###", "\n\n"]
            tokenizer: tokenizer对象
            input_ids (torch.Tensor): 输入的token IDs，形状 [batch_size, seq_len]
        """
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        
        # 将每个关键词转换为token IDs
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            # 移除BOS token（如果存在）
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            # 记录最长关键词长度
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]  # 输入序列长度

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        检查单个序列是否应该停止
        
        参数:
            output_ids: 当前生成的token IDs，形状 [1, seq_len]
            scores: 生成分数
            **kwargs: 其他参数
        
        返回:
            bool: 如果检测到关键词返回True（停止生成）
        """
        # 只检查新生成的部分（不超过最长关键词长度）
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        
        # 将关键词IDs移到相同设备
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        
        # 方法1: 直接比较token IDs（精确匹配）
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        
        # 方法2: 解码后比较字符串（处理tokenization差异）
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        检查整个batch是否应该停止
        
        参数:
            output_ids: 生成的token IDs，形状 [batch_size, seq_len]
            scores: 生成分数
            **kwargs: 其他参数
        
        返回:
            bool: 如果batch中所有序列都应该停止返回True
        
        注意:
            只有当batch中所有序列都检测到关键词时才停止
        """
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        # 所有序列都满足停止条件才真正停止
        return all(outputs)
