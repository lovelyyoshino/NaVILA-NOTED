"""
模型加载入口模块

提供简化的模型加载接口，是使用NaVILA模型的主要入口点。
"""

import os
import typing
from typing import List, Optional

# 类型检查时导入PreTrainedModel类型，运行时设为None以避免不必要的依赖
if typing.TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = None

__all__ = ["load"]


def load(
    model_path: str,
    model_base: Optional[str] = None,
    devices: Optional[List[int]] = None,
    **kwargs,
) -> PreTrainedModel:
    """
    加载预训练的NaVILA/VILA模型
    
    这是加载模型的主要入口函数，提供了简化的接口来加载和配置模型。
    
    参数:
        model_path (str): 模型路径或HuggingFace模型ID
            - 本地路径: "/path/to/model" 
            - HuggingFace ID: "a8cheng/navila-siglip-llama3-8b-v1.5-pretrain"
        
        model_base (Optional[str]): 基础模型路径（用于LoRA等增量模型）
            - 如果加载的是完整模型，设为None
            - 如果加载的是LoRA权重，需要指定基础模型路径
        
        devices (Optional[List[int]]): 指定使用的GPU设备列表
            - 例如 [0, 1, 2, 3] 表示使用前4个GPU
            - None表示自动使用所有可用GPU
        
        **kwargs: 传递给模型加载器的其他参数
            - load_8bit: 是否使用8位量化
            - load_4bit: 是否使用4位量化
            - device_map: 设备映射策略
            等等
    
    返回:
        PreTrainedModel: 加载好的模型实例
    
    使用示例:
        # 加载完整模型
        model = load("a8cheng/navila-siglip-llama3-8b-v1.5-pretrain")
        
        # 加载到特定GPU
        model = load("path/to/model", devices=[0, 1])
        
        # 使用8位量化
        model = load("path/to/model", load_8bit=True)
    
    注意事项:
        - 函数会自动设置对话模式
        - 会自动处理模型路径的展开和验证
        - devices参数会自动配置每个GPU的最大内存
    """
    import torch

    # 导入必要的工具函数
    from llava.conversation import auto_set_conversation_mode
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    # 自动设置对话模式（根据模型类型设置不同的对话模板）
    auto_set_conversation_mode(model_path)

    # 从路径提取模型名称
    model_name = get_model_name_from_path(model_path)
    
    # 展开用户路径（例如 ~/model -> /home/user/model）
    model_path = os.path.expanduser(model_path)
    
    # 如果存在model子目录，使用它（某些checkpoint保存格式）
    if os.path.exists(os.path.join(model_path, "model")):
        model_path = os.path.join(model_path, "model")

    # 设置max_memory以限制使用哪些GPU
    if devices is not None:
        # 不能同时指定devices和max_memory
        assert "max_memory" not in kwargs, "`max_memory` should not be set when `devices` is set"
        
        # 为每个指定的设备配置最大内存（使用该GPU的全部显存）
        kwargs.update(max_memory={device: torch.cuda.get_device_properties(device).total_memory for device in devices})

    # 加载预训练模型（返回值是一个元组，[1]是模型本身）
    model = load_pretrained_model(model_path, model_name, model_base, **kwargs)[1]
    
    return model
