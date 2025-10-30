"""
LLaVA/VILA模块初始化文件

该模块是NaVILA项目的核心多模态处理模块，基于LLaVA架构。
提供模型加载和媒体处理的基础功能。
"""

# 导入模型加载入口函数
from .entry import *

# 导入媒体类型定义（Image, Video, File等）
from .media import *
