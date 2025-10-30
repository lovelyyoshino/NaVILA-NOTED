"""
数据处理模块

该模块负责NaVILA训练数据的加载、预处理和管理：
- 支持多种数据格式（JSON、WebDataset等）
- 处理多模态数据（图像、视频、文本）
- 数据集混合和采样策略
- 导航任务数据集（R2R、RxR、EnvDrop等）
- 通用视觉-语言数据集

主要组件：
- builder.py: 数据集构建器
- dataset.py: 数据集基类和实现
- dataset_impl/: 具体数据集实现
- datasets_mixture.py: 多数据集混合策略
- simple_vila_webdataset.py: WebDataset格式支持
"""

# 导入数据集构建函数
from .builder import *

# 导入数据集基类和工具
from .dataset import *

# 导入具体数据集实现
from .dataset_impl import *

# 导入数据集混合工具
from .datasets_mixture import *

# 导入VILA WebDataset实现
from .simple_vila_webdataset import VILAWebDataset
