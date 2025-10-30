# LLaVA/VILA 模块说明文档

## 📖 模块概述

`llava/` 模块是NaVILA项目的核心多模态处理模块，基于LLaVA (Large Language and Vision Assistant) 架构。该模块提供了完整的视觉-语言模型实现，支持图像理解、视频理解和视觉-语言导航任务。

**核心特性：**
- ✨ 多模态融合：支持图像、视频、文本的统一处理
- 🔧 模块化设计：可自由组合语言模型、视觉编码器和投影器
- 🚀 分布式训练：支持多节点、多GPU的高效训练
- 📊 任务评估：内置多种benchmark的评估工具
- 🎯 导航任务：专门优化的视觉-语言导航能力

## 🏗️ 目录结构详解

### 📁 根目录文件

```
llava/
├── __init__.py                 # 模块初始化文件
│                                # 导出核心功能: load(), Image, Video等
│
├── constants.py                # 全局常量定义
│                                # - 特殊token索引 (IMAGE_TOKEN_INDEX, IGNORE_INDEX)
│                                # - 特殊token字符串 (<image>, <video>等)
│                                # - VILA哨兵标记
│
├── conversation.py             # 对话管理系统 ⭐核心
│                                # - SeparatorStyle: 不同对话格式的分隔符样式
│                                # - Conversation: 对话状态管理类
│                                # - 预定义模板: llama_3, mistral, vicuna等
│                                # - auto_set_conversation_mode(): 自动选择模板
│
├── entry.py                    # 模型加载入口 ⭐推荐
│                                # - load(): 简化的模型加载接口
│                                # - 自动配置GPU分配和内存管理
│
├── media.py                    # 媒体类型定义
│                                # - Media: 媒体基类
│                                # - File: 文件媒体类
│                                # - Image: 图像类
│                                # - Video: 视频类
│
└── mm_utils.py                 # 多模态工具函数 ⭐重要
                                 # - vlnce_frame_sampling(): VLN-CE任务的帧采样
                                 # - opencv_extract_frames(): 视频帧提取
                                 # - process_images(): 图像预处理
                                 # - tokenizer_image_token(): 特殊token处理
                                 # - KeywordsStoppingCriteria: 生成停止条件
```

### 📁 cli/ - 命令行工具

```
cli/
├── run.py                      # SLURM任务运行工具 ⭐集群必备
│                                # 功能:
│                                # - 简化SLURM任务提交流程
│                                # - 支持任务超时自动重启
│                                # - 统一的输出目录管理
│                                # 使用:
│                                # vila-run -J my_job -N 2 --gpus-per-node 8 \
│                                #   -m train python train.py
│                                # 要求:
│                                # - 设置 VILA_SLURM_ACCOUNT 环境变量
│                                # - 设置 VILA_SLURM_PARTITION 环境变量
│
└── eval.py                     # 模型评估批处理工具 ⭐评估必备
                                 # 功能:
                                 # - 在多个benchmark上批量评估模型
                                 # - 支持任务过滤（按名称、标签）
                                 # - 自动跳过已完成的任务
                                 # - 结果汇总和可视化
                                 # 使用:
                                 # vila-eval -m /path/to/model -c llama_3
                                 # vila-eval -m model -c llama_3 -t mme,pope
                                 # vila-eval -m model -c llama_3 -i image
```

### 📁 data/ - 数据处理模块

```
data/
├── __init__.py                 # 数据模块初始化
│                                # 说明数据加载和预处理功能
│
├── base.py                     # 数据集基类
│                                # - BaseDataset: 所有数据集的基类
│                                # - 定义标准接口和通用方法
│
├── builder.py                  # 数据集构建器 ⭐核心
│                                # - make_data_module(): 创建数据模块
│                                # - 根据配置自动构建DataLoader
│                                # - 支持数据集混合
│
├── dataset.py                  # 标准数据集实现
│                                # - LazySupervisedDataset: 延迟加载数据集
│                                # - 处理JSON格式的对话数据
│                                # - 支持图像和视频
│
├── datasets_mixture.py         # 多数据集混合 ⭐重要
│                                # - DatasetMixture: 混合多个数据集
│                                # - 支持采样比例控制
│                                # - 动态数据增强
│
├── simple_vila_webdataset.py  # WebDataset支持
│                                # - VILA格式的WebDataset加载器
│                                # - 高效的流式数据处理
│
├── utils.py                    # 数据工具函数
│                                # - 数据预处理函数
│                                # - 数据格式转换
│
├── dataset_impl/               # 具体数据集实现
│   ├── __init__.py
│   └── llava.py               # LLaVA格式数据集
│                                # - 支持ShareGPT格式
│                                # - 支持多轮对话
│
└── registry/                   # 数据集注册表
    └── default.yaml           # 默认数据集配置
                                 # - 注册所有可用数据集
                                 # - 配置数据路径和格式
```

### 📁 eval/ - 评估模块

```
eval/
├── __init__.py                 # 评估模块初始化
│                                # - EVAL_ROOT: 评估脚本根目录
│                                # - TASKS: 所有评估任务的配置
│
├── run_navigation.py          # 导航任务评估 ⭐导航专用
│                                # - R2R评估
│                                # - RxR评估
│                                # - 自动计算SPL、SR等指标
│
├── run_vila.py                # VILA模型通用评估
│                                # - 图像理解任务
│                                # - 视频理解任务
│
├── eval_textvqa.py            # TextVQA评估
├── eval_refcoco.py            # RefCOCO目标定位评估
├── eval_mathvista.py          # MathVista数学推理评估
├── eval_mmmu.py               # MMMU多模态理解评估
│
├── model_vqa_*.py             # 各种VQA任务的模型接口
│   ├── model_vqa_loader.py   # 通用VQA加载器
│   ├── model_vqa_video.py    # 视频VQA
│   ├── model_vqa_ego_schema.py  # EgoSchema评估
│   └── ...
│
├── lmms/                      # LMMS评估框架集成
│   ├── models/               # LMMS模型接口
│   └── tasks/                # LMMS任务定义
│
├── mathvista_utils/          # MathVista工具
├── mmmu_utils/               # MMMU工具
├── vision_niah_vila/         # Vision NIAH评估
│
└── registry.yaml             # 评估任务注册表
                               # - 定义所有可评估的任务
                               # - 指定评估脚本和指标路径
```

### 📁 model/ - 模型定义模块

```
model/
├── __init__.py                 # 模型模块初始化 ⭐重要
│                                # - 导出所有模型类
│                                # - 注册HuggingFace模型
│
├── builder.py                  # 模型构建器 ⭐核心
│                                # - load_pretrained_model(): 加载预训练模型
│                                # - 处理模型合并和LoRA
│
├── llava_arch.py              # 核心架构定义 ⭐最重要
│                                # - LlavaMetaModel: 模型基类
│                                # - LlavaMetaForCausalLM: 因果语言模型基类
│                                # - 定义encode_images()等核心方法
│
├── configuration_llava.py     # 模型配置类
│                                # - LlavaConfig: 继承自PretrainedConfig
│                                # - 定义所有模型超参数
│
├── loss.py                    # 损失函数
│                                # - 自定义损失函数
│                                # - 多任务损失组合
│
├── utils.py                   # 模型工具函数
│
├── apply_delta.py             # 应用模型增量
├── make_delta.py              # 生成模型增量
├── consolidate.py             # 模型合并工具
│
├── language_model/            # 语言模型后端 ⭐可扩展
│   ├── builder.py            # 语言模型构建器
│   │
│   ├── llava_llama.py        # LLaMA后端 (推荐)
│   │                          # - LlavaLlamaForCausalLM
│   │                          # - 支持LLaMA 2/3
│   │
│   ├── llava_mistral.py      # Mistral后端
│   │                          # - LlavaMistralForCausalLM
│   │                          # - 7B高效模型
│   │
│   ├── llava_mixtral.py      # Mixtral后端 (MoE)
│   │                          # - LlavaMixtralForCausalLM
│   │                          # - 专家混合架构
│   │
│   ├── llava_gemma.py        # Gemma后端
│   │                          # - LlavaGemmaForCausalLM
│   │                          # - Google开源模型
│   │
│   ├── llava_qwen.py         # Qwen后端
│   ├── llava_phi3.py         # Phi-3后端
│   └── ...
│
├── multimodal_encoder/       # 视觉编码器 ⭐可选择
│   ├── builder.py           # 视觉编码器构建器
│   │
│   ├── siglip_encoder.py    # SigLIP编码器 (推荐)
│   │                         # - Google的SigLIP模型
│   │                         # - 高质量视觉特征
│   │
│   ├── clip_encoder.py      # CLIP编码器
│   │                         # - OpenAI的CLIP模型
│   │                         # - 经典选择
│   │
│   ├── intern_encoder.py    # InternViT编码器
│   │                         # - 高分辨率支持
│   │                         # - 6B参数
│   │
│   ├── radio_encoder.py     # RADIO编码器
│   │                         # - 鲁棒性视觉编码
│   │
│   └── ...
│
└── multimodal_projector/     # 多模态投影器
    ├── builder.py           # 投影器构建器
    │
    └── base_projector.py    # 基础投影器实现
                              # - MLPProjector: 简单MLP
                              # - MLPDownsampleProjector: 带下采样
                              # - 将视觉token映射到语言空间
```

### 📁 train/ - 训练模块

```
train/
├── __init__.py                 # 训练模块初始化
│                                # 说明训练流程和优化策略
│
├── train.py                    # 基础训练脚本
│                                # - 标准的Transformers Trainer训练流程
│                                # - 适用于简单场景
│
├── train_mem.py                # 内存优化训练 ⭐推荐
│                                # - 启用梯度检查点
│                                # - DeepSpeed ZeRO集成
│                                # - 适合大模型和长序列
│
├── train_long.py               # 长序列训练
│                                # - 序列并行支持
│                                # - 处理超长上下文
│
├── train_hybrid.py             # 混合并行训练
│                                # - 数据并行 + 模型并行 + 序列并行
│                                # - 最大化训练效率
│
├── llava_trainer.py           # 自定义Trainer ⭐核心
│                                # - LLaVATrainer: 继承自Transformers Trainer
│                                # - 自定义训练循环
│                                # - 支持多模态数据
│
├── args.py                    # 训练参数定义
│                                # - ModelArguments: 模型相关参数
│                                # - DataArguments: 数据相关参数
│                                # - TrainingArguments: 训练相关参数
│
├── utils.py                   # 训练工具函数
│                                # - 学习率调度
│                                # - Checkpoint管理
│
├── callbacks/                 # 训练回调
│   └── autoresume_callback.py # 自动恢复回调
│                                # - 支持训练中断后自动恢复
│
├── sequence_parallel/         # 序列并行实现 ⭐高级
│   ├── ulysses_attn.py       # Ulysses注意力
│   │                          # - 跨GPU的注意力并行
│   │
│   ├── hybrid_attn.py        # 混合注意力
│   │                          # - Ulysses + Ring的混合策略
│   │
│   └── ring/                 # Ring注意力
│       └── ...                # - 环形通信的注意力并行
│
├── deepspeed_replace/         # DeepSpeed自定义模块
│   └── ...                    # - 替换DeepSpeed的默认实现
│                               # - 优化通信和内存
│
└── transformers_replace/      # Transformers自定义模块
    └── ...                    # - 替换Transformers的默认实现
                                # - 优化推理和训练
```

### 📁 utils/ - 通用工具

```
utils/
├── __init__.py                 # 工具模块初始化
├── distributed.py              # 分布式工具
│                                # - 初始化分布式环境
│                                # - 进程间通信
│
├── io.py                       # IO工具
│                                # - JSON/YAML加载和保存
│                                # - 安全的文件操作
│
├── logging.py                  # 日志工具
│                                # - 统一的日志格式
│                                # - 多进程日志管理
│
├── media.py                    # 媒体处理工具
│                                # - 图像/视频格式转换
│
├── tokenizer.py                # Tokenizer工具
│                                # - 特殊token处理
│
└── merge_lora_weights_and_save_hf_model.py  # LoRA合并工具
                                              # - 将LoRA权重合并到基础模型
```

### 📁 其他模块

```
trl/                            # 强化学习模块 (TRL集成)
├── trainer/                    # RLHF Trainer
└── models/                     # 价值模型和奖励模型

wids/                           # WebDataset索引系统
├── wids.py                    # 核心索引类
├── wids_dl.py                 # 下载工具
└── ...                        # 索引管理和优化

data_aug/                       # 数据增强工具
├── caption2qa.py              # 标题转问答
└── video_inference.py         # 视频推理
```


## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
cd NaVILA-NOTED
pip install -e .

# 设置环境变量（如使用SLURM）
export VILA_SLURM_ACCOUNT=your_account
export VILA_SLURM_PARTITION=your_partition
```

### 2. 加载模型

```python
import llava

# 方法1: 使用简化的load接口（推荐）
model = llava.load("a8cheng/navila-siglip-llama3-8b-v1.5-pretrain")

# 方法2: 指定设备
model = llava.load(
    "path/to/model",
    devices=[0, 1],  # 使用GPU 0和1
    load_8bit=True   # 8-bit量化
)

# 方法3: 更底层的加载方式
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="a8cheng/navila-siglip-llama3-8b-v1.5-pretrain",
    model_base=None,
    model_name="navila"
)
```

### 3. 图像理解

```python
from llava.media import Image
from llava.conversation import conv_llama_3
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

# 加载图像
image = Image("/path/to/image.jpg")

# 准备对话
conv = conv_llama_3.copy()
conv.append_message(conv.roles[0], "<image>\n请描述这张图片中的内容")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# 处理图像
image_tensor = process_images([image.data], image_processor, model.config)

# Token化
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)

# 推理
with torch.inference_mode():
    output_ids = model.generate(
        input_ids.unsqueeze(0).cuda(),
        images=image_tensor.cuda(),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )

# 解码输出
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

### 4. 视频理解

```python
from llava.media import Video
from llava.mm_utils import opencv_extract_frames, process_images

# 从视频提取帧
frames = opencv_extract_frames(
    video_path="/path/to/video.mp4",
    frames=8,   # 提取8帧
    fps=3       # 最大FPS限制
)

# 处理视频帧（与图像处理相同）
video_tensor = process_images(frames, image_processor, model.config)

# 推理（使用8个<image> token）
prompt = "<image> " * 8 + "\n描述这个视频中发生了什么"
# ... 后续处理与图像相同
```

### 5. 训练模型

**单机多卡训练：**
```bash
# 设置环境变量
export n_node=1
export GPUS_PER_NODE=4
export MASTER_PORT=29500
export MASTER_ADDR=localhost
export CURRENT_RANK=0

# 运行训练脚本
bash scripts/train/sft_8frames.sh
```

**SLURM集群训练：**
```bash
# 使用vila-run工具
vila-run \
  -J navila_training \
  -N 2 \
  --gpus-per-node 8 \
  -m train \
  -t 48:00:00 \
  bash scripts/train/sft_8frames.sh
```

**训练参数说明：**
```bash
# 在 sft_8frames.sh 中修改关键参数：
--model_name_or_path      # 基础模型路径
--vision_tower            # 视觉编码器
--data_mixture            # 数据集混合
--num_video_frames 8      # 视频帧数
--per_device_train_batch_size 10  # 批次大小
--learning_rate 1e-4      # 学习率
--num_train_epochs 1      # 训练轮数
```

### 6. 评估模型

**评估所有任务：**
```bash
vila-eval \
  -m /path/to/checkpoint \
  -c llama_3 \
  -n 8
```

**评估特定任务：**
```bash
# 只评估MME和POPE
vila-eval \
  -m /path/to/checkpoint \
  -c llama_3 \
  -t mme,pope

# 只评估图像任务（排除视频）
vila-eval \
  -m /path/to/checkpoint \
  -c llama_3 \
  -i image \
  -e video
```

## 🔧 配置说明

### 模型配置

**关键参数：**
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model_name_or_path` | 基础语言模型路径 | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `--vision_tower` | 视觉编码器 | `google/siglip-so400m-patch14-384` |
| `--mm_projector_type` | 投影器类型 | `mlp_downsample` |
| `--num_video_frames` | 视频帧数 | `8` |
| `--mm_vision_select_layer` | 选择视觉层 | `-2` (倒数第二层) |

**组合推荐：**
```bash
# 配置1: SigLIP + LLaMA-3-8B (推荐)
--model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
--vision_tower google/siglip-so400m-patch14-384 \
--mm_projector_type mlp_downsample

# 配置2: CLIP + Mistral-7B (经济)
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--vision_tower openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp

# 配置3: InternViT + LLaMA-3-70B (高性能)
--model_name_or_path meta-llama/Meta-Llama-3-70B-Instruct \
--vision_tower OpenGVLab/InternViT-6B-448px-V1-5 \
--mm_projector_type mlp_downsample
```

### 数据配置

**数据集混合策略：**
```bash
# 导航任务专用
--data_mixture r2r+rxr+envdrop+human+scanqa

# 通用视觉理解
--data_mixture sharegpt4v+video_chatgpt+llava_instruct

# 混合训练（导航 + 通用）
--data_mixture r2r+rxr+sharegpt4v+video_chatgpt
```

**数据路径配置：**
```bash
--data_path /path/to/NaVILA-Dataset  # 数据集根目录
--image_folder /path/to/images       # 图像文件夹
--video_folder /path/to/videos       # 视频文件夹
```

### 训练配置

**批次大小计算：**
```
全局批次大小 = per_device_batch_size × num_gpus × gradient_accumulation_steps
```

**配置示例：**
| 场景 | per_device_batch | accumulation_steps | 全局批次 | 显存需求 |
|------|------------------|--------------------|----------|----------|
| 小规模实验 | 4 | 1 | 32 (8 GPU) | ~24GB |
| 标准训练 | 10 | 2 | 160 (8 GPU) | ~40GB |
| 大规模训练 | 16 | 4 | 512 (8 GPU) | ~80GB |

**学习率策略：**
```bash
--learning_rate 1e-4              # 基础学习率
--lr_scheduler_type cosine        # 余弦退火
--warmup_ratio 0.03               # 预热比例
--weight_decay 0.0                # 权重衰减
```

**保存策略：**
```bash
--save_strategy epoch             # 每个epoch保存
--save_total_limit 2              # 最多保留2个checkpoint
--save_steps 1000                 # 或每1000步保存
```

## 📊 支持的任务

### 1. 视觉-语言导航 (VLN) ⭐核心任务

| 任务 | 数据集 | 指标 | 说明 |
|------|--------|------|------|
| **室内导航** | R2R | SR, SPL, nDTW | 根据指令在室内场景中导航 |
| **多语言导航** | RxR | SR, SPL | 支持英语、印地语、泰卢固语 |
| **环境泛化** | EnvDrop | SR, SPL | 测试模型在新环境的泛化能力 |
| **3D场景问答** | ScanQA | CIDEr, BLEU | 3D场景中的问答任务 |

### 2. 视频理解

| 任务 | Benchmark | 说明 |
|------|-----------|------|
| **视频问答** | Video-ChatGPT | 开放式视频问答 |
| **视频描述** | MSVD, MSR-VTT | 视频标题生成 |
| **时间推理** | NExT-QA | 时序关系理解 |
| **长视频理解** | EgoSchema | 长视频场景理解 |

### 3. 图像理解

| 任务 | Benchmark | 说明 |
|------|-----------|------|
| **通用VQA** | VQAv2, GQA | 图像问答 |
| **OCR** | TextVQA, DocVQA | 文本识别和理解 |
| **细粒度识别** | RefCOCO | 目标定位和引用 |
| **推理** | POPE, MME | 幻觉检测和多模态评估 |
| **数学推理** | MathVista | 视觉数学问题 |
| **科学推理** | MMMU | 多学科理解 |

### 4. 特殊能力

- **多语言支持**: 英语、中文、印地语、泰卢固语
- **空间推理**: 3D场景理解和空间关系
- **时序推理**: 视频中的时间关系理解
- **长上下文**: 支持超长序列（>32K tokens）

## 🛠️ 开发指南

### 1. 添加新的数据集

**步骤：**
```bash
# 1. 创建数据集实现文件
touch llava/data/dataset_impl/my_dataset.py
```

```python
# 2. 实现数据集类
from llava.data.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__()
        # 加载数据
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回单个样本
        return {
            "image": image_path,
            "conversations": conversations
        }
```

```yaml
# 3. 在 data/registry/default.yaml 注册
my_dataset:
  type: llava.data.dataset_impl.my_dataset.MyDataset
  data_path: /path/to/data
  description: "我的自定义数据集"
```

```bash
# 4. 使用数据集
--data_mixture my_dataset+r2r
```

### 2. 添加新的模型后端

**步骤：**
```bash
# 1. 创建模型文件
touch llava/model/language_model/llava_mymodel.py
```

```python
# 2. 实现模型类
from llava.model.llava_arch import LlavaMetaForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig

class LlavaMyModelConfig(LlavaConfig):
    model_type = "llava_mymodel"

class LlavaMyModelForCausalLM(MyModelForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMyModelConfig
    
    def __init__(self, config):
        super().__init__(config)
```

```python
# 3. 在 model/__init__.py 注册
from .language_model.llava_mymodel import (
    LlavaMyModelConfig,
    LlavaMyModelForCausalLM
)

AutoConfig.register("llava_mymodel", LlavaMyModelConfig)
AutoModelForCausalLM.register(LlavaMyModelConfig, LlavaMyModelForCausalLM)
```

### 3. 添加新的视觉编码器

**步骤：**
```bash
# 1. 创建编码器文件
touch llava/model/multimodal_encoder/my_encoder.py
```

```python
# 2. 实现编码器类
import torch.nn as nn

class MyVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.vision_tower_name = vision_tower
        # 初始化视觉编码器
        
    def forward(self, images):
        # 编码图像
        return image_features
```

```python
# 3. 在 multimodal_encoder/builder.py 注册
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if 'my_encoder' in vision_tower.lower():
        return MyVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # ...
```

### 4. 调试技巧

**打印模型架构：**
```python
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

**监控显存使用：**
```bash
watch -n 1 nvidia-smi
```

**启用详细日志：**
```bash
export LOGLEVEL=DEBUG
python your_script.py
```

**使用小数据集测试：**
```bash
# 快速验证流程
--max_steps 10 \
--save_steps 5 \
--eval_steps 5
```

## 📝 注意事项

1. **内存管理**: 
   - 视频任务需要大量显存
   - 使用梯度检查点和DeepSpeed ZeRO
   - 建议使用`train_mem.py`

2. **数据预处理**:
   - 视频帧提取可能很慢
   - 建议预先提取帧到磁盘
   - 使用`lazy_preprocess`延迟加载

3. **分布式训练**:
   - 确保所有节点网络互通
   - 正确设置环境变量
   - 使用相同的配置文件

4. **模型兼容性**:
   - 不同语言模型需要不同的对话模板
   - 注意tokenizer的特殊token
   - 检查模型配置文件

## 🔗 相关资源

- **LLaVA原始项目**: https://github.com/haotian-liu/LLaVA/
- **VILA项目**: https://github.com/Efficient-Large-Model/VILA

