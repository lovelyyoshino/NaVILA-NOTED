"""
NaVILA模型定义模块

该模块包含NaVILA/VILA模型的核心实现，支持多种语言模型后端：
- LLaMA系列：LlavaLlamaModel（基于LLaMA的多模态模型）
- Mistral：LlavaMistralForCausalLM（基于Mistral的多模态模型）
- Mixtral：LlavaMixtralForCausalLM（基于Mixtral MoE的多模态模型）

主要组件：
- language_model/: 语言模型后端实现
- multimodal_encoder/: 视觉编码器（SigLIP, CLIP, InternViT等）
- multimodal_projector/: 多模态投影器（连接视觉和语言）
- llava_arch.py: 核心架构定义
- builder.py: 模型构建器
"""

# 导入LLaMA后端的配置和模型
from .language_model.llava_llama import LlavaLlamaConfig, LlavaLlamaModel

# 导入Mistral后端的配置和模型  
from .language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM

# 导入Mixtral MoE后端的配置和模型
from .language_model.llava_mixtral import LlavaMixtralConfig, LlavaMixtralForCausalLM
