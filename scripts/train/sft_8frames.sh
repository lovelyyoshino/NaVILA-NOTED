#!/bin/bash

################################################################################
# NaVILA模型监督微调（SFT）训练脚本
# 
# 功能：使用8帧视频输入对NaVILA-8B模型进行监督微调
# 模型：基于LLaMA3-8B和SigLIP视觉编码器的多模态导航模型
# 数据：混合使用导航任务数据集和通用视觉-语言数据集
################################################################################

################################################################################
# 使用示例
################################################################################
# 
# === 示例1: 单机单卡训练 ===
# export n_node=1                    # 单节点
# export GPUS_PER_NODE=1             # 1个GPU
# export MASTER_PORT=29500           # 主节点端口
# export MASTER_ADDR="localhost"     # 本地训练
# export CURRENT_RANK=0              # 节点排名0
# bash scripts/train/sft_8frames.sh
#
# === 示例2: 单机多卡训练（4卡）===
# export n_node=1                    # 单节点
# export GPUS_PER_NODE=4             # 4个GPU
# export MASTER_PORT=29500           # 主节点端口
# export MASTER_ADDR="localhost"     # 本地训练
# export CURRENT_RANK=0              # 节点排名0
# bash scripts/train/sft_8frames.sh
#
# === 示例3: 多机多卡训练（2机器x8卡=16卡）===
# 
# 在主节点（Node 0，IP: 192.168.1.100）运行：
# export n_node=2                    # 总共2个节点
# export GPUS_PER_NODE=8             # 每个节点8个GPU
# export MASTER_PORT=29500           # 主节点端口
# export MASTER_ADDR="192.168.1.100" # 主节点IP地址
# export CURRENT_RANK=0              # 当前节点排名：0（主节点）
# bash scripts/train/sft_8frames.sh
#
# 在工作节点（Node 1，IP: 192.168.1.101）运行：
# export n_node=2                    # 总共2个节点
# export GPUS_PER_NODE=8             # 每个节点8个GPU
# export MASTER_PORT=29500           # 主节点端口
# export MASTER_ADDR="192.168.1.100" # 主节点IP地址（指向Node 0）
# export CURRENT_RANK=1              # 当前节点排名：1（工作节点）
# bash scripts/train/sft_8frames.sh
#
# === 示例4: 使用SLURM集群 ===
# 如果使用SLURM，这些环境变量通常会自动设置：
# - n_node 可以从 $SLURM_NNODES 获取
# - GPUS_PER_NODE 可以从 $SLURM_GPUS_PER_NODE 获取
# - CURRENT_RANK 可以从 $SLURM_NODEID 获取
# - MASTER_ADDR 可以从 $SLURM_NODELIST 中提取
#
# SLURM脚本示例：
# #!/bin/bash
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:8
# #SBATCH --cpus-per-task=64
# 
# export n_node=$SLURM_NNODES
# export GPUS_PER_NODE=8
# export MASTER_PORT=29500
# export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# export CURRENT_RANK=$SLURM_NODEID
# 
# bash scripts/train/sft_8frames.sh
#
################################################################################

# ==================== 环境变量检查 ====================
# 检查必需的环境变量是否已设置
if [ -z "$n_node" ] || [ -z "$GPUS_PER_NODE" ] || [ -z "$MASTER_PORT" ] || \
   [ -z "$MASTER_ADDR" ] || [ -z "$CURRENT_RANK" ]; then
    echo "错误: 缺少必需的环境变量！"
    echo ""
    echo "请设置以下环境变量后再运行："
    echo "  export n_node=<节点数量>           # 例如: 1（单机）或 2（双机）"
    echo "  export GPUS_PER_NODE=<每节点GPU数> # 例如: 1, 4, 8"
    echo "  export MASTER_PORT=<端口号>        # 例如: 29500"
    echo "  export MASTER_ADDR=<主节点地址>    # 例如: localhost 或 192.168.1.100"
    echo "  export CURRENT_RANK=<当前节点排名> # 例如: 0（主节点）或 1,2,3...（工作节点）"
    echo ""
    echo "快速启动示例（单机单卡）："
    echo "  export n_node=1 GPUS_PER_NODE=1 MASTER_PORT=29500 MASTER_ADDR=localhost CURRENT_RANK=0"
    echo "  bash scripts/train/sft_8frames.sh"
    exit 1
fi

# 显示当前配置
echo "======================================================================"
echo "                    NaVILA训练配置信息"
echo "======================================================================"
echo "节点配置:"
echo "  总节点数:        $n_node"
echo "  每节点GPU数:     $GPUS_PER_NODE"
echo "  总GPU数:         $((n_node * GPUS_PER_NODE))"
echo "  当前节点排名:    $CURRENT_RANK"
echo ""
echo "网络配置:"
echo "  主节点地址:      $MASTER_ADDR"
echo "  主节点端口:      $MASTER_PORT"
echo ""
echo "训练配置:"
echo "  输出目录:        $OUTPUT"
echo "  批次大小/GPU:    10"
echo "  梯度累积步数:    2"
echo "  有效批次大小:    $((10 * 2 * n_node * GPUS_PER_NODE))"
echo "======================================================================"
echo ""

# 模型checkpoint输出目录
OUTPUT="./checkpoints/navila-8b-8f-sft"

# ==================== 分布式训练启动命令 ====================
# 使用PyTorch的torchrun进行多节点、多GPU分布式训练
torchrun --nnodes=$n_node \                      # 节点数量（从环境变量读取）
    --nproc_per_node=$GPUS_PER_NODE \            # 每个节点的GPU数量
    --master_port=$MASTER_PORT \                  # 主节点端口号
    --master_addr $MASTER_ADDR \                  # 主节点地址
    --node_rank=$CURRENT_RANK \                   # 当前节点的排名
    llava/train/train_mem.py \                    # 训练脚本（内存优化版本）
    \
    # ==================== 采样与优化配置 ====================
    --longvila_sampler True \                     # 启用LongVILA采样器（处理长视频序列）
    --deepspeed ./scripts/zero3.json \            # DeepSpeed ZeRO-3配置文件路径
    \
    # ==================== 模型配置 ====================
    --model_name_or_path a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \  # 预训练模型路径
    --version llama_3 \                           # 模型版本（LLaMA3）
    --seed 10 \                                   # 随机种子，保证实验可复现
    \
    # ==================== 数据集配置 ====================
    # 混合多个数据集进行训练（使用+连接）：
    # - r2r: Room-to-Room导航数据集
    # - rxr: Multilingual Room-Across-Room导航数据集
    # - envdrop: 环境变化鲁棒性数据集
    # - human: 人类导航演示数据
    # - scanqa: 3D场景问答数据集
    # - video_chatgpt: 视频对话数据集
    # - sharegpt_video: ShareGPT视频数据
    # - sharegpt4v_sft: ShareGPT4V监督微调数据
    --data_mixture r2r+rxr+envdrop+human+scanqa+video_chatgpt+sharegpt_video+sharegpt4v_sft \
    \
    # ==================== 视觉编码器配置 ====================
    --vision_tower google/siglip-so400m-patch14-384 \  # SigLIP视觉编码器（384分辨率）
    --mm_vision_select_feature cls_patch \        # 同时使用CLS token和patch特征
    --mm_projector mlp_downsample \               # 多层感知机投影器（带下采样）
    --num_video_frames 8 \                        # 每个视频样本使用8帧
    --tune_vision_tower True \                    # 微调视觉编码器
    --tune_mm_projector True \                    # 微调多模态投影器
    --tune_language_model True \                  # 微调语言模型
    --mm_vision_select_layer -2 \                 # 选择倒数第2层的视觉特征
    --mm_use_im_start_end False \                 # 不使用图像起始/结束标记
    --mm_use_im_patch_token False \               # 不使用图像patch token
    --image_aspect_ratio resize \                 # 图像宽高比处理方式：直接resize
    \
    # ==================== 训练精度配置 ====================
    --bf16 True \                                 # 使用BF16混合精度训练（适用于A100等GPU）
    --tf32 True \                                 # 启用TF32（Tensor Float 32）加速
    \
    # ==================== 输出与保存配置 ====================
    --output_dir $OUTPUT \                        # 模型输出目录
    --save_strategy "steps" \                     # 按步数保存模型
    --save_steps 100 \                            # 每100步保存一次
    --save_total_limit 1 \                        # 最多保存1个checkpoint（节省空间）
    \
    # ==================== 训练超参数 ====================
    --num_train_epochs 1 \                        # 训练轮数：1个epoch
    --per_device_train_batch_size 10 \            # 每个GPU的批次大小：10
    --gradient_accumulation_steps 2 \             # 梯度累积步数：2（有效batch_size = 10*2*GPU数）
    --learning_rate 1e-4 \                        # 学习率：1e-4
    --weight_decay 0. \                           # 权重衰减：0（不使用L2正则化）
    --warmup_ratio 0.03 \                         # 预热比例：3%的训练步数用于预热
    --lr_scheduler_type "cosine" \                # 学习率调度器：余弦退火
    \
    # ==================== 评估与日志配置 ====================
    --do_eval False \                             # 不进行评估（纯训练模式）
    --logging_steps 1 \                           # 每步记录日志
    --report_to wandb \                           # 使用Weights & Biases记录训练过程
    \
    # ==================== 内存优化配置 ====================
    --model_max_length 4096 \                     # 模型最大序列长度：4096 tokens
    --gradient_checkpointing True \               # 启用梯度检查点（减少显存占用）
    --lazy_preprocess True \                      # 延迟预处理（在需要时才处理数据）
    --dataloader_num_workers 16 \                 # 数据加载器工作进程数：16
    \
    # ==================== 视频特定配置 ====================
    --fps 0.0                                     # 帧率设置：0.0表示自适应采样
