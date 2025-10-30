"""
训练模块

该模块包含NaVILA模型的训练逻辑和优化策略：
- 分布式训练支持（DeepSpeed、序列并行）
- 内存优化（梯度检查点、ZeRO优化）
- 长序列训练支持
- 自定义Trainer实现
- SLURM集群支持

主要组件：
- train.py: 基础训练脚本
- train_mem.py: 内存优化训练脚本
- train_long.py: 长序列训练脚本
- llava_trainer.py: 自定义Trainer类
- args.py: 训练参数定义
- sequence_parallel/: 序列并行实现
- deepspeed_replace/: DeepSpeed自定义组件
"""
