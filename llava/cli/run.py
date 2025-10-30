"""
SLURM任务运行工具

该脚本是一个SLURM集群任务提交的命令行工具，用于简化在SLURM集群上运行训练/评估任务。
主要功能：
- 自动生成SLURM任务配置
- 支持任务超时自动重启
- 统一的输出目录管理
- 环境变量自动设置

使用方法:
    vila-run -J my_job -N 2 --gpus-per-node 8 -m train python train.py
    
要求:
    必须设置以下环境变量：
    - VILA_SLURM_ACCOUNT: SLURM账户名
    - VILA_SLURM_PARTITION: SLURM分区名
"""

import argparse
import datetime
import os
import subprocess


def main() -> None:
    """
    主函数：解析参数并提交SLURM任务
    
    命令行参数:
        --job-name, -J: 任务名称（必需）
            - 支持 %t 占位符，会被替换为当前时间戳
            - 例如: "experiment-%t" → "experiment-20231130143022"
        
        --nodes, -N: 节点数量（默认1）
            - 指定使用多少个计算节点
        
        --gpus-per-node: 每个节点的GPU数（默认8）
            - 指定每个节点使用的GPU数量
        
        --mode, -m: 运行模式（默认"train"）
            - 可选: "train", "eval", "test" 等
            - 用于组织输出目录结构
        
        --time, -t: 任务时长限制（默认"4:00:00"）
            - 格式: "HH:MM:SS"
            - 必须至少5分钟
        
        cmd: 实际要执行的命令（剩余所有参数）
            - 例如: python train.py --config config.yaml
    
    工作流程:
        1. 解析命令行参数
        2. 生成任务名称和输出目录
        3. 计算超时时间（实际时间 - 5分钟）
        4. 获取SLURM配置（账户和分区）
        5. 构建srun命令
        6. 执行任务，超时则自动重启
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="在SLURM集群上运行训练/评估任务的工具",
        epilog="示例: vila-run -J my_exp -N 2 -m train python train.py"
    )
    parser.add_argument("--job-name", "-J", type=str, required=True,
                       help="任务名称（支持%%t时间戳占位符）")
    parser.add_argument("--nodes", "-N", type=int, default=1,
                       help="节点数量（默认: 1）")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                       help="每个节点的GPU数（默认: 8）")
    parser.add_argument("--mode", "-m", type=str, default="train",
                       help="运行模式（默认: train）")
    parser.add_argument("--time", "-t", type=str, default="4:00:00",
                       help="任务时长限制，格式HH:MM:SS（默认: 4:00:00）")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                       help="要执行的命令及其参数")
    args = parser.parse_args()

    # ==================== 步骤1: 生成任务名称和输出目录 ====================
    # 处理时间戳占位符：%t → YYYYMMDDHHMMSS
    if "%t" in args.job_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        args.job_name = args.job_name.replace("%t", timestamp)
    
    # 构建输出目录结构：runs/{mode}/{job_name}/
    # 例如: runs/train/my_experiment_20231130143022/
    output_dir = os.path.join("runs", args.mode, args.job_name)

    # ==================== 步骤2: 计算超时时间 ====================
    # 解析时间字符串
    time = datetime.datetime.strptime(args.time, "%H:%M:%S")
    
    # 检查最小时长要求（至少5分钟）
    if time < datetime.datetime.strptime("0:05:00", "%H:%M:%S"):
        raise ValueError("Time must be at least 5 minutes")
    
    # 计算超时时间：实际时间 - 5分钟（留出保存checkpoint的时间）
    timeout = time - datetime.timedelta(minutes=5)
    timeout_minutes = timeout.hour * 60 + timeout.minute
    timeout = f"{timeout_minutes}m"  # 转换为timeout命令的格式

    # ==================== 步骤3: 获取SLURM配置 ====================
    # 从环境变量读取SLURM账户和分区信息
    if "VILA_SLURM_ACCOUNT" not in os.environ or "VILA_SLURM_PARTITION" not in os.environ:
        raise ValueError(
            "`VILA_SLURM_ACCOUNT` and `VILA_SLURM_PARTITION` must be set in the environment.\n"
            "请设置: export VILA_SLURM_ACCOUNT=your_account\n"
            "       export VILA_SLURM_PARTITION=your_partition"
        )
    account = os.environ["VILA_SLURM_ACCOUNT"]
    partition = os.environ["VILA_SLURM_PARTITION"]

    # ==================== 步骤4: 设置环境变量 ====================
    # 复制当前环境并添加任务特定的变量
    env = os.environ.copy()
    env["RUN_NAME"] = args.job_name      # 任务名称（供训练脚本使用）
    env["OUTPUT_DIR"] = output_dir       # 输出目录（供训练脚本使用）

    # ==================== 步骤5: 构建SLURM命令 ====================
    # 构建srun命令（SLURM的任务执行命令）
    cmd = ["srun"]
    cmd += ["--account", account]                                    # SLURM账户
    cmd += ["--partition", partition]                                # SLURM分区
    cmd += ["--job-name", f"{account}:{args.mode}/{args.job_name}"] # 任务标识
    cmd += ["--output", f"{output_dir}/slurm/%J-%t.out"]           # 标准输出文件
    cmd += ["--error", f"{output_dir}/slurm/%J-%t.err"]            # 标准错误文件
    cmd += ["--nodes", str(args.nodes)]                             # 节点数
    cmd += ["--gpus-per-node", str(args.gpus_per_node)]            # 每节点GPU数
    cmd += ["--time", args.time]                                    # 时长限制
    cmd += ["--exclusive"]                                          # 独占节点
    cmd += ["timeout", timeout]                                     # Linux timeout命令
    cmd += args.cmd                                                 # 用户指定的实际命令
    
    # 打印完整的命令（便于调试）
    print("=" * 70)
    print("SLURM任务配置:")
    print(f"  任务名称: {args.job_name}")
    print(f"  节点数: {args.nodes} × {args.gpus_per_node} GPUs")
    print(f"  时长限制: {args.time}")
    print(f"  输出目录: {output_dir}")
    print("=" * 70)
    print("执行命令:")
    print(" ".join(cmd))
    print("=" * 70)

    # ==================== 步骤6: 运行任务并自动重启 ====================
    # 自动重启机制：如果任务因超时而中止（返回码124），则自动重启
    attempt = 1
    while True:
        print(f"\n[尝试 {attempt}] 启动任务...")
        returncode = subprocess.run(cmd, env=env).returncode
        
        # 返回码124表示timeout命令的超时
        if returncode != 124:
            break  # 任务正常结束或因其他原因失败
        
        # 任务超时，准备重启
        print("⚠️  任务超时，准备自动重启...")
        print("   (确保你的训练脚本支持从checkpoint恢复)")
        attempt += 1
    
    # 打印最终状态
    if returncode == 0:
        print(f"✓ 任务成功完成（退出码: {returncode}）")
    else:
        print(f"✗ 任务失败（退出码: {returncode}）")


if __name__ == "__main__":
    main()
