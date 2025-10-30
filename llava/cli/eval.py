"""
模型评估批处理工具

该脚本是一个批量评估工具，用于在多个benchmark任务上评估NaVILA模型。
主要功能：
- 支持多任务并发评估
- 任务过滤（按名称、标签）
- 自动跳过已完成的任务
- 结果汇总和可视化
- SLURM集群支持

使用方法:
    # 评估所有任务
    vila-eval -m /path/to/model -c llama_3
    
    # 只评估特定任务
    vila-eval -m /path/to/model -c llama_3 -t mme,pope
    
    # 按标签过滤
    vila-eval -m /path/to/model -c llama_3 -i image --exclude-tags video

任务配置:
    任务列表在 llava.eval.TASKS 中定义，包含：
    - 任务名称和描述
    - 评估脚本路径
    - 指标提取路径
    - 任务标签（用于分类和过滤）
"""

import os
import subprocess
import time
from argparse import ArgumentParser
from collections import deque

from tabulate import tabulate

from llava.eval import EVAL_ROOT, TASKS
from llava.utils import io


def load_task_results(output_dir: str, task: str):
    """
    加载任务的评估结果
    
    参数:
        output_dir (str): 输出根目录
        task (str): 任务名称
    
    返回:
        dict or None: 结果字典，如果文件不存在则返回None
    
    结果文件路径:
        {output_dir}/{task}/results.json
    """
    results_path = os.path.join(output_dir, task, "results.json")
    if os.path.exists(results_path):
        return io.load(results_path)
    return None


def main() -> None:
    """
    主函数：批量运行评估任务并汇总结果
    
    命令行参数:
        --model-path, -m: 模型路径（必需）
            - 可以是本地路径或HuggingFace模型ID
            - 例如: "/path/to/model" 或 "a8cheng/navila-8b"
        
        --conv-mode, -c: 对话模式（必需）
            - 指定使用哪种对话模板
            - 例如: "llama_3", "vicuna_v1", "mistral"
        
        --nproc-per-node, -n: 每个节点的进程数（默认8）
            - 用于分布式评估
            - 通常设置为GPU数量
        
        --tasks, -t: 要运行的任务列表（可选）
            - 逗号分隔的任务名称
            - 例如: "mme,pope,gqa"
            - 不指定则运行所有任务
        
        --include-tags, -i: 包含的标签（可选）
            - 只运行包含这些标签的任务
            - 逗号分隔
            - 例如: "image,video"
        
        --exclude-tags, -e: 排除的标签（可选）
            - 排除包含这些标签的任务
            - 逗号分隔
            - 例如: "slow,deprecated"
    
    工作流程:
        1. 解析参数并过滤任务
        2. 检查已完成的任务（跳过）
        3. 为每个任务准备评估命令
        4. 并发运行评估任务
        5. 收集并汇总结果
        6. 生成结果表格
    """
    # 创建参数解析器
    parser = ArgumentParser(
        description="在多个benchmark上批量评估NaVILA模型",
        epilog="示例: vila-eval -m /path/to/model -c llama_3 -t mme,pope"
    )
    parser.add_argument("--model-path", "-m", type=str, required=True,
                       help="模型路径或HuggingFace ID")
    parser.add_argument("--conv-mode", "-c", type=str, required=True,
                       help="对话模式（如llama_3, vicuna_v1等）")
    parser.add_argument("--nproc-per-node", "-n", type=int, default=8,
                       help="每个节点的进程数（默认: 8）")
    parser.add_argument("--tasks", "-t", type=str,
                       help="要运行的任务（逗号分隔）")
    parser.add_argument("--include-tags", "-i", type=str,
                       help="包含的标签（逗号分隔）")
    parser.add_argument("--exclude-tags", "-e", type=str,
                       help="排除的标签（逗号分隔）")
    args = parser.parse_args()

    # ==================== 步骤1: 准备输出目录 ====================
    # 从模型路径提取模型名称（用于组织输出）
    model_name = os.path.basename(args.model_path).lower()
    # 构建输出目录结构：runs/eval/{model_name}/
    output_dir = os.path.join("runs", "eval", model_name)
    print(f"\n模型名称: {model_name}")
    print(f"输出目录: {output_dir}\n")

    # ==================== 步骤2: 过滤任务 ====================
    # 根据任务名称和标签过滤要运行的任务
    tasks = []
    skipped_tasks = []
    
    for task, metainfo in TASKS.items():
        tags = set(metainfo.get("tags", []))
        
        # 过滤条件1：任务名称
        if args.tasks is not None and task not in args.tasks.split(","):
            skipped_tasks.append((task, "不在指定任务列表中"))
            continue
        
        # 过滤条件2：包含标签（任务必须有至少一个指定标签）
        if args.include_tags is not None and tags.isdisjoint(args.include_tags.split(",")):
            skipped_tasks.append((task, f"没有包含标签: {args.include_tags}"))
            continue
        
        # 过滤条件3：排除标签（任务不能有任何排除的标签）
        if args.exclude_tags is not None and tags.intersection(args.exclude_tags.split(",")):
            skipped_tasks.append((task, f"包含排除标签: {args.exclude_tags}"))
            continue
        
        tasks.append(task)
    
    print("=" * 70)
    print(f"将为模型 {model_name} 运行 {len(tasks)} 个评估任务")
    print(f"任务列表: {', '.join(tasks)}")
    if skipped_tasks:
        print(f"\n跳过 {len(skipped_tasks)} 个任务（不符合过滤条件）")
    print("=" * 70)

    # ==================== 步骤3: 准备评估命令 ====================
    cmds = {}  # 存储每个任务的命令
    
    for task in tasks:
        # 检查任务是否已完成
        if load_task_results(output_dir, task=task):
            print(f"⊙ 跳过 {task}（已有评估结果）")
            continue

        # 根据任务名称格式构建命令
        cmd = []
        if task.startswith("lmms-"):
            # LMMS框架的任务
            task_name = task.replace("lmms-", "")
            cmd += [f"{EVAL_ROOT}/lmms.sh", task_name, args.model_path]
        elif "-" in task:
            # 带数据集分割的任务（如 "gqa-test"）
            name, split = task.split("-")
            cmd += [f"{EVAL_ROOT}/{name}.sh", args.model_path, model_name, split]
        else:
            # 标准任务
            cmd += [f"{EVAL_ROOT}/{task}.sh", args.model_path, model_name]
        
        # 添加对话模式参数
        cmd += [args.conv_mode]

        # 根据运行环境决定并发策略
        if os.environ.get("SLURM_JOB_ID"):
            # 在SLURM环境中：串行运行
            concurrency = 1
            print(f"  检测到SLURM环境，将串行运行任务")
        else:
            # 本地环境：并发运行，使用vila-run包装
            concurrency = 10
            cmd = [f"vila-run -m eval -J {model_name}/{task}"] + cmd

        cmds[task] = " ".join(cmd)

    if not cmds:
        print("\n所有任务都已完成或被跳过！")
        # 仍然继续收集和显示结果
    
    # ==================== 步骤4: 设置环境变量 ====================
    env = os.environ.copy()
    env["NPROC_PER_NODE"] = str(args.nproc_per_node)

    # ==================== 步骤5: 并发运行评估命令 ====================
    print(f"\n开始运行评估（并发数: {concurrency}）...\n")
    
    remaining = deque(cmds.keys())  # 待运行的任务队列
    processes = {}                   # 当前运行的进程 {task: process}
    returncodes = {}                 # 任务返回码 {task: returncode}
    
    try:
        while remaining or processes:
            # 启动新任务（直到达到并发限制）
            while remaining and len(processes) < concurrency:
                task = remaining.popleft()
                cmd = cmds[task]
                print(f"▶ 启动任务: {task}")
                print(f"  命令: {cmd}")
                processes[task] = subprocess.Popen(cmd, env=env, shell=True)

            # 检查已完成的任务
            for task, process in list(processes.items()):
                if process.poll() is not None:
                    returncodes[task] = process.returncode
                    status = "✓" if process.returncode == 0 else "✗"
                    print(f"{status} 任务完成: {task} (退出码: {process.returncode})")
                    processes.pop(task)

            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，终止所有进程...")
        for task, process in processes.items():
            print(f"  终止: {task}")
            process.terminate()
        for task, process in processes.items():
            process.wait()
        print("所有进程已终止")
        return

    # ==================== 步骤6: 检查执行结果 ====================
    print("\n" + "=" * 70)
    print("任务执行总结:")
    failed_tasks = []
    for task, returncode in returncodes.items():
        if returncode != 0:
            print(f"  ✗ {task}: 失败（退出码 {returncode}）")
            failed_tasks.append(task)
        else:
            print(f"  ✓ {task}: 成功")
    
    if failed_tasks:
        print(f"\n警告: {len(failed_tasks)} 个任务失败")
    print("=" * 70)

    # ==================== 步骤7: 收集并汇总结果 ====================
    print("\n收集评估结果...")
    metrics = {}
    
    for task in tasks:
        # 加载任务结果
        results = load_task_results(output_dir, task=task)
        if results is None:
            print(f"  ⚠ {task}: 未找到结果文件")
            continue
        
        # 从结果中提取指标（按TASKS配置的路径）
        for metric_name, metric_path in TASKS[task]["metrics"].items():
            val = results
            # 沿着路径导航到指标值（如 "results/accuracy"）
            for key in metric_path.split("/"):
                val = val[key]
            # 使用 "task/metric" 作为指标键
            metrics[f"{task}/{metric_name}"] = val
            print(f"  ✓ {task}/{metric_name}: {val}")

    # ==================== 步骤8: 保存和展示结果 ====================
    # 保存汇总的指标到JSON文件
    metrics_file = os.path.join(output_dir, "metrics.json")
    io.save(metrics_file, metrics, indent=4)
    print(f"\n指标已保存到: {metrics_file}")
    
    # 使用表格展示结果
    print("\n" + "=" * 70)
    print("评估结果汇总:")
    print("=" * 70)
    print(tabulate(metrics.items(), tablefmt="simple_outline", headers=["指标", "数值"]))
    print("=" * 70)


if __name__ == "__main__":
    main()
