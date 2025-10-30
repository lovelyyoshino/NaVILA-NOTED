"""
YouTube视频批量下载工具

该脚本用于批量下载NaVILA数据集中Human部分所需的YouTube视频。
读取video_ids.txt文件中的视频ID列表，使用yt-dlp工具进行并发下载。

依赖项:
    pip install yt-dlp

使用方法:
    python download_youtube_videos.py
    
配置说明:
    - 默认下载到 /media/bigdisk/NaVILA-NOTED/NaVILA-Dataset/Human/videos 目录
    - 默认使用8个并发下载进程
    - 视频格式: MP4, 分辨率: 最高可用（优先1080p）
    - 失败的视频ID会记录到 failed_downloads.txt
"""

import os
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
import time
from datetime import datetime


# ==================== 配置参数 ====================
# 数据集根目录
DATASET_ROOT = "/media/bigdisk/NaVILA-NOTED/NaVILA-Dataset/Human"

# 视频ID列表文件路径
VIDEO_IDS_FILE = os.path.join(DATASET_ROOT, "video_ids.txt")

# 视频保存目录
OUTPUT_DIR = os.path.join(DATASET_ROOT, "videos")

# 失败记录文件
FAILED_LOG = os.path.join(DATASET_ROOT, "failed_downloads.txt")

# 并发下载进程数（根据网络带宽和CPU核心数调整）
NUM_WORKERS = 8

# 视频质量配置
# format选项说明:
# - 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
#   优先选择mp4格式的最佳视频+音频，否则选择最佳mp4，最后选择任何最佳格式
VIDEO_FORMAT = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[ext=mp4]/best"

# 下载重试次数
MAX_RETRIES = 3

# 下载超时时间（秒）
DOWNLOAD_TIMEOUT = 600  # 10分钟


def load_video_ids(file_path: str) -> List[str]:
    """
    从文件中读取YouTube视频ID列表
    
    参数:
        file_path (str): 视频ID文件路径
    
    返回:
        List[str]: 视频ID列表（过滤掉空行）
    
    异常:
        FileNotFoundError: 如果文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"视频ID文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有行，去除空白字符，过滤空行
        video_ids = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 成功读取 {len(video_ids)} 个视频ID")
    return video_ids


def check_video_exists(video_id: str, output_dir: str) -> bool:
    """
    检查视频文件是否已经存在
    
    参数:
        video_id (str): YouTube视频ID
        output_dir (str): 视频保存目录
    
    返回:
        bool: 如果视频已存在返回True
    """
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    return os.path.exists(video_path)


def download_video(args: Tuple[str, str, int]) -> Tuple[str, bool, str]:
    """
    下载单个YouTube视频
    
    参数:
        args (Tuple): 包含 (video_id, output_dir, retry_count) 的元组
            - video_id (str): YouTube视频ID
            - output_dir (str): 视频保存目录
            - retry_count (int): 当前重试次数
    
    返回:
        Tuple[str, bool, str]: (视频ID, 是否成功, 错误信息)
    
    功能说明:
        使用yt-dlp下载视频，支持以下特性：
        1. 自动选择最佳质量（最高1080p）
        2. 合并视频和音频轨道
        3. 转换为MP4格式
        4. 限制文件大小和下载速度（可选）
    """
    video_id, output_dir, retry_count = args
    
    # 视频URL
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # 输出文件路径模板
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
    
    # 构建yt-dlp命令
    # 命令选项说明:
    # -f: 指定视频格式
    # --merge-output-format: 合并后的输出格式
    # -o: 输出文件名模板
    # --no-playlist: 不下载播放列表，只下载单个视频
    # --no-warnings: 不显示警告信息
    # --no-check-certificate: 跳过SSL证书验证（某些地区需要）
    # --socket-timeout: 套接字超时时间
    # --retries: 下载重试次数
    # --fragment-retries: 片段重试次数
    # --quiet: 静默模式（减少输出）
    # --no-progress: 不显示进度条
    cmd = [
        "yt-dlp",
        "-f", VIDEO_FORMAT,
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        "--socket-timeout", str(DOWNLOAD_TIMEOUT),
        "--retries", str(MAX_RETRIES),
        "--fragment-retries", str(MAX_RETRIES),
        "--quiet",
        "--no-progress",
        video_url
    ]
    
    try:
        # 执行下载命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
            check=True
        )
        
        # 验证文件是否成功创建
        expected_file = os.path.join(output_dir, f"{video_id}.mp4")
        if os.path.exists(expected_file):
            file_size = os.path.getsize(expected_file) / (1024 * 1024)  # 转换为MB
            print(f"✓ [{video_id}] 下载成功 ({file_size:.2f} MB)")
            return (video_id, True, "")
        else:
            error_msg = "文件未创建"
            print(f"✗ [{video_id}] 下载失败: {error_msg}")
            return (video_id, False, error_msg)
            
    except subprocess.TimeoutExpired:
        error_msg = f"下载超时 (>{DOWNLOAD_TIMEOUT}秒)"
        print(f"✗ [{video_id}] {error_msg}")
        return (video_id, False, error_msg)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"yt-dlp错误: {e.stderr.strip() if e.stderr else str(e)}"
        
        # 检查是否是视频不可用的错误
        if "unavailable" in error_msg.lower() or "private" in error_msg.lower():
            print(f"✗ [{video_id}] 视频不可用（可能已删除或设为私密）")
        elif "copyright" in error_msg.lower():
            print(f"✗ [{video_id}] 版权限制")
        else:
            print(f"✗ [{video_id}] 下载失败: {error_msg[:100]}")
        
        return (video_id, False, error_msg)
        
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        print(f"✗ [{video_id}] {error_msg}")
        return (video_id, False, error_msg)


def save_failed_list(failed_videos: List[Tuple[str, str]], log_file: str):
    """
    将失败的视频ID和错误信息保存到文件
    
    参数:
        failed_videos (List[Tuple[str, str]]): 失败的视频列表 [(video_id, error_msg), ...]
        log_file (str): 日志文件路径
    """
    if not failed_videos:
        return
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# 下载失败的视频列表 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总计: {len(failed_videos)} 个失败\n\n")
        
        for video_id, error_msg in failed_videos:
            f.write(f"{video_id}\t{error_msg}\n")
    
    print(f"\n失败列表已保存到: {log_file}")


def main():
    """
    主函数：批量下载YouTube视频
    
    工作流程:
        1. 检查依赖和目录
        2. 读取视频ID列表
        3. 过滤已下载的视频
        4. 使用多进程并发下载
        5. 记录失败的视频
        6. 显示统计信息
    """
    
    print("=" * 70)
    print("YouTube视频批量下载工具 - NaVILA数据集")
    print("=" * 70)
    print()
    
    # ==================== 1. 检查依赖 ====================
    print("[ 1/6 ] 检查依赖...")
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ yt-dlp 版本: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ 错误: 未找到 yt-dlp")
        print("请安装: pip install yt-dlp")
        return
    
    # ==================== 2. 创建输出目录 ====================
    print("\n[ 2/6 ] 创建输出目录...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ 输出目录: {OUTPUT_DIR}")
    
    # ==================== 3. 读取视频ID列表 ====================
    print("\n[ 3/6 ] 读取视频ID列表...")
    try:
        video_ids = load_video_ids(VIDEO_IDS_FILE)
    except Exception as e:
        print(f"✗ 错误: {e}")
        return
    
    # ==================== 4. 过滤已下载的视频 ====================
    print("\n[ 4/6 ] 检查已存在的视频...")
    existing_videos = [vid for vid in video_ids if check_video_exists(vid, OUTPUT_DIR)]
    videos_to_download = [vid for vid in video_ids if not check_video_exists(vid, OUTPUT_DIR)]
    
    print(f"✓ 已存在: {len(existing_videos)} 个视频")
    print(f"✓ 待下载: {len(videos_to_download)} 个视频")
    
    if not videos_to_download:
        print("\n所有视频已下载完成！")
        return
    
    # ==================== 5. 开始并发下载 ====================
    print(f"\n[ 5/6 ] 开始下载 (并发数: {NUM_WORKERS})...")
    print("-" * 70)
    
    start_time = time.time()
    
    # 准备下载任务参数
    download_args = [(vid, OUTPUT_DIR, 0) for vid in videos_to_download]
    
    # 使用进程池并发下载
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(download_video, download_args)
    
    # ==================== 6. 统计和保存结果 ====================
    print("\n" + "-" * 70)
    print("[ 6/6 ] 处理结果...")
    
    # 分类结果
    successful = [(vid, err) for vid, success, err in results if success]
    failed = [(vid, err) for vid, success, err in results if not success]
    
    # 保存失败列表
    if failed:
        save_failed_list(failed, FAILED_LOG)
    
    # 显示统计信息
    elapsed_time = time.time() - start_time
    total_downloaded = len(existing_videos) + len(successful)
    
    print("\n" + "=" * 70)
    print("下载完成统计")
    print("=" * 70)
    print(f"总视频数:     {len(video_ids)}")
    print(f"已有视频:     {len(existing_videos)}")
    print(f"本次下载:     {len(videos_to_download)}")
    print(f"  - 成功:     {len(successful)}")
    print(f"  - 失败:     {len(failed)}")
    print(f"当前总计:     {total_downloaded} / {len(video_ids)}")
    print(f"完成度:       {total_downloaded / len(video_ids) * 100:.1f}%")
    print(f"用时:         {elapsed_time / 60:.1f} 分钟")
    print("=" * 70)
    
    if failed:
        print(f"\n⚠ 有 {len(failed)} 个视频下载失败")
        print(f"失败列表已保存到: {FAILED_LOG}")
        print("你可以稍后重新运行此脚本来重试失败的视频")
    else:
        print("\n✓ 所有视频下载成功！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断下载")
        print("下次运行脚本时会自动跳过已下载的视频")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()

