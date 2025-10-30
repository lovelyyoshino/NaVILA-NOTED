"""
视频帧提取工具

该脚本用于从视频文件中提取原始帧图像，主要用于处理NaVILA数据集中的Human视频数据。
使用ffmpeg作为底层工具进行帧提取，支持多进程并行处理以提高效率。
"""

import multiprocessing.dummy as mp  # 导入线程池模块（伪多进程）
import re  # 导入正则表达式模块
import subprocess  # 导入子进程模块，用于调用外部命令
from os import listdir, mkdir  # 导入文件系统操作函数


def extract_frames(videopath, dest, fps=1):
    """
    从指定的视频文件中提取帧图像
    
    参数:
        videopath (str): 视频文件的完整路径
        dest (str): 提取帧的目标保存目录
        fps (int): 每秒提取的帧数，默认为1（即每秒提取1帧）
    
    功能说明:
        1. 创建目标目录（如果不存在）
        2. 使用ffmpeg命令行工具提取视频帧
        3. 帧图像以4位数字命名（如0001.jpg, 0002.jpg等）保存为JPEG格式
    """
    
    try:
        mkdir(dest)
        print("creating " + dest + " subdirectory")
    except:
        print(dest + " subdirectory already exists")

    # 调用ffmpeg命令提取视频帧
    # -i: 输入视频文件
    # -vf fps=N: 设置视频过滤器，按指定fps提取帧
    # %04d.jpg: 输出文件名格式，4位数字编号
    output = subprocess.call(
        [
            "ffmpeg",
            "-i",
            videopath,
            "-vf",
            "fps=" + str(fps),
            dest + "/%04d.jpg",
        ]
    )
    if output:
        print("Failed to extract frames")


def extract_all_frames():
    """
    批量处理所有视频文件并提取帧
    
    功能说明:
        1. 创建总的帧存储根目录
        2. 扫描视频目录中的所有视频文件
        3. 使用多线程池并行处理多个视频
        4. 为每个视频创建独立的子目录存储其提取的帧
    
    注意事项:
        - 需要将 /PATH_TO_DATA 替换为实际的数据路径
        - 使用8个并行进程同时处理，可根据CPU核心数调整
        - 如果视频已经处理过，会自动跳过
    """
    
    # 创建帧存储的根目录
    try:
        mkdir("/media/bigdisk/NaVILA-NOTED/NaVILA-Dataset/Human/raw_frames", exist_ok=True)
        print("creating frames subdirectory")
    except:
        print("frames subdirectory already exists")
    
    # 获取所有视频文件列表
    videos = listdir("/media/bigdisk/NaVILA-NOTED/NaVILA-Dataset/Human/videos")

    def eaf(vid):
        """
        单个视频的提取处理函数（extract_and_frame的缩写）
        
        参数:
            vid (str): 视频文件名（如 "xxx.mp4"）
        
        功能:
            1. 从文件名中提取视频ID（去掉.mp4后缀）
            2. 为该视频创建独立的帧存储子目录
            3. 调用extract_frames进行实际的帧提取
            4. 如果目录已存在则跳过（避免重复处理）
        """
        vid_id = re.match("(.*).mp4", vid)[1]  # 使用正则提取视频ID
        subdir = "/media/bigdisk/NaVILA-NOTED/NaVILA-Dataset/Human/raw_frames/" + vid_id
        try:
            mkdir(subdir)
            # 以1fps的速率提取帧（每秒1帧）
            extract_frames("/media/bigdisk/NaVILA-NOTED/NaVILA-Dataset/Human/videos/" + vid, subdir, fps=1)
        except FileExistsError:
            print(f"skipping {vid}")  # 目录已存在，跳过该视频

    # 创建视频列表
    vids = [vid for vid in videos]
    
    # 创建线程池，使用8个并行工作进程
    p = mp.Pool(processes=8)
    
    # 并行处理所有视频
    p.map(eaf, vids)
    
    # 关闭线程池并等待所有任务完成
    p.close()
    p.join()


if __name__ == "__main__":
    # 脚本主入口：执行批量帧提取
    extract_all_frames()