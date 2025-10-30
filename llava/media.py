"""
媒体类型定义模块

定义了处理多模态数据的基础类层次结构：
- Media: 所有媒体类型的基类
- File: 文件类型的基类
- Image: 图像文件类型
- Video: 视频文件类型

这些类用于在模型输入中表示不同类型的媒体数据。
"""

__all__ = ["Media", "File", "Image", "Video"]


class Media:
    """
    媒体基类
    
    所有媒体类型（图像、视频等）的抽象基类。
    用于类型检查和多态处理。
    """
    pass


class File(Media):
    """
    文件媒体类
    
    表示基于文件的媒体数据。
    
    属性:
        path (str): 文件的路径（可以是本地路径或URL）
    """
    def __init__(self, path: str) -> None:
        """
        初始化文件媒体对象
        
        参数:
            path (str): 文件路径
        """
        self.path = path


class Image(File):
    """
    图像文件类
    
    表示图像类型的媒体数据。
    继承自File类，用于标识图像输入。
    
    使用示例:
        image = Image("/path/to/image.jpg")
        image = Image("https://example.com/image.png")
    """
    pass


class Video(File):
    """
    视频文件类
    
    表示视频类型的媒体数据。
    继承自File类，用于标识视频输入。
    
    使用示例:
        video = Video("/path/to/video.mp4")
        video = Video("https://example.com/video.mp4")
    """
    pass
