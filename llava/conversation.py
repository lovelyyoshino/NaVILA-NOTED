# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

"""
对话管理模块

该模块定义了多模态对话系统的核心数据结构和模板。
主要功能包括：
- 管理用户和助手之间的对话历史
- 支持多种LLM的对话格式（LLaMA、Mistral、MPT等）
- 处理多模态输入（文本+图像/视频）
- 提供不同模型的对话模板和分隔符样式
"""

import dataclasses
from enum import Enum, auto
from typing import List

from llava.utils.logging import logger


class SeparatorStyle(Enum):
    """
    对话分隔符样式枚举
    
    定义了不同语言模型使用的对话格式分隔符样式：
    - AUTO: 自动检测
    - SINGLE: 单一分隔符（如 "###"）
    - TWO: 两个不同的分隔符（用户/助手消息使用不同分隔符）
    - MPT: MPT模型特定格式
    - PLAIN: 纯文本格式，无特殊分隔符
    - LLAMA_2: LLaMA 2模型的对话格式
    - MISTRAL: Mistral模型的对话格式
    - LLAMA_3: LLaMA 3模型的对话格式（使用特殊标记）
    """

    AUTO = auto()       # 自动选择
    SINGLE = auto()     # 单分隔符模式
    TWO = auto()        # 双分隔符模式  
    MPT = auto()        # MPT模型格式
    PLAIN = auto()      # 纯文本格式
    LLAMA_2 = auto()    # LLaMA 2格式
    MISTRAL = auto()    # Mistral格式
    LLAMA_3 = auto()    # LLaMA 3格式


@dataclasses.dataclass
class Conversation:
    """
    对话历史管理类
    
    维护完整的对话历史记录，并根据不同模型的格式要求生成提示文本。
    
    属性:
        system (str): 系统提示词（给模型的初始指令）
        roles (List[str]): 对话角色列表，通常是 ["用户", "助手"] 或 ["USER", "ASSISTANT"]
        messages (List[List[str]]): 对话消息列表，每条消息是 [角色, 内容] 的形式
        offset (int): 消息偏移量，用于跳过某些初始消息
        sep_style (SeparatorStyle): 分隔符样式
        sep (str): 主分隔符字符串（默认 "###"）
        sep2 (str): 次分隔符字符串（用于TWO模式）
        version (str): 对话模板版本标识
        skip_next (bool): 是否跳过下一条消息（内部使用）
    
    使用示例:
        conv = Conversation(
            system="你是一个有帮助的助手。",
            roles=["用户", "助手"],
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_3
        )
        conv.append_message("用户", "你好")
        conv.append_message("助手", "你好！有什么可以帮助你的吗？")
        prompt = conv.get_prompt()
    """

    system: str                                      # 系统提示词
    roles: List[str]                                 # 角色列表 [用户角色, 助手角色]
    messages: List[List[str]]                        # 消息历史 [[角色, 内容], ...]
    offset: int                                      # 消息偏移量
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE  # 分隔符样式
    sep: str = "###"                                 # 主分隔符
    sep2: str = None                                 # 次分隔符
    version: str = "Unknown"                         # 模板版本

    skip_next: bool = False                          # 是否跳过下一条消息

    def get_prompt(self):
        """
        生成模型输入的提示文本
        
        根据配置的分隔符样式，将对话历史格式化为模型可以理解的提示文本。
        不同的语言模型需要不同的格式。
        
        返回:
            str: 格式化后的提示文本
        
        支持的格式样式:
            - SINGLE: 使用单一分隔符，格式如 "角色: 消息###"
            - TWO: 使用两个分隔符交替，适用于Vicuna等模型
            - LLAMA_3: LLaMA 3的特殊格式，使用header标记
            - MPT: MPT模型的格式
            - LLAMA_2/MISTRAL: 使用[INST]标记的格式
            - PLAIN: 纯文本格式，无角色标记
        
        特殊处理:
            - 如果消息包含图像（tuple类型），会正确处理图像标记
            - 支持mmtag版本的特殊图像标记格式
        """
        messages = self.messages
        # 处理包含图像的第一条消息
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            # mmtag版本：使用XML风格的图像标记
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            # 标准版本：在文本前加上<image>标记
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        # SINGLE样式：单一分隔符，格式 "角色: 消息###"
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        
        # TWO样式：双分隔符交替，用户和助手消息使用不同分隔符
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        
        # LLAMA_3样式：使用LLaMA 3的特殊header标记
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            ret = self.system + self.sep
            for rid, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    # 最后一条消息使用sep2（通常是<|end_of_text|>），其他使用sep
                    sep = self.sep if rid < len(messages) - 1 else self.sep2
                    ret += role + message + sep
                else:
                    ret += role
        
        # MPT样式：MPT模型的格式，使用<|im_start|>和<|im_end|>标记
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        
        # LLAMA_2和MISTRAL样式：使用[INST]指令标记的格式
        elif self.sep_style == SeparatorStyle.LLAMA_2 or self.sep_style == SeparatorStyle.MISTRAL:
            # 定义系统消息包装函数
            if self.sep_style == SeparatorStyle.LLAMA_2:
                wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"  # LLaMA 2用<<SYS>>标记
            else:
                wrap_sys = lambda msg: f"{msg}" + ("\n" if msg else "")  # Mistral直接使用
            
            # 定义指令包装函数
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""
            
            # Mistral需要在开头加<s>标记
            if self.sep_style == SeparatorStyle.MISTRAL:
                ret += "<s>"

            for i, (role, message) in enumerate(messages):
                # 第一条消息必须来自用户
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    # 第一条消息包含系统提示
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    # 偶数索引（用户消息）用[INST]包装
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    # 奇数索引（助手消息）
                    else:
                        if self.sep_style == SeparatorStyle.LLAMA_2:
                            ret += " " + message + " " + self.sep2
                        else:
                            ret += message + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        
        # PLAIN样式：纯文本格式，不添加角色标记
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        """
        向对话历史添加一条新消息
        
        参数:
            role (str): 消息的角色（通常是self.roles中的一个）
            message: 消息内容，可以是：
                - str: 纯文本消息
                - tuple: (文本, 图像, 处理模式) 的三元组
        
        使用示例:
            conv.append_message(conv.roles[0], "你好")  # 用户消息
            conv.append_message(conv.roles[1], "你好！")  # 助手消息
        """
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        """
        从对话历史中提取所有图像
        
        参数:
            return_pil (bool): 
                - True: 返回PIL Image对象
                - False: 返回base64编码的字符串
        
        返回:
            List: 图像列表（PIL Image或base64字符串）
        
        图像处理模式:
            - "Pad": 将图像填充为正方形（背景色：122,116,104）
            - "Default"/"Crop": 保持原样
            - "Resize": 调整大小到336x336
        
        注意:
            - 只处理偶数索引的消息（通常是用户消息）
            - 图像会被调整大小以控制最大/最小边长
        """
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    from PIL import Image

                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":

                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        """
        将对话历史转换为Gradio聊天界面格式
        
        返回:
            List[List]: Gradio chatbot格式的消息列表
                每个元素是 [用户消息, 助手消息] 的形式
        
        功能:
            - 将图像转换为HTML <img>标签（base64编码）
            - 调整图像大小以适合显示
            - 组织消息为问答对的形式
        
        用于:
            在Gradio Web界面中显示对话历史
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        """
        创建当前对话对象的深拷贝
        
        返回:
            Conversation: 新的对话对象，包含相同的配置和消息历史
        
        用途:
            - 创建对话分支
            - 保存对话状态
            - 避免修改原始对话
        """
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        """
        将对话对象转换为字典格式
        
        返回:
            dict: 包含对话配置和历史的字典
        
        字典字段:
            - system: 系统提示词
            - roles: 角色列表
            - messages: 消息历史
            - offset: 消息偏移量
            - sep: 主分隔符
            - sep2: 次分隔符
        
        注意:
            如果消息包含图像（tuple类型），只保留文本部分
        """
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


# ==================== 预定义对话模板 ====================
# 以下是各种模型和场景的预定义对话模板

conv_auto = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.AUTO,
    sep="\n",
)
"""自动检测模板：会根据模型名称自动选择合适的对话格式"""

conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
"""Vicuna v0模板：早期版本，包含示例对话，使用单分隔符"""

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
"""Vicuna v1模板：标准版本，使用双分隔符，适合通用对话"""

# kentang-mit@: This conversation template is designed for SFT on VFLAN.
conv_vicuna_v1_nosys = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="v1_nosys",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
"""Vicuna v1 (无系统提示)模板：专为VFLAN数据集的SFT设计"""

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
"""LLaMA 2模板：Meta官方LLaMA 2的对话格式，包含安全和道德指导"""

conv_mistral = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="mistral",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="",
    sep2="</s>",
)
"""Mistral模板：Mistral AI的对话格式，使用[INST]标记"""

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
"""LLaVA-LLaMA 2模板：结合LLaMA 2和视觉能力的多模态对话格式"""

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)
"""MPT模板：MosaicML的MPT模型对话格式"""

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)
"""LLaVA Plain模板：纯文本格式，无角色标记，适合某些特殊场景"""

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
"""LLaVA v0模板：LLaVA的早期版本对话格式"""

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)
"""LLaVA v0 MMTag模板：使用XML风格的<Image>标记来标识视觉内容"""

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)
"""LLaVA v1模板：LLaVA的标准版本，广泛使用"""


conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)
"""LLaVA v1 MMTag模板：LLaVA v1的MMTag版本"""

hermes_2 = Conversation(
    system="<|im_start|>system\nAnswer the questions.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
    messages=(),
    offset=0,
    version="hermes-2",
)
"""Hermes 2模板：Nous Research的Hermes 2模型对话格式"""


# Template added by Yukang. Note (kentang-mit@): sep is <|eot_id|> for official template.
llama_3_chat = Conversation(
    system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama_v3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
    sep2="<|end_of_text|>",
)
"""
LLaMA 3对话模板：Meta LLaMA 3的官方对话格式
使用特殊的header标记：<|start_header_id|>、<|end_header_id|>
结束标记：<|eot_id|>（end of turn）、<|end_of_text|>
"""


# 默认对话模板（会被auto_set_conversation_mode自动设置）
default_conversation = conv_auto

# 对话模板注册表：模板名称到模板对象的映射
conv_templates = {
    "auto": conv_auto,
    "default": conv_vicuna_v0,
    "hermes-2": hermes_2,
    "llama_3": llama_3_chat,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "vicuna_v1_nosys": conv_vicuna_v1_nosys,
    "llama_2": conv_llama_2,
    "mistral": conv_mistral,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "mpt": conv_mpt,
}
"""
对话模板注册表

包含所有可用的对话模板，可以通过名称快速访问。
在训练和推理时，通过指定template名称来选择对话格式。
"""


# 模型名称到对话模式的映射
CONVERSATION_MODE_MAPPING = {
    "vila1.5-3b": "vicuna_v1",
    "vila1.5-8b": "llama_3",
    "vila1.5-13b": "vicuna_v1",
    "vila1.5-40b": "hermes-2",
    "llama-2": "llava_llama_2",
    "llama2": "llava_llama_2",
    "llama-3": "llama_3",
    "llama3": "llama_3",
    "mpt": "mpt",
}
"""
模型名称到对话模式的映射表

用于根据模型名称自动选择合适的对话模板。
例如：
- "vila1.5-8b" -> "llama_3"（使用LLaMA 3格式）
- "vila1.5-40b" -> "hermes-2"（使用Hermes 2格式）
- "llama-3-xxx" -> "llama_3"（使用LLaMA 3格式）
"""


def auto_set_conversation_mode(model_name_or_path: str) -> str:
    """
    根据模型名称自动设置对话模式
    
    参数:
        model_name_or_path (str): 模型名称或路径
            例如: "a8cheng/navila-siglip-llama3-8b-v1.5-pretrain"
    
    功能:
        1. 从模型名称中提取关键词（如"llama3"、"vila1.5-8b"等）
        2. 在CONVERSATION_MODE_MAPPING中查找匹配项
        3. 设置全局的default_conversation为对应的模板
    
    异常:
        ValueError: 如果无法从模型名称中识别出有效的对话模式
    
    使用示例:
        auto_set_conversation_mode("vila1.5-8b-pretrain")
        # 会自动设置为llama_3对话模式
    
    注意:
        - 这个函数会修改全局变量default_conversation
        - 在加载模型时会自动调用
        - 匹配是大小写不敏感的
    """
    global default_conversation
    
    # 遍历映射表，查找匹配的模型关键词
    for k, v in CONVERSATION_MODE_MAPPING.items():
        if k in model_name_or_path.lower():
            logger.info(f"Setting conversation mode to `{v}` based on model name/path `{model_name_or_path}`.")
            default_conversation = conv_templates[v]
            return
    
    # 如果没有找到匹配项，抛出异常
    raise ValueError(f"Cannot find a valid conversation mode for model name/path `{model_name_or_path}`.")
