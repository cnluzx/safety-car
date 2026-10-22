import os
import sys
import re

# ======= 核心防卡死补丁：强制 pygame 进入无头（无图形）模式 =======
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "alsa"

import pygame

class Broadcaster:
    def __init__(self):
        # 1. 音频目录：默认 <项目根>/sound，可用 SOUND_DIR 环境变量覆盖
        self.base_path = os.environ.get("SOUND_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sound"
        )
        
        # 2. 建立 1 到 8 与音频文件的精确映射字典
        # 如果您的实际音频顺序与下方不同，您可以直接在这里调整文件名
        self.audio_mapping = {
            1: "free-speech.mp3",      # 对应 1
            2: "free-speech (1).mp3",  # 对应 2
            3: "free-speech (2).mp3",  # 对应 3
            4: "free-speech (3).mp3",  # 对应 4
            5: "free-speech (4).mp3",  # 对应 5
            6: "free-speech (5).mp3",  # 对应 6
            7: "free-speech (6).mp3",  # 对应 7
            8: "free-speech (7).mp3",  # 对应 8
            9: "free-speech (8).mp3",  # 对应 9
            10: "free-speech (9).mp3", # 对应 10
            11: "free-speech (10).mp3", # 对应 11
            12: "free-speech (11).mp3",  # 对应 12
        }

        # YOLO业务ID：1～10为嫌疑人，11～20为刀具。
        self.target_audio_mapping = {
            **{
                target_id: f"识别到{target_id}号嫌疑人.mp3"
                for target_id in range(1, 11)
            },
            **{
                target_id: f"发现物体{target_id - 10}号刀具.mp3"
                for target_id in range(11, 21)
            },
        }
        self.no_target_audio = "yolo_none.mp3"
        
        self.is_mock = False
        
        try:
            pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=4096)
            pygame.mixer.init()
            print(f"[Audio] 语音播报模块初始化完成 (音频目录: {self.base_path})")
        except Exception as e:
            print(f"[Audio WARN] 找不到声卡硬件或驱动冲突，进入【模拟静默播报】模式: {e}")
            self.is_mock = True
            
    def _get_file_by_id_or_text(self, input_val):
        """
        辅助函数：从输入的数字或中文字符串中提取数字 (1-8)，并返回对应的完整音频文件路径
        """
        target_id = None
        
        # 如果传入的是整型或浮点型
        if isinstance(input_val, (int, float)):
            target_id = int(input_val)
        else:
            # 如果传入的是字符串（如 "到达3号巡逻点，表示播报成功"），使用正则提取其中的数字
            numbers = re.findall(r'\d+', str(input_val))
            if numbers:
                target_id = int(numbers[0])
        
        # 校验提取出的 ID 是否在 1 到 8 的映射范围内
        if target_id in self.audio_mapping:
            filename = self.audio_mapping[target_id]
            return os.path.join(self.base_path, filename)
        
        return None

    def play_file(self, path):
        """播放指定路径的音频"""
        if self.is_mock:
            print(f"[Audio 模拟发声] 正在播放: {path}")
            return

        if os.path.exists(path):
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"[Audio Error] 播放出错: {e}")
        else:
            print(f"[Audio Error] 音频文件不存在，请检查路径: {path}")

    def say_point(self, point_id):
        """报站：到达某巡逻点 (支持直接传入 ID 或 字符串)"""
        path = self._get_file_by_id_or_text(point_id)
        if path:
            print(f"[Audio] 准备播报巡逻点音频: {os.path.basename(path)}")
            self.play_file(path)
        else:
            print(f"[Audio Error] 无法匹配巡逻点 {point_id} 的音频文件")

    def say_result(self, target_name):
        """兼容旧接口；目标识别统一交给 say_target 处理。"""
        if isinstance(target_name, (int, float)):
            self.say_target(int(target_name))
            return
        numbers = re.findall(r"\d+", str(target_name))
        self.say_target(int(numbers[0]) if numbers else -1)

    def say_target(self, target_id):
        """播报YOLO目标；-1或None表示未识别到目标。"""
        if target_id in (-1, None):
            filename = self.no_target_audio
            description = "未识别到目标"
        else:
            try:
                normalized = int(target_id)
            except (TypeError, ValueError):
                print(f"[Audio Error] 无效目标ID: {target_id}")
                return
            filename = self.target_audio_mapping.get(normalized)
            if filename is None:
                print(f"[Audio Error] 没有业务ID {normalized} 的目标音频")
                return
            description = (
                f"{normalized}号嫌疑人"
                if normalized <= 10
                else f"{normalized - 10}号刀具"
            )

        path = os.path.join(self.base_path, filename)
        print(f"[Audio] 播报目标: {description} ({filename})")
        self.play_file(path)
