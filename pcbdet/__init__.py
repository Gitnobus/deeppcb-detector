"""DeepPCB demo 核心包：数据转换 / YOLO 训练辅助 / 官方协议评估。"""
from .config import Config, load_config, PROJECT_ROOT
from .convert_yolo import build_dataset, build_diff_dataset, load_split

__all__ = ["Config", "load_config", "PROJECT_ROOT", "build_dataset", "build_diff_dataset", "load_split"]
__version__ = "0.1.0"
