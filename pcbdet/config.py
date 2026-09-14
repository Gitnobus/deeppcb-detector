"""配置加载与路径解析。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

DATA_ROOT_ENV = "DEEPPCB_ROOT"  # DeepPCB 数据集根目录的环境变量覆盖

DOWNLOAD_HINT = (
    "DeepPCB 数据集未随本仓库分发，请先获取：\n"
    "  git clone https://github.com/tangsanli5201/DeepPCB\n"
    f"然后通过以下任一方式指定数据根目录（即包含 PCBData/ 的目录）：\n"
    f"  1. 设置环境变量 {DATA_ROOT_ENV}=<数据集根目录>\n"
    "  2. 命令行加 --data-root <数据集根目录>\n"
    "  3. 修改 configs/config.yaml 的 data_root"
)


class Config:
    """对 config.yaml 的轻量封装，支持属性访问与相对路径解析。"""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT,
                 data_root_override: str | Path | None = None):
        self._data = data
        self._root = root
        self._data_root_override = data_root_override

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as e:
            raise AttributeError(f"配置项不存在: {key}") from e
        # 嵌套节（如 train/eval/app）递归包装，支持 cfg.train.model 链式访问
        if isinstance(value, dict):
            return Config(value, self._root)
        return value

    def resolve(self, p: str | Path) -> Path:
        """相对路径 -> 相对项目根目录的绝对路径；绝对路径原样返回。"""
        p = Path(p)
        return p if p.is_absolute() else (self._root / p).resolve()

    @property
    def data_root(self) -> Path:
        """DeepPCB 数据根目录。优先级：--data-root 参数 > 环境变量 DEEPPCB_ROOT > 配置文件。"""
        if self._data_root_override:
            raw: Any = self._data_root_override
        elif os.environ.get(DATA_ROOT_ENV):
            raw = os.environ[DATA_ROOT_ENV]
        else:
            raw = self._data.get("data_root", "./DeepPCB")
        return self.resolve(raw)

    def check_data_root(self) -> Path:
        """校验数据根目录可用（含 PCBData/），不可用则给出下载指引后退出。"""
        root = self.data_root
        if (root / "PCBData").is_dir():
            return root
        raise SystemExit(
            f"数据根目录无效: {root}\n（未找到 PCBData/ 子目录）\n\n{DOWNLOAD_HINT}"
        )

    @property
    def dataset_dir(self) -> Path:
        return self.resolve(self._data["dataset_dir"])

    @property
    def output_dir(self) -> Path:
        return self.resolve(self._data["output_dir"])

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def load_config(path: str | Path | None = None,
                data_root: str | Path | None = None) -> Config:
    """加载配置。data_root 为 CLI --data-root 传入的覆盖值（优先级最高）。"""
    p = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config(data, data_root_override=data_root)
