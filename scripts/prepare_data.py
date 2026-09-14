"""步骤 1：把 DeepPCB 原始数据转换为 YOLO 格式并生成 pcb.yaml。

用法（cv 环境）:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --out D:/somewhere/pcb   # 输出到别处
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcbdet import load_config
from pcbdet.convert_yolo import build_dataset, build_diff_dataset, write_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepPCB -> YOLO 数据转换")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--out", default=None, help="输出目录（默认取配置 dataset_dir）")
    parser.add_argument("--force", action="store_true", help="输出目录已存在时仍重新复制")
    parser.add_argument("--diff", action="store_true",
                        help="构建 E6 三通道数据集（待检/模板/差分），输出到 <dataset_dir> 的同级 pcb_diff/")
    parser.add_argument("--diff-out", default=None, help="三通道数据集输出目录")
    parser.add_argument("--data-root", default=None,
                        help="DeepPCB 数据根目录（含 PCBData/）；也可用环境变量 DEEPPCB_ROOT")
    args = parser.parse_args()

    cfg = load_config(args.config, data_root=args.data_root)
    cfg.check_data_root()

    if args.diff:
        out_dir = Path(args.diff_out) if args.diff_out else cfg.dataset_dir.parent / "pcb_diff"
        marker = out_dir / "pcb.yaml"
        if marker.exists() and not args.force:
            print(f"检测到已构建的三通道数据集: {marker}\n如需重建请加 --force")
            return
        build_diff_dataset(cfg, out_dir)
        print("完成。训练示例: python scripts/train.py --data "
              f"{(out_dir / 'pcb.yaml')} --no-hsv --name n_diff")
        return

    out_dir = Path(args.out) if args.out else cfg.dataset_dir

    marker = out_dir / "pcb.yaml"
    if marker.exists() and not args.force:
        # 不重新复制图片，但刷新 pcb.yaml —— 目录移动/改名后 path 字段仍然有效
        yaml_path = write_dataset_yaml(out_dir, cfg.class_names)
        print(f"检测到已转换的数据集，已刷新数据集配置: {yaml_path}\n如需重建请加 --force")
        return

    stats = build_dataset(cfg, out_dir)
    for split, s in stats["splits"].items():
        per = "  ".join(f"{k}={v}" for k, v in s["boxes_per_class"].items())
        print(f"  [{split}] images={s['images']} boxes={s['boxes_total']} | {per}")
    print("完成。下一步: python scripts/train.py")


if __name__ == "__main__":
    main()
