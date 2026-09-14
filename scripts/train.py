"""步骤 2：用 ultralytics YOLO 训练 PCB 缺陷检测模型。

用法（cv 环境）:
    python scripts/train.py                       # 按配置训练
    python scripts/train.py --epochs 2 --batch 16 # 冒烟测试
    python scripts/train.py --model yolov8s.pt --epochs 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcbdet import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 训练入口")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="数据集 yaml（默认 <dataset_dir>/pcb.yaml）")
    parser.add_argument("--model", default=None, help="预训练权重（默认取配置）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help='GPU id 或 "cpu"')
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", default=None, help="运行名称（runs/detect/<name>）")
    parser.add_argument("--resume", default=None, help="从 last.pt 断点续训")
    parser.add_argument("--mosaic", type=float, default=None,
                        help="mosaic 增广概率；0 = 全程关闭（二值 PCB 图可尝试）")
    parser.add_argument("--close-mosaic", type=int, default=None,
                        help="最后 N 个 epoch 关闭 mosaic；0 等价于全程关闭")
    parser.add_argument("--no-hsv", action="store_true",
                        help="关闭 HSV 色彩增广（三通道差分数据集必须加，保护差分通道语义）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42；固定种子下训练完全可复现）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg.train

    data = args.data or str(cfg.dataset_dir / "pcb.yaml")
    if not Path(data).exists():
        raise FileNotFoundError(
            f"数据集配置不存在: {data}\n请先运行: python scripts/prepare_data.py"
        )

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model or t.model)

    extra = {}
    if args.mosaic is not None:
        extra["mosaic"] = args.mosaic
    # close_mosaic：CLI > 配置文件
    close_mosaic = args.close_mosaic if args.close_mosaic is not None else getattr(t, "close_mosaic", None)
    if close_mosaic is not None:
        extra["close_mosaic"] = close_mosaic
    if args.no_hsv:
        extra.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    # patience：CLI 暂不单列，读配置（默认 20，E6 消融显示 ~27 epoch 即收敛）
    patience = getattr(t, "patience", None)
    if patience is not None:
        extra["patience"] = patience

    results = model.train(
        data=data,
        epochs=args.epochs if args.epochs is not None else t.epochs,
        batch=args.batch if args.batch is not None else t.batch,
        imgsz=args.imgsz if args.imgsz is not None else t.imgsz,
        device=args.device if args.device is not None else t.device,
        workers=args.workers if args.workers is not None else t.workers,
        name=args.name or t.name,
        project=str(cfg.output_dir / "runs"),
        seed=args.seed,
        **extra,
    )
    save_dir = getattr(results, "save_dir", "runs 目录")
    print(f"\n训练完成，权重目录: {save_dir}")
    print("下一步评估示例:")
    print(f'  python scripts/evaluate.py --weights "{Path(save_dir) / "best.pt"}"')


if __name__ == "__main__":
    main()
